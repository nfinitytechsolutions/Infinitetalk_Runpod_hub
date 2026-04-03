import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import librosa
import shutil
import math
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S2V pipeline constants
S2V_FPS = 16
S2V_CHUNK_LENGTH = 77  # frames per chunk (~4.8s at 16fps)


def truncate_base64_for_log(base64_str, max_length=50):
    if not base64_str:
        return "None"
    if len(base64_str) <= max_length:
        return base64_str
    return f"{base64_str[:max_length]}... (총 {len(base64_str)} 문자)"


server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())


def download_file_from_url(url, output_path):
    try:
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=30", url],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info(f"URL download success: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"wget download failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("Download timeout")
    except Exception as e:
        raise Exception(f"Download error: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, "wb") as f:
            f.write(decoded_data)
        logger.info(f"Base64 saved to '{file_path}'")
        return file_path
    except (binascii.Error, ValueError) as e:
        raise Exception(f"Base64 decode failed: {e}")


def process_input(input_data, temp_dir, output_filename, input_type):
    if input_type == "path":
        return input_data
    elif input_type == "url":
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")

    logger.info(f"Workflow node count: {len(prompt)}")

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"Prompt queued: {result}")
        return result
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error: {e.code} - {e.reason}")
        logger.error(f"Response: {e.read().decode('utf-8')}")
        raise
    except Exception as e:
        logger.error(f"Queue error: {e}")
        raise


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    logger.info(f"Workflow started: prompt_id={prompt_id}")

    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is not None:
                    logger.info(f"Executing node: {data['node']}")
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    logger.info("Workflow complete")
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]

    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        videos_output = []
        if "gifs" in node_output:
            for video in node_output["gifs"]:
                video_path = video["fullpath"]
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    logger.info(f"Video found: {video_path} ({file_size} bytes)")
                videos_output.append(video_path)
        output_videos[node_id] = videos_output

    return output_videos


def load_workflow(workflow_path):
    with open(workflow_path, "r") as file:
        return json.load(file)


def get_audio_duration(audio_path):
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.warning(f"Audio duration failed ({audio_path}): {e}")
        return None


def build_s2v_workflow(prompt, num_chunks):
    """Dynamically extend the S2V workflow with additional chunks.

    The base workflow in S2V_single.json has:
    - Nodes 1-12: loaders, encoders, WanSoundImageToVideo (initial chunk)
    - Node 13: KSampler for chunk 1
    - Node 30: VAEDecode
    - Node 31: VHS_VideoCombine

    For num_chunks > 1, we insert WanSoundImageToVideoExtend + KSampler pairs
    between the initial KSampler (node 13) and VAEDecode (node 30).
    """
    if num_chunks <= 1:
        # Single chunk — no extension needed, base workflow works as-is
        return prompt

    # We'll chain: chunk1(13) -> extend1(100)+sample1(101) -> extend2(102)+sample2(103) -> ...
    # The last KSampler output feeds into VAEDecode (node 30)

    prev_latent_node = "13"  # KSampler output from chunk 1

    for i in range(1, num_chunks):
        extend_node_id = str(100 + (i - 1) * 2)
        sampler_node_id = str(101 + (i - 1) * 2)

        # WanSoundImageToVideoExtend node
        prompt[extend_node_id] = {
            "inputs": {
                "length": S2V_CHUNK_LENGTH,
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["7", 0],
                "video_latent": [prev_latent_node, 0],
                "audio_encoder_output": ["10", 0],
                "ref_image": ["11", 0],
            },
            "class_type": "WanSoundImageToVideoExtend",
            "_meta": {"title": f"S2V Extend - Chunk {i + 1}"},
        }

        # KSampler for this extension chunk
        prompt[sampler_node_id] = {
            "inputs": {
                "seed": 1,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": [extend_node_id, 0],
                "negative": [extend_node_id, 1],
                "latent_image": [extend_node_id, 2],
            },
            "class_type": "KSampler",
            "_meta": {"title": f"KSampler - Chunk {i + 1}"},
        }

        prev_latent_node = sampler_node_id

    # Update VAEDecode to read from the last KSampler
    prompt["30"]["inputs"]["samples"] = [prev_latent_node, 0]

    return prompt


def _run_comfyui_job(prompt):
    """Connect to ComfyUI via WebSocket, queue a prompt, and return the output video path."""
    import time

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    http_url = f"http://{server_address}:8188/"

    # HTTP health check (max 3 min)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connected (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(
                f"HTTP failed (attempt {http_attempt+1}/{max_http_attempts}): {e}"
            )
            if http_attempt == max_http_attempts - 1:
                raise Exception("Cannot connect to ComfyUI server.")
            time.sleep(1)

    # WebSocket connect (max 3 min)
    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connected (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(
                f"WebSocket failed (attempt {attempt+1}/{max_attempts}): {e}"
            )
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timeout (3min)")
            time.sleep(5)

    videos = get_videos(ws, prompt)
    ws.close()
    logger.info("WebSocket closed")

    # Find output video
    for node_id in videos:
        if videos[node_id]:
            video_path = videos[node_id][0]
            logger.info(f"Output video from node {node_id}: {video_path}")
            if os.path.exists(video_path):
                return video_path

    return None


def handler(job):
    job_input = job.get("input", {})

    # Log input (truncate base64)
    log_input = job_input.copy()
    for key in ["image_base64", "video_base64", "wav_base64"]:
        if key in log_input:
            log_input[key] = truncate_base64_for_log(log_input[key])
    logger.info(f"Received job input: {log_input}")

    task_id = f"task_{uuid.uuid4()}"

    # --- Process image input ---
    media_path = None
    if "image_path" in job_input:
        media_path = process_input(
            job_input["image_path"], task_id, "input_image.jpg", "path"
        )
    elif "image_url" in job_input:
        media_path = process_input(
            job_input["image_url"], task_id, "input_image.jpg", "url"
        )
    elif "image_base64" in job_input:
        media_path = process_input(
            job_input["image_base64"], task_id, "input_image.jpg", "base64"
        )
    else:
        media_path = "/examples/image.jpg"
        logger.info("Using default image: /examples/image.jpg")

    # --- Process audio input ---
    wav_path = None
    if "wav_path" in job_input:
        wav_path = process_input(
            job_input["wav_path"], task_id, "input_audio.wav", "path"
        )
    elif "wav_url" in job_input:
        wav_path = process_input(
            job_input["wav_url"], task_id, "input_audio.wav", "url"
        )
    elif "wav_base64" in job_input:
        wav_path = process_input(
            job_input["wav_base64"], task_id, "input_audio.wav", "base64"
        )
    else:
        wav_path = "/examples/audio.mp3"
        logger.info("Using default audio: /examples/audio.mp3")

    # --- Parameters ---
    prompt_text = job_input.get("prompt", "A person talking naturally")
    width = job_input.get("width", 640)
    height = job_input.get("height", 640)

    # Calculate number of S2V chunks from audio duration
    audio_duration = get_audio_duration(wav_path)
    if audio_duration is None:
        audio_duration = 5.0  # fallback

    num_chunks = max(1, math.ceil(audio_duration * S2V_FPS / S2V_CHUNK_LENGTH))
    logger.info(
        f"Audio duration: {audio_duration:.2f}s, S2V chunks: {num_chunks} "
        f"(~{num_chunks * S2V_CHUNK_LENGTH / S2V_FPS:.1f}s of video)"
    )

    logger.info(
        f"Settings: prompt='{prompt_text}', width={width}, height={height}"
    )
    logger.info(f"Image: {media_path}")
    logger.info(f"Audio: {wav_path}")

    # --- Validate files ---
    if not os.path.exists(media_path):
        logger.error(f"Image file not found: {media_path}")
        return {"error": f"Image file not found: {media_path}"}

    if not os.path.exists(wav_path):
        logger.error(f"Audio file not found: {wav_path}")
        return {"error": f"Audio file not found: {wav_path}"}

    logger.info(f"Image size: {os.path.getsize(media_path)} bytes")
    logger.info(f"Audio size: {os.path.getsize(wav_path)} bytes")

    # --- Build S2V workflow ---
    prompt = load_workflow("/S2V_single.json")

    # Inject dynamic inputs
    prompt["11"]["inputs"]["image"] = media_path
    prompt["9"]["inputs"]["audio"] = wav_path
    prompt["5"]["inputs"]["text"] = prompt_text
    prompt["12"]["inputs"]["width"] = width
    prompt["12"]["inputs"]["height"] = height

    # Dynamically extend workflow for longer audio
    prompt = build_s2v_workflow(prompt, num_chunks)

    logger.info(f"Final workflow: {len(prompt)} nodes, {num_chunks} chunks")

    # ------------------------------------------------------------------
    # Run S2V generation
    # ------------------------------------------------------------------
    logger.info("Running Wan 2.2 S2V generation...")
    output_video_path = _run_comfyui_job(prompt)

    if not output_video_path:
        logger.error("No output video found.")
        return {"error": "No output video produced."}

    # ------------------------------------------------------------------
    # Two-pass face enhancement (optional)
    # ------------------------------------------------------------------
    two_pass_face = job_input.get("two_pass_face", False)
    if two_pass_face:
        logger.info("Two-pass face enhancement enabled")
        try:
            from face_pipeline import auto_crop_input, composite_two_pass

            two_pass_params = job_input.get("two_pass_params", {})
            target_coverage = two_pass_params.get("target_coverage", 0.35)
            crop_margin = two_pass_params.get("crop_margin", 0.15)

            cropped_image_path = auto_crop_input(
                media_path,
                target_coverage=target_coverage,
                margin=crop_margin,
            )

            if cropped_image_path is not None:
                logger.info("Pass 2: Running S2V (face close-up)...")
                prompt2 = load_workflow("/S2V_single.json")

                prompt2["11"]["inputs"]["image"] = cropped_image_path
                prompt2["9"]["inputs"]["audio"] = wav_path
                prompt2["5"]["inputs"]["text"] = prompt_text
                face_crop_size = two_pass_params.get("face_crop_size", 512)
                prompt2["12"]["inputs"]["width"] = face_crop_size
                prompt2["12"]["inputs"]["height"] = face_crop_size

                prompt2 = build_s2v_workflow(prompt2, num_chunks)

                face_video_path = _run_comfyui_job(prompt2)

                if face_video_path:
                    logger.info("Compositing Pass 2 face onto Pass 1...")
                    composited_path = f"/tmp/twopass_{task_id}.mp4"
                    composite_two_pass(
                        full_video_path=output_video_path,
                        face_video_path=face_video_path,
                        output_path=composited_path,
                        face_margin=two_pass_params.get("face_margin", 0.2),
                        feather_radius=two_pass_params.get("feather_radius", 20),
                        temporal_window=two_pass_params.get("temporal_window", 5),
                        detect_interval=two_pass_params.get("detect_interval", 5),
                    )
                    output_video_path = composited_path
                    logger.info("Two-pass composite done")
                else:
                    logger.warning("Pass 2 produced no video -- using Pass 1 only")
            else:
                logger.info("Face already fills frame -- skipping Pass 2")

        except Exception as e:
            logger.error(f"Two-pass enhancement failed (using Pass 1): {e}")
            import traceback

            logger.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Face-fix postprocessing (Crop-Restore-Stitch pipeline)
    # ------------------------------------------------------------------
    face_fix = job_input.get("face_fix", False)
    if face_fix:
        logger.info("Face-fix postprocessing pipeline starting...")
        try:
            from face_pipeline import run_face_pipeline

            face_fix_params = job_input.get("face_fix_params", {})
            fixed_path = f"/tmp/facefix_{task_id}.mp4"

            run_face_pipeline(
                input_video=output_video_path,
                output_video=fixed_path,
                fidelity_weight=face_fix_params.get("fidelity_weight", 0.6),
                face_margin=face_fix_params.get("face_margin", 0.2),
                crop_size=face_fix_params.get("crop_size", 512),
                temporal_window=face_fix_params.get("temporal_window", 5),
                feather_radius=face_fix_params.get("feather_radius", 15),
                detect_interval=face_fix_params.get("detect_interval", 5),
                restore_batch_size=face_fix_params.get("restore_batch_size", 8),
                codeformer_model_path=face_fix_params.get(
                    "codeformer_model_path", "/models/codeformer/codeformer.pth"
                ),
                upscale_enabled=face_fix_params.get("upscale_enabled", False),
                upscale_model=face_fix_params.get(
                    "upscale_model", "RealESRGAN_x2plus"
                ),
                upscale_target_height=face_fix_params.get(
                    "upscale_target_height", 1080
                ),
                upscale_tile_size=face_fix_params.get("upscale_tile_size", 512),
            )
            output_video_path = fixed_path
            logger.info("Face-fix postprocessing done")
        except Exception as e:
            logger.error(f"Face-fix postprocessing failed (using original): {e}")
            import traceback

            logger.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Output: network volume or base64
    # ------------------------------------------------------------------
    use_network_volume = job_input.get("network_volume", False)

    if use_network_volume:
        logger.info("Copying video to network volume")
        try:
            output_filename = f"s2v_{task_id}.mp4"
            output_path = f"/runpod-volume/{output_filename}"

            source_file_size = os.path.getsize(output_video_path)
            shutil.copy2(output_video_path, output_path)
            copied_file_size = os.path.getsize(output_path)

            if source_file_size == copied_file_size:
                logger.info(f"Video copied to '{output_path}'")
            else:
                logger.warning(
                    f"Size mismatch: source={source_file_size}, copy={copied_file_size}"
                )

            return {"video_path": output_path}

        except Exception as e:
            logger.error(f"Video copy failed: {e}")
            return {"error": f"Video copy failed: {e}"}
    else:
        logger.info("Base64 encoding output")
        try:
            file_size = os.path.getsize(output_video_path)
            logger.info(f"Video file size: {file_size} bytes")

            with open(output_video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")

            logger.info(f"Base64 encoded: {len(video_data)} chars")
            return {"video": video_data}

        except Exception as e:
            logger.error(f"Base64 encoding failed: {e}")
            return {"error": f"Base64 encoding failed: {e}"}


runpod.serverless.start({"handler": handler})
