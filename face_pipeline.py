"""Crop-Restore-Stitch postprocessing pipeline for fixing melting teeth artifacts.

Pipeline stages:
  1. Face detection & tracking (InsightFace)
  2. Face-centric cropping (512x512 with margin)
  3. CodeFormer face restoration (teeth/detail fix)
  4. Temporal smoothing (reduce frame-to-frame flicker)
  5. Feathered re-compositing with color matching (seamless stitch back)
"""

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(args: list[str], desc: str = "ffmpeg"):
    cmd = ["ffmpeg", "-y"] + args
    logger.info(f"[FacePipeline] {desc}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed (rc={result.returncode}): {result.stderr[:500]}")
    return result


def _probe_video(video_path: str) -> tuple[float, int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")
    parts = result.stdout.strip().split(",")
    width = int(parts[0])
    height = int(parts[1])
    fps_parts = parts[2].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    return fps, width, height


def extract_frames(video_path: str, work_dir: str):
    fps, width, height = _probe_video(video_path)
    logger.info(f"[FacePipeline] Video: {width}x{height} @ {fps}fps")

    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    _run_ffmpeg([
        "-i", video_path, "-vsync", "0",
        os.path.join(frames_dir, "frame_%06d.png"),
    ], desc="Extract frames")

    # Extract audio
    audio_path = os.path.join(work_dir, "audio.wav")
    try:
        _run_ffmpeg([
            "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "2", audio_path,
        ], desc="Extract audio")
    except RuntimeError:
        logger.info("[FacePipeline] No audio track found")
        audio_path = ""

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    logger.info(f"[FacePipeline] Extracted {len(frame_files)} frames")
    return fps, frames_dir, audio_path, len(frame_files)


def encode_video(frames_dir: str, audio_path: str, output_path: str, fps: float):
    args = [
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
    ]
    if audio_path and os.path.exists(audio_path):
        args += ["-i", audio_path, "-c:a", "aac", "-b:a", "192k"]
    args += [
        "-c:v", "libx264", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ]
    _run_ffmpeg(args, desc="Encode video")
    return output_path


# ---------------------------------------------------------------------------
# Face detection & tracking
# ---------------------------------------------------------------------------

class FaceDetector:
    """InsightFace-based face detector with multi-frame tracking."""

    def __init__(self, det_size=(640, 640), min_confidence=0.5):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.min_confidence = min_confidence
        logger.info(f"[FacePipeline] InsightFace initialized (det_size={det_size})")

    def detect_faces(self, frame):
        faces = self.app.get(frame)
        results = []
        for face in faces:
            if face.det_score < self.min_confidence:
                continue
            results.append({
                "bbox": face.bbox.astype(int),
                "confidence": float(face.det_score),
                "embedding": face.embedding if face.embedding is not None else np.zeros(512),
            })
        return results

    def detect_and_track(self, frames, detect_interval=5):
        """Detect faces across frames and assign consistent person IDs.

        Returns dict: person_id -> list[dict | None] (one per frame).
        """
        num_frames = len(frames)
        key_detections = {}
        for i in range(0, num_frames, detect_interval):
            key_detections[i] = self.detect_faces(frames[i])
        if (num_frames - 1) not in key_detections and num_frames > 0:
            key_detections[num_frames - 1] = self.detect_faces(frames[-1])

        # Assign person IDs using embedding cosine similarity
        next_pid = 0
        known_embeddings = {}
        sim_threshold = 0.4

        for frame_idx in sorted(key_detections.keys()):
            for face in key_detections[frame_idx]:
                best_id, best_sim = -1, sim_threshold
                for pid, emb in known_embeddings.items():
                    sim = _cosine_sim(face["embedding"], emb)
                    if sim > best_sim:
                        best_sim, best_id = sim, pid
                if best_id >= 0:
                    face["person_id"] = best_id
                    known_embeddings[best_id] = 0.7 * known_embeddings[best_id] + 0.3 * face["embedding"]
                else:
                    face["person_id"] = next_pid
                    known_embeddings[next_pid] = face["embedding"].copy()
                    next_pid += 1

        # Build per-person tracks
        person_ids = set()
        for faces in key_detections.values():
            for f in faces:
                person_ids.add(f["person_id"])

        tracks = {pid: [None] * num_frames for pid in person_ids}
        for frame_idx, faces in key_detections.items():
            for f in faces:
                tracks[f["person_id"]][frame_idx] = f

        # Interpolate bounding boxes for intermediate frames
        for pid in person_ids:
            _interpolate_track(tracks[pid])

        logger.info(f"[FacePipeline] Tracked {len(person_ids)} person(s) across {num_frames} frames")
        return tracks

    @staticmethod
    def crop_face(frame, face, margin=0.2, size=512):
        """Crop face region with margin, resize to size x size.

        Returns (cropped_image, crop_params_dict).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face["bbox"][:4].astype(int)
        face_w, face_h = x2 - x1, y2 - y1

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        half_side = int(max(face_w, face_h) * (1 + margin) / 2)

        crop_x1 = max(0, center_x - half_side)
        crop_y1 = max(0, center_y - half_side)
        crop_x2 = min(w, center_x + half_side)
        crop_y2 = min(h, center_y + half_side)

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)

        params = {
            "crop_x1": crop_x1, "crop_y1": crop_y1,
            "crop_x2": crop_x2, "crop_y2": crop_y2,
        }
        return cropped_resized, params


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _interpolate_track(track):
    """Fill gaps <= 10 frames by linearly interpolating bounding boxes. Modifies in-place."""
    max_gap = 10
    n = len(track)
    i = 0
    while i < n:
        if track[i] is not None:
            i += 1
            continue
        gap_start = i
        while i < n and track[i] is None:
            i += 1
        gap_end = i
        if gap_end - gap_start > max_gap:
            continue
        prev = track[gap_start - 1] if gap_start > 0 else None
        nxt = track[gap_end] if gap_end < n else None
        if prev is None and nxt is None:
            continue
        for j in range(gap_start, gap_end):
            if prev and nxt:
                t = (j - gap_start + 1) / (gap_end - gap_start + 1)
                interp_bbox = (1 - t) * prev["bbox"] + t * nxt["bbox"]
                track[j] = {"bbox": interp_bbox.astype(int), "embedding": prev["embedding"], "person_id": prev["person_id"]}
            elif prev:
                track[j] = {"bbox": prev["bbox"].copy(), "embedding": prev["embedding"], "person_id": prev["person_id"]}
            elif nxt:
                track[j] = {"bbox": nxt["bbox"].copy(), "embedding": nxt["embedding"], "person_id": nxt["person_id"]}


# ---------------------------------------------------------------------------
# Face restoration (CodeFormer)
# ---------------------------------------------------------------------------

class FaceRestorer:
    """CodeFormer-based face restoration."""

    def __init__(self, model_path="/models/codeformer/codeformer.pth", fidelity_weight=0.6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fidelity_weight = fidelity_weight
        self._load_model(model_path)
        logger.info(f"[FacePipeline] CodeFormer loaded (fidelity={fidelity_weight}, device={self.device})")

    def _load_model(self, model_path):
        from codeformer_arch import CodeFormer
        self.model = CodeFormer(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "params_ema" in checkpoint:
            self.model.load_state_dict(checkpoint["params_ema"])
        elif "params" in checkpoint:
            self.model.load_state_dict(checkpoint["params"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def restore_batch(self, face_crops, batch_size=8):
        """Restore a list of 512x512 BGR face crops. Returns list of restored crops."""
        results = []
        for i in range(0, len(face_crops), batch_size):
            batch = face_crops[i:i + batch_size]
            tensors = []
            for crop in batch:
                if crop.shape[0] != 512 or crop.shape[1] != 512:
                    crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                t = torch.from_numpy(crop[:, :, ::-1].copy()).float() / 255.0
                t = t.permute(2, 0, 1)
                t = (t - 0.5) / 0.5
                tensors.append(t)

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                output = self.model(batch_tensor, w=self.fidelity_weight, adain=True)
                if isinstance(output, (tuple, list)):
                    output = output[0]

            for j in range(output.shape[0]):
                out = output[j].clamp(-1, 1)
                out = ((out * 0.5 + 0.5) * 255.0).byte()
                out = out.permute(1, 2, 0).cpu().numpy()
                out = out[:, :, ::-1].copy()  # RGB -> BGR
                results.append(out)

            if len(results) % 100 == 0:
                logger.info(f"[FacePipeline] Restored {len(results)}/{len(face_crops)} faces")

        return results


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def temporal_smooth(crops, window=5):
    """Apply Gaussian temporal smoothing across sequential face crops to reduce flicker."""
    n = len(crops)
    if n <= 1 or window <= 1:
        return crops

    half_w = window // 2
    kernel = np.array([
        np.exp(-0.5 * (x / max(half_w * 0.5, 1)) ** 2)
        for x in range(-half_w, half_w + 1)
    ], dtype=np.float32)
    kernel /= kernel.sum()

    smoothed = []
    for i in range(n):
        acc = np.zeros_like(crops[i], dtype=np.float32)
        wsum = 0.0
        for k, w in enumerate(kernel):
            j = i + k - half_w
            if 0 <= j < n:
                acc += w * crops[j].astype(np.float32)
                wsum += w
        if wsum > 0:
            acc /= wsum
        smoothed.append(np.clip(acc, 0, 255).astype(np.uint8))
    return smoothed


# ---------------------------------------------------------------------------
# Feathered re-compositing
# ---------------------------------------------------------------------------

def _create_feathered_mask(height, width, feather_radius):
    mask = np.ones((height, width), dtype=np.float32)
    r = min(feather_radius, height // 4, width // 4)
    if r <= 0:
        return mask
    for i in range(r):
        alpha = (i + 1) / (r + 1)
        mask[i, :] = np.minimum(mask[i, :], alpha)
        mask[height - 1 - i, :] = np.minimum(mask[height - 1 - i, :], alpha)
        mask[:, i] = np.minimum(mask[:, i], alpha)
        mask[:, width - 1 - i] = np.minimum(mask[:, width - 1 - i], alpha)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=r * 0.3)
    return mask


def _match_color(source, target):
    """Match color distribution of source to target using LAB mean/std transfer."""
    if source.shape != target.shape:
        return source
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        sm, ss = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std()
        tm, ts = tgt_lab[:, :, ch].mean(), tgt_lab[:, :, ch].std()
        if ss < 1e-6:
            continue
        src_lab[:, :, ch] = (src_lab[:, :, ch] - sm) * (ts / ss) + tm
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def stitch_face(frame, restored_crop, params, feather_radius=15):
    """Stitch restored face crop back into frame with feathered blending + color matching."""
    crop_w = params["crop_x2"] - params["crop_x1"]
    crop_h = params["crop_y2"] - params["crop_y1"]
    if crop_w <= 0 or crop_h <= 0:
        return frame

    resized = cv2.resize(restored_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    original_region = frame[params["crop_y1"]:params["crop_y2"], params["crop_x1"]:params["crop_x2"]]
    resized = _match_color(resized, original_region)

    mask = _create_feathered_mask(crop_h, crop_w, feather_radius)
    mask_3ch = mask[:, :, np.newaxis]
    blended = mask_3ch * resized.astype(np.float32) + (1 - mask_3ch) * original_region.astype(np.float32)

    result = frame.copy()
    result[params["crop_y1"]:params["crop_y2"], params["crop_x1"]:params["crop_x2"]] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_face_pipeline(
    input_video: str,
    output_video: str,
    fidelity_weight: float = 0.6,
    face_margin: float = 0.2,
    crop_size: int = 512,
    temporal_window: int = 5,
    feather_radius: int = 15,
    detect_interval: int = 5,
    restore_batch_size: int = 8,
    codeformer_model_path: str = "/models/codeformer/codeformer.pth",
) -> str:
    """Run the full Crop-Restore-Stitch face-fix pipeline on a video.

    Returns path to the output video.
    """
    work_dir = tempfile.mkdtemp(prefix="facefix_")
    logger.info(f"[FacePipeline] Work dir: {work_dir}")

    # Stage 0: Extract frames
    logger.info("[FacePipeline] Stage 0: Extracting frames...")
    fps, frames_dir, audio_path, frame_count = extract_frames(input_video, work_dir)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not frame_files:
        raise RuntimeError("No frames extracted from video")

    frames = [cv2.imread(os.path.join(frames_dir, f)) for f in frame_files]

    # Stage 1: Face detection & tracking
    logger.info("[FacePipeline] Stage 1: Detecting and tracking faces...")
    detector = FaceDetector()
    tracks = detector.detect_and_track(frames, detect_interval=detect_interval)

    if not tracks:
        logger.info("[FacePipeline] No faces detected -- skipping postprocessing")
        import shutil
        shutil.copy2(input_video, output_video)
        return output_video

    # Stage 2-3: Crop, Restore, Smooth, Stitch per person
    logger.info("[FacePipeline] Stage 2-3: Restoring faces...")
    restorer = FaceRestorer(
        model_path=codeformer_model_path,
        fidelity_weight=fidelity_weight,
    )

    for person_id, track in tracks.items():
        logger.info(f"[FacePipeline] Processing person {person_id}...")

        crop_indices, crops, crop_params_list = [], [], []
        for i, face in enumerate(track):
            if face is None:
                continue
            crop, params = detector.crop_face(frames[i], face, margin=face_margin, size=crop_size)
            crop_indices.append(i)
            crops.append(crop)
            crop_params_list.append(params)

        if not crops:
            continue

        logger.info(f"[FacePipeline]   Restoring {len(crops)} face crops...")
        restored_crops = restorer.restore_batch(crops, batch_size=restore_batch_size)

        # Temporal smoothing
        if temporal_window > 1 and len(restored_crops) > 1:
            logger.info(f"[FacePipeline]   Temporal smoothing (window={temporal_window})...")
            restored_crops = temporal_smooth(restored_crops, temporal_window)

        # Feathered stitch back
        logger.info(f"[FacePipeline]   Stitching back (feather={feather_radius}px)...")
        for idx, (frame_i, params) in enumerate(zip(crop_indices, crop_params_list)):
            frames[frame_i] = stitch_face(frames[frame_i], restored_crops[idx], params, feather_radius)

    # Free GPU memory
    del restorer
    torch.cuda.empty_cache()

    # Stage 4: Write fixed frames and re-encode
    logger.info("[FacePipeline] Stage 4: Encoding output video...")
    fixed_frames_dir = os.path.join(work_dir, "fixed_frames")
    os.makedirs(fixed_frames_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(fixed_frames_dir, f"frame_{i + 1:06d}.png"), frame)

    encode_video(fixed_frames_dir, audio_path, output_video, fps)
    logger.info(f"[FacePipeline] Pipeline complete: {output_video}")
    return output_video
