# Use specific version of nvidia cuda image
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

# wget 설치 (URL 다운로드를 위해)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa

# Face-fix pipeline dependencies (Crop-Restore-Stitch for melting teeth fix)
# Note: basicsr is NOT needed for CodeFormer — arch is vendored in codeformer_arch.py
# realesrgan (Real-ESRGAN) is used for whole-frame upscaling after face-fix stitch
RUN pip install insightface onnxruntime-gpu opencv-python-headless realesrgan

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt


# --- Wan 2.2 S2V models ---
# Diffusion model: Wan 2.2 Speech-to-Video 14B (fp8, 16.4 GB)
RUN wget -q https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors -O /ComfyUI/models/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors

# Lightning LoRA: 4-step acceleration (1.23 GB)
RUN wget -q https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors -O /ComfyUI/models/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors

# VAE: shared Wan 2.1/2.2 VAE (254 MB)
RUN wget -q https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors -O /ComfyUI/models/vae/wan_2.1_vae.safetensors

# Text encoder: UMT5-XXL (same as before)
RUN wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors -O /ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# Audio encoder: wav2vec2 large English for S2V (631 MB)
RUN mkdir -p /ComfyUI/models/audio_encoders && \
    wget -q https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors -O /ComfyUI/models/audio_encoders/wav2vec2_large_english_fp16.safetensors

# CodeFormer face restoration model (~370MB)
RUN mkdir -p /models/codeformer && \
    wget -q https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -O /models/codeformer/codeformer.pth

# Real-ESRGAN x2plus upscaler model (~67MB)
RUN mkdir -p /models/realesrgan && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O /models/realesrgan/RealESRGAN_x2plus.pth

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
#force
