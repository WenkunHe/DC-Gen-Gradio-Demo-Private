# DC-Gen Gradio Demo

DC-Gen adapts high-resolution visual generation and editing models (e.g., FLUX, Wan2.1, Qwen-Image-Edit) to deeply compressed latent spaces through efficient post-training. It enables native 4K image synthesis, and achieves up to 54x acceleration.

## Models

| Tab | VAE | Resolution |
|-----|-----|------------|
| DC-Gen-FLUX-1K | DC-AE-f32c32 | 1K, flexible aspect ratios |
| DC-Gen-FLUX-4K | DC-AE-1.5-f64c128 | 4K, flexible aspect ratios |
| DC-Gen-Wan2.1-T2V-14B-720P | DC-AE-V-f32t4c32 | 720×1280 |
| DC-Gen-Wan2.1-I2V-14B-720P | DC-AE-V-f32t4c32 | 720×1280 |
| DC-Gen-Qwen-Image-Edit-Res1K | DC-AE-f32c32 | 1K, flexible aspect ratios |

## Local Setup

```bash
git clone git@github.com:WenkunHe/DC-Gen-Gradio-Demo-Private.git
cd DC-Gen-Gradio-Demo-Private
conda create -n dcgen python=3.10 && conda activate dcgen
pip install -r requirements.txt

# Single GPU: lazy load/unload between requests
HF_TOKEN=<your_hf_token> python app.py

# 7 GPUs: load all models persistently, one per GPU
HF_TOKEN=<your_hf_token> python app.py --load_before_generation
```

The first run downloads model weights from HuggingFace Hub into `pretrained_models/`. A HuggingFace token with access to `nvidia/` is required.
