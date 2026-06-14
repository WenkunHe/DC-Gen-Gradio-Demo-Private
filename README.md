---
title: DC-Gen
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
---

# DC-Gen Gradio Demo

Interactive demo for **DC-Gen**, a high-resolution text-to-image generation framework that uses Deep Compression AutoEncoders (DC-AE) to enable native 1K and 4K image synthesis on top of FLUX.

## Models

| Tab | Model | VAE | Resolution |
|-----|-------|-----|------------|
| DC-Gen 1K | `nvidia/DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K` | DC-AE-f32 | up to 1024×1024, flexible aspect ratios |
| DC-Gen 4K | `nvidia/DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K` | DC-AE-f64 | 4096×4096 |

## Local Setup

```bash
git clone -b dc-gen-gradio-demo https://github.com/NVlabs/DC-Gen.git
cd DC-Gen
conda create -n dcgen python=3.10 && conda activate dcgen
pip install -r requirements.txt
python app.py
```

The first run downloads model weights from HuggingFace Hub into `pretrained_models/`.
