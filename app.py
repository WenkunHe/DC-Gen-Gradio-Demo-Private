#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import uuid

import gradio as gr
import torch
from huggingface_hub import snapshot_download

from pipeline_dcgen_flux import DCGen_FluxPipeline

INTRODUCTION = """
# DC-Gen: High-Resolution Image Generation with DC-AE

DC-Gen is a high-resolution text-to-image generation framework built on FLUX, using Deep Compression
AutoEncoders (DC-AE) to enable native 1K and 4K image generation.

- **1K model**: uses DC-AE-f32 latent space, supports flexible aspect ratios up to 1024×1024 equivalent
- **4K model**: uses DC-AE-f64 latent space, generates 4096×4096 images natively

---
> Run locally:
```bash
git clone https://github.com/NVlabs/DC-Gen.git && cd DC-Gen
conda create -n dcgen python=3.10 && conda activate dcgen
pip install -r requirements.txt
python app.py
```
---
"""

HUB_REPO_1K = 'nvidia/DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K'
HUB_REPO_4K = 'nvidia/DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K'

ASPECT_RATIOS_1K = {
    '1:1  (1024×1024)': (1024, 1024),
    '9:7  (1152×896)':  (1152, 896),
    '7:9  (896×1152)':  (896, 1152),
    '3:1  (1728×576)':  (1728, 576),
    '1:3  (576×1728)':  (576, 1728),
    '4:3  (1152×832)':  (1152, 832),
    '3:4  (832×1152)':  (832, 1152),
    '16:9 (1344×768)':  (1344, 768),
    '9:16 (768×1344)':  (768, 1344),
}

EXAMPLES_1K = [
    "A well-groomed man with short, sleek brown hair and fashionable eyeglasses smiles warmly into the camera in this polished professional portrait.",
    "Anime style. A graceful anime girl with flowing silver hair adorned with a delicate blue ribbon stands in a serene outdoor scene, a vibrant blue butterfly perched on her shoulder.",
    "A serene landscape where rolling green mountains cradle a tranquil lake, their peaks mirrored perfectly on the glassy surface. Mist drifts gently along the slopes.",
]

EXAMPLES_4K = [
    "A portrait of a beautiful young woman with long, voluminous, wavy blonde hair standing in Venice, with a narrow canal and classic Venetian buildings in the background. She wears a light blue blouse with a white lace collar, tucked into bright yellow pants.",
    "Two figures paddle a canoe in a tranquil lake, surrounded by towering mountains and lush trees, with a waterfall cascading down rocky cliffs in the background. Soft light filters through the clouds, casting reflections on the water's surface.",
]

repo_root = pathlib.Path(__file__).resolve().parent
output_dir = repo_root / 'results'
output_dir.mkdir(parents=True, exist_ok=True)


def load_pipeline(hub_repo: str) -> DCGen_FluxPipeline:
    local_dir = repo_root / 'pretrained_models' / hub_repo
    if not (local_dir / 'model_index.json').exists():
        token = os.environ.get('HF_TOKEN')
        snapshot_download(repo_id=hub_repo, repo_type='dataset', local_dir=str(local_dir), token=token)
    pipe = DCGen_FluxPipeline.from_pretrained(str(local_dir), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


print('Loading 1K pipeline...')
pipe_1k = load_pipeline(HUB_REPO_1K)
print('Loading 4K pipeline...')
pipe_4k = load_pipeline(HUB_REPO_4K)
print('Both pipelines loaded.')


def generate_1k(prompt: str, aspect_ratio: str, num_steps: int, guidance: float, seed: int) -> str:
    h, w = ASPECT_RATIOS_1K[aspect_ratio]
    pipe_1k.to('cuda')
    with torch.no_grad():
        out = pipe_1k(
            prompt.strip(),
            height=h,
            width=w,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=torch.Generator('cuda').manual_seed(int(seed)),
            use_flux_2=True,
        ).images[0]
    pipe_1k.to('cpu')
    torch.cuda.empty_cache()
    path = output_dir / f'1k_{uuid.uuid4().hex[:8]}.jpg'
    out.save(str(path))
    return str(path)


def generate_4k(prompt: str, num_steps: int, guidance: float, seed: int) -> str:
    pipe_4k.to('cuda')
    with torch.no_grad():
        out = pipe_4k(
            prompt.strip(),
            height=4096,
            width=4096,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=torch.Generator('cuda').manual_seed(int(seed)),
            use_flux_2=False,
        ).images[0]
    pipe_4k.to('cpu')
    torch.cuda.empty_cache()
    path = output_dir / f'4k_{uuid.uuid4().hex[:8]}.jpg'
    out.save(str(path))
    return str(path)


with gr.Blocks(title='DC-Gen') as demo:
    gr.Markdown(INTRODUCTION)

    with gr.Tabs():
        with gr.Tab('DC-Gen 1K'):
            gr.Markdown('### DC-Gen-FLUX.1-Krea-Dev — 1K Generation (DC-AE-f32)')
            with gr.Row():
                with gr.Column():
                    prompt_1k = gr.Textbox(label='Prompt', lines=4)
                    aspect_1k = gr.Dropdown(
                        list(ASPECT_RATIOS_1K.keys()),
                        value='1:1  (1024×1024)',
                        label='Aspect Ratio',
                    )
                    with gr.Row():
                        steps_1k = gr.Slider(1, 50, value=20, step=1, label='Steps')
                        guidance_1k = gr.Slider(1.0, 10.0, value=3.5, step=0.1, label='Guidance Scale')
                    seed_1k = gr.Number(0, label='Seed', precision=0)
                with gr.Column():
                    out_1k = gr.Image(label='Output', type='filepath')
                    btn_1k = gr.Button('Generate', variant='primary')

            gr.Markdown('### Examples')
            for ex in EXAMPLES_1K:
                with gr.Row():
                    gr.Textbox(value=ex, show_label=False, interactive=False, lines=2)
                    use_btn = gr.Button('Use', scale=0, min_width=60)
                    use_btn.click(lambda p=ex: p, outputs=[prompt_1k])

            btn_1k.click(generate_1k, inputs=[prompt_1k, aspect_1k, steps_1k, guidance_1k, seed_1k], outputs=[out_1k])

        with gr.Tab('DC-Gen 4K'):
            gr.Markdown('### DC-Gen-FLUX.1-Krea-Dev — 4K Generation (DC-AE-f64, 4096×4096)')
            with gr.Row():
                with gr.Column():
                    prompt_4k = gr.Textbox(label='Prompt', lines=4)
                    with gr.Row():
                        steps_4k = gr.Slider(1, 50, value=20, step=1, label='Steps')
                        guidance_4k = gr.Slider(1.0, 10.0, value=3.5, step=0.1, label='Guidance Scale')
                    seed_4k = gr.Number(0, label='Seed', precision=0)
                with gr.Column():
                    out_4k = gr.Image(label='Output', type='filepath')
                    btn_4k = gr.Button('Generate', variant='primary')

            gr.Markdown('### Examples')
            for ex in EXAMPLES_4K:
                with gr.Row():
                    gr.Textbox(value=ex, show_label=False, interactive=False, lines=2)
                    use_btn = gr.Button('Use', scale=0, min_width=60)
                    use_btn.click(lambda p=ex: p, outputs=[prompt_4k])

            btn_4k.click(generate_4k, inputs=[prompt_4k, steps_4k, guidance_4k, seed_4k], outputs=[out_4k])

demo.queue(default_concurrency_limit=1)

if __name__ == '__main__':
    demo.launch(show_error=True, share=True, server_name='0.0.0.0', server_port=7861)
