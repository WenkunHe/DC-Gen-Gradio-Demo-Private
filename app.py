#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import uuid

import gradio as gr
import torch
from huggingface_hub import snapshot_download

repo_root = pathlib.Path(__file__).resolve().parent
output_dir = repo_root / 'results'
output_dir.mkdir(parents=True, exist_ok=True)

# Clone AnyFlow main repo for the `far/` package used by the AnyFlow pipeline
anyflow_src = repo_root / 'AnyFlow'
if not anyflow_src.exists():
    subprocess.run(
        ['git', 'clone', '--depth', '1', 'https://github.com/NVlabs/AnyFlow.git', str(anyflow_src)],
        check=True,
    )
sys.path.insert(0, str(anyflow_src))

from pipeline_dcgen_flux import DCGen_FluxPipeline  # noqa: E402

INTRODUCTION = """
# DC-Gen: High-Resolution Image Generation with DC-AE

DC-Gen is a high-resolution text-to-image generation framework built on FLUX, using Deep Compression
AutoEncoders (DC-AE) to enable native 1K and 4K image synthesis on top of FLUX.

- **1K model**: uses DC-AE-f32 latent space, supports flexible aspect ratios up to 1024×1024 equivalent
- **4K model**: uses DC-AE-f64 latent space, generates 4096×4096 images natively

---
> Run locally:
```bash
git clone git@github.com:WenkunHe/DC-Gen-Gradio-Demo-Private.git
cd DC-Gen-Gradio-Demo-Private
conda create -n dcgen python=3.10 && conda activate dcgen
pip install -r requirements.txt
HF_TOKEN=<your_hf_token> python app.py
```
---
"""

HUB_REPO          = 'nvidia/DC-Gen-FLUX.1-Krea-Dev'
HUB_REPO_1K        = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K'
HUB_REPO_1K_ANYFLOW = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K-Anyflow'
HUB_REPO_4K        = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K'

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


def _download_subdir(subdir: str) -> pathlib.Path:
    local_dir = repo_root / 'pretrained_models' / subdir
    if not (local_dir / 'model_index.json').exists():
        token = os.environ.get('HF_TOKEN')
        snapshot_download(
            repo_id=HUB_REPO,
            repo_type='model',
            local_dir=str(local_dir),
            allow_patterns=f'{subdir}/*',
            token=token,
        )
        nested = local_dir / subdir
        if nested.exists():
            for item in nested.iterdir():
                item.rename(local_dir / item.name)
            nested.rmdir()
    return local_dir


def load_pipeline_standard(subdir: str) -> DCGen_FluxPipeline:
    local_dir = _download_subdir(subdir)
    pipe = DCGen_FluxPipeline.from_pretrained(str(local_dir), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_pipeline_anyflow(subdir: str):
    local_dir = _download_subdir(subdir)
    # Copy transformer model into far/models so the import resolves correctly
    import shutil
    dst = anyflow_src / 'far' / 'models' / 'transformer_dcgen_flux_model.py'
    if not dst.exists():
        shutil.copy(local_dir / 'transformer_dcgen_flux_model.py', dst)
    # Import custom classes
    sys.path.insert(0, str(local_dir))
    from pipeline_dcgen_flux_anyflow import DCGenFluxAnyFlowPipeline  # noqa: E402
    from far.models.transformer_dcgen_flux_model import DCGenFluxFlowMapModel  # noqa: E402
    from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler  # noqa: E402
    from diffusers import AutoencoderDC
    from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

    dtype = torch.bfloat16

    # Load each component individually to avoid diffusers' custom-library issubclass check
    vae           = AutoencoderDC.from_pretrained(local_dir / 'vae', torch_dtype=dtype)
    tokenizer     = AutoTokenizer.from_pretrained(local_dir / 'tokenizer')
    tokenizer_2   = AutoTokenizer.from_pretrained(local_dir / 'tokenizer_2')
    text_encoder  = CLIPTextModel.from_pretrained(local_dir / 'text_encoder', torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(local_dir / 'text_encoder_2', torch_dtype=dtype)
    transformer   = DCGenFluxFlowMapModel.from_pretrained(local_dir / 'transformer', torch_dtype=dtype)
    scheduler     = FlowMapDiscreteScheduler.from_pretrained(local_dir / 'scheduler')

    null_embeds = torch.load(local_dir / 'flux_null_embedding.pth', map_location='cpu', weights_only=True)

    pipe = DCGenFluxAnyFlowPipeline(
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        scheduler=scheduler,
        null_prompt_embeds=null_embeds.get('null_prompt_embeds'),
        null_pooled_prompt_embeds=null_embeds.get('null_pooled_prompt_embeds'),
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


print('Loading 1K-AnyFlow pipeline...')
pipe_1k_anyflow = load_pipeline_anyflow(HUB_REPO_1K_ANYFLOW)
print('Loading 1K pipeline...')
pipe_1k = load_pipeline_standard(HUB_REPO_1K)
print('Loading 4K pipeline...')
pipe_4k = load_pipeline_standard(HUB_REPO_4K)
print('All pipelines loaded.')


def generate_1k(prompt: str, use_anyflow: str, aspect_ratio: str, num_steps: int, guidance: float, seed: int) -> str:
    h, w = ASPECT_RATIOS_1K[aspect_ratio]
    pipe = pipe_1k_anyflow if use_anyflow == 'AnyFlow' else pipe_1k
    pipe.to('cuda')
    with torch.no_grad():
        kwargs = dict(
            height=h,
            width=w,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=torch.Generator('cuda').manual_seed(int(seed)),
        )
        if use_anyflow != 'AnyFlow':
            kwargs['use_flux_2'] = True
        out = pipe(prompt.strip(), **kwargs).images[0]
    pipe.to('cpu')
    torch.cuda.empty_cache()
    tag = 'anyflow' if use_anyflow == 'AnyFlow' else 'standard'
    path = output_dir / f'1k_{tag}_{uuid.uuid4().hex[:8]}.jpg'
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
                    model_toggle = gr.Radio(
                        choices=['AnyFlow', 'Standard'],
                        value='AnyFlow',
                        label='Model',
                        info='AnyFlow: on-policy distilled model (faster, any-step); Standard: base 1K model',
                    )
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

            btn_1k.click(
                generate_1k,
                inputs=[prompt_1k, model_toggle, aspect_1k, steps_1k, guidance_1k, seed_1k],
                outputs=[out_1k],
            )

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
