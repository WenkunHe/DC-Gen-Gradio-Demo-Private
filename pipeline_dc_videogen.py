"""
DC-VideoGen pipeline builders — T2V and I2V using local model checkpoints.
Requires flash_attn (available in the dcae conda environment).
"""

import pathlib
import sys

import torch
import torch.nn as nn

# ── local paths ───────────────────────────────────────────────────────────────
VIDEOGEN_ROOT = pathlib.Path('/home/hcai/workspace/code/dc-dev-videogen-fix/DC-VideoGen-Wan2.1-14B-Diffusers')
DC_DEV        = pathlib.Path('/home/hcai/workspace/code/dc-dev-videogen-fix/dc-dev')
FUSIONX       = pathlib.Path('/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/fusionx/diffuser_checkpoints')
CKPT          = VIDEOGEN_ROOT / 'checkpoints'

for _p in (str(VIDEOGEN_ROOT), str(DC_DEV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers import UniPCMultistepScheduler, WanTransformer3DModel  # noqa: E402
from transformers import CLIPImageProcessor, T5TokenizerFast, UMT5EncoderModel  # noqa: E402

from dc_ae_v import DCAEV, dc_ae_v_f32t4_chunk_causal  # noqa: E402
from pipeline_dc_videogen_wan_t2v import DCVideoGenWanTextToVideoPipeline  # noqa: E402
from pipeline_dc_videogen_wan_i2v import DCVideoGenWanImageToVideoPipeline  # noqa: E402


# ── VAE wrapper ───────────────────────────────────────────────────────────────

class AEWrapper(nn.Module):
    def __init__(self, model_name: str, model_path: str):
        super().__init__()
        self.config = type('C', (), {
            'scale_factor_temporal': 4,
            'scale_factor_spatial':  32,
            'z_dim':                 32,
            'scaling_factor':        0.7241,
        })()
        cfg = dc_ae_v_f32t4_chunk_causal(model_name, model_path)
        self.ae = DCAEV(cfg).to(dtype=torch.bfloat16)

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def encode(self, video):
        return self.ae.encode(video)

    def decode(self, latents, return_dict=True):
        return (self.ae.decode(latents), None)


# ── CLIP vision encoder wrapper ───────────────────────────────────────────────

class _CLIPOutput:
    def __init__(self, features):
        self.hidden_states = (None, features, None)  # hidden_states[-2] = features


class CLIPVisionWrapper(nn.Module):
    """Wraps the dc-dev XLMRobertaCLIP ViT-H/14 to match diffusers encode_image interface."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        from dc_ai.videogencore.models.wan_blocks.clip import VisionTransformer
        self.vision = VisionTransformer(
            image_size=224, patch_size=14, dim=1280, mlp_ratio=4,
            out_dim=1024, num_heads=16, num_layers=32,
        )
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        vision_state = {k[len('visual.'):]: v for k, v in state.items() if k.startswith('visual.')}
        missing, unexpected = self.vision.load_state_dict(vision_state, strict=False)
        if missing:
            print(f'[CLIPVisionWrapper] {len(missing)} missing keys')
        if unexpected:
            print(f'[CLIPVisionWrapper] {len(unexpected)} unexpected keys')

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, pixel_values, output_hidden_states=False, **kwargs):
        # Keep in fp32 — custom LayerNorm in dc-dev VisionTransformer requires fp32 weights.
        pixel_values = pixel_values.to(dtype=next(self.vision.parameters()).dtype)
        features = self.vision(pixel_values)  # [B, 257, 1280]
        return _CLIPOutput(features)


# ── shared text/scheduler loader ─────────────────────────────────────────────

def _load_common():
    tokenizer    = T5TokenizerFast.from_pretrained(str(FUSIONX), subfolder='tokenizer')
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(FUSIONX), subfolder='text_encoder', torch_dtype=torch.bfloat16)
    scheduler    = UniPCMultistepScheduler.from_pretrained(str(FUSIONX), subfolder='scheduler')
    return tokenizer, text_encoder, scheduler


# ── pipeline builders ─────────────────────────────────────────────────────────

def build_t2v_pipeline() -> DCVideoGenWanTextToVideoPipeline:
    print('[VideoGen] Building T2V pipeline...')
    ae = AEWrapper('dc-ae-v-f32t4c32-1.0-bf16', str(CKPT / 'dc-ae-v-f32t4c32-1.0-bf16.pt'))

    transformer = WanTransformer3DModel(
        patch_size=(1, 1, 1), in_channels=32, out_channels=32,
    ).to(torch.bfloat16)
    sd = torch.load(CKPT / 'dc_videogen_wan2.1_t2v_14b_720p.pt', map_location='cpu', weights_only=True)
    missing, unexpected = transformer.load_state_dict(sd, strict=False)
    print(f'  transformer: missing={len(missing)} unexpected={len(unexpected)}')

    tokenizer, text_encoder, scheduler = _load_common()

    pipe = DCVideoGenWanTextToVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder,
        vae=ae, scheduler=scheduler, transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_i2v_pipeline() -> DCVideoGenWanImageToVideoPipeline:
    print('[VideoGen] Building I2V pipeline...')
    ae = AEWrapper('dc-ae-v-f32t4c32-1.0-bf16', str(CKPT / 'dc-ae-v-f32t4c32-1.0-bf16.pt'))

    transformer = WanTransformer3DModel(
        patch_size=(1, 1, 1), in_channels=68, out_channels=32,
        image_dim=1280, added_kv_proj_dim=5120,
    ).to(torch.bfloat16)
    sd = torch.load(CKPT / 'dc_videogen_wan2.1_i2v_14b_720p.pt', map_location='cpu', weights_only=True)
    missing, unexpected = transformer.load_state_dict(sd, strict=False)
    print(f'  transformer: missing={len(missing)} unexpected={len(unexpected)}')

    tokenizer, text_encoder, scheduler = _load_common()

    clip_path = str(DC_DEV / 'assets/checkpoints/i2v/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')
    image_encoder = CLIPVisionWrapper(clip_path)  # kept in fp32; custom LayerNorm requires fp32 weights

    image_processor = CLIPImageProcessor(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        size={'shortest_edge': 224},
        crop_size={'height': 224, 'width': 224},
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        resample=3,
    )

    pipe = DCVideoGenWanImageToVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder,
        vae=ae, scheduler=scheduler,
        image_processor=image_processor, image_encoder=image_encoder,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe
