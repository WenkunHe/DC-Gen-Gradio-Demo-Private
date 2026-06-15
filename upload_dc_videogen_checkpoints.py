#!/usr/bin/env python3
"""
Upload DC-VideoGen checkpoints to HuggingFace.

Usage:
    HF_TOKEN=<your_token> python upload_dc_videogen_checkpoints.py

Files uploaded to nvidia/DC-VideoGen-Wan2.1-14B:
  dc-ae-v-f32t4c32-1.0-bf16.pt                        (~4 GB)
  dc_videogen_wan2.1_t2v_14b_720p.pt                   (~27 GB)
  dc_videogen_wan2.1_i2v_14b_720p.pt                   (~31 GB)
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  (~4.5 GB)
  text_encoder/   (UMT5 ~11 GB)
  tokenizer/      (~21 MB)
  scheduler/      (~8 KB)
"""

import os
import pathlib
import sys

from huggingface_hub import HfApi, create_repo

# ── source paths (edit if different on your machine) ──────────────────────────
CKPT_DIR  = pathlib.Path('/home/hcai/workspace/code/dc-dev-videogen-fix/DC-VideoGen-Wan2.1-14B-Diffusers/checkpoints')
CLIP_PATH = pathlib.Path('/home/hcai/workspace/code/dc-dev-videogen-fix/dc-dev/assets/checkpoints/i2v/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')
FUSIONX   = pathlib.Path('/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/fusionx/diffuser_checkpoints')

HUB_REPO  = 'nvidia/DC-VideoGen-Wan2.1-14B'

# ── auth ──────────────────────────────────────────────────────────────────────
token = os.environ.get('HF_TOKEN')
if not token:
    sys.exit('Error: set HF_TOKEN environment variable before running.')

api = HfApi(token=token)

# ── create repo (no-op if already exists) ─────────────────────────────────────
print(f'Creating/verifying repo {HUB_REPO} ...')
create_repo(HUB_REPO, repo_type='model', exist_ok=True, token=token, private=False)

# ── upload helper ─────────────────────────────────────────────────────────────
def upload(local_path: pathlib.Path, repo_path: str):
    print(f'Uploading {local_path.name} -> {repo_path} ...')
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=HUB_REPO,
        repo_type='model',
        token=token,
    )
    print(f'  done: {repo_path}')

def upload_folder(local_dir: pathlib.Path, repo_prefix: str):
    print(f'Uploading folder {local_dir.name}/ -> {repo_prefix}/ ...')
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=repo_prefix,
        repo_id=HUB_REPO,
        repo_type='model',
        token=token,
    )
    print(f'  done: {repo_prefix}/')

# ── checkpoint files ──────────────────────────────────────────────────────────
upload(CKPT_DIR / 'dc-ae-v-f32t4c32-1.0-bf16.pt',           'dc-ae-v-f32t4c32-1.0-bf16.pt')
upload(CKPT_DIR / 'dc_videogen_wan2.1_t2v_14b_720p.pt',      'dc_videogen_wan2.1_t2v_14b_720p.pt')
upload(CKPT_DIR / 'dc_videogen_wan2.1_i2v_14b_720p.pt',      'dc_videogen_wan2.1_i2v_14b_720p.pt')
upload(CLIP_PATH,                                             'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')

# ── text encoder / tokenizer / scheduler (from FUSIONX) ──────────────────────
upload_folder(FUSIONX / 'text_encoder', 'text_encoder')
upload_folder(FUSIONX / 'tokenizer',    'tokenizer')
upload_folder(FUSIONX / 'scheduler',    'scheduler')

print(f'\nAll files uploaded to https://huggingface.co/{HUB_REPO}')
