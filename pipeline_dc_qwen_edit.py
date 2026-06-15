"""
DC-Qwen-Image-Edit pipeline builder.
Checkpoints auto-downloaded from HuggingFace on first use.
"""

import os
import pathlib
import sys
import torch
from huggingface_hub import snapshot_download

HUB_REPO_QWEN_EDIT = 'nvidia/DC-Qwen-Image-Edit'

_repo_root = pathlib.Path(__file__).resolve().parent
CKPT = _repo_root / 'pretrained_models' / 'dc_qwen_edit'

_REQUIRED_CKPT_PATHS = [
    'model_index.json',
    'transformer/config.json',
    'text_encoder/config.json',
    'vae/config.json',
]


def _ensure_qwen_edit_ckpt() -> pathlib.Path:
    if not all((CKPT / f).exists() for f in _REQUIRED_CKPT_PATHS):
        missing = [f for f in _REQUIRED_CKPT_PATHS if not (CKPT / f).exists()]
        print(f'[QwenEdit] Missing: {missing}')
        print(f'[QwenEdit] Downloading from {HUB_REPO_QWEN_EDIT} ...')
        CKPT.mkdir(parents=True, exist_ok=True)
        token = os.environ.get('HF_TOKEN')
        snapshot_download(
            repo_id=HUB_REPO_QWEN_EDIT,
            repo_type='model',
            local_dir=str(CKPT),
            token=token,
            ignore_patterns=['*.py'],  # we bundle the pipeline file ourselves
        )
    return CKPT


def build_qwen_edit_pipeline():
    print('[QwenEdit] Building pipeline...')
    ckpt = _ensure_qwen_edit_ckpt()

    # Add the checkpoint dir to sys.path so from_pretrained can find the
    # custom pipeline_qwen_image_edit.py bundled in the repo.
    pipeline_dir = pathlib.Path(__file__).resolve().parent
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))

    from pipeline_qwen_image_edit import DCQwenImageEditPipeline

    pipe = DCQwenImageEditPipeline.from_pretrained(
        str(ckpt),
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe
