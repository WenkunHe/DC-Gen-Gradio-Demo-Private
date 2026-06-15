"""
DC-Qwen-Image-Edit pipeline builder.
Checkpoints auto-downloaded from HuggingFace on first use.
Files live under upload/DC-Qwen-Image-Edit/DC-Gen-Qwen-Image-Edit/ in the repo;
they are moved to pretrained_models/dc_qwen_edit/ (prefix stripped) after download.
"""

import os
import pathlib
import shutil
import sys
import torch
from huggingface_hub import snapshot_download

HUB_REPO_QWEN_EDIT = 'nvidia/DC-Qwen-Image-Edit'

_repo_root = pathlib.Path(__file__).resolve().parent
CKPT = _repo_root / 'pretrained_models' / 'dc_qwen_edit'

_SENTINEL = 'model_index.json'


def _ensure_qwen_edit_ckpt() -> pathlib.Path:
    if (CKPT / _SENTINEL).exists():
        return CKPT

    token = os.environ.get('HF_TOKEN')
    print(f'[QwenEdit] Downloading from {HUB_REPO_QWEN_EDIT} ...')
    CKPT.mkdir(parents=True, exist_ok=True)

    # Download with actual files (not symlinks) directly into CKPT.
    # Files land at CKPT/upload/DC-Qwen-Image-Edit/DC-Gen-Qwen-Image-Edit/**
    snapshot_download(
        repo_id=HUB_REPO_QWEN_EDIT,
        repo_type='model',
        local_dir=str(CKPT),
        local_dir_use_symlinks=False,
        token=token,
    )

    # Find model_index.json under CKPT (walk follows real files, no symlink issues)
    src = None
    for root, dirs, files in os.walk(str(CKPT)):
        if _SENTINEL in files:
            src = pathlib.Path(root)
            break

    if src is None or src == CKPT:
        # Already at the right place or not found — either way we're done
        return CKPT

    print(f'[QwenEdit] Moving model from {src.relative_to(CKPT)} to root ...')
    for item in src.iterdir():
        dst = CKPT / item.name
        if dst.exists():
            shutil.rmtree(str(dst)) if dst.is_dir() else dst.unlink()
        shutil.move(str(item), str(dst))

    # Remove leftover upload/ scaffold
    upload_dir = CKPT / src.relative_to(CKPT).parts[0]
    if upload_dir.exists() and upload_dir != CKPT:
        shutil.rmtree(str(upload_dir))

    return CKPT


def build_qwen_edit_pipeline():
    print('[QwenEdit] Building pipeline...')
    ckpt = _ensure_qwen_edit_ckpt()

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
