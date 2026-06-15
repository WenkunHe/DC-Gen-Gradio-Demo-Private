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
# Path inside the HF repo where the actual model lives
_HUB_MODEL_SUBDIR = pathlib.Path('upload') / 'DC-Qwen-Image-Edit' / 'DC-Gen-Qwen-Image-Edit'

_repo_root = pathlib.Path(__file__).resolve().parent
CKPT = _repo_root / 'pretrained_models' / 'dc_qwen_edit'

_SENTINEL = 'model_index.json'


def _ensure_qwen_edit_ckpt() -> pathlib.Path:
    if (CKPT / _SENTINEL).exists():
        return CKPT

    # Previous failed attempts may have left garbage in CKPT that confuses
    # snapshot_download into skipping files it thinks are already present.
    # The actual file data lives in the global HF cache, so wiping CKPT and
    # re-running snapshot_download just re-copies from cache (fast).
    if CKPT.exists():
        print(f'[QwenEdit] Removing incomplete download at {CKPT} ...')
        shutil.rmtree(str(CKPT))

    token = os.environ.get('HF_TOKEN')
    print(f'[QwenEdit] Downloading from {HUB_REPO_QWEN_EDIT} ...')
    CKPT.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HUB_REPO_QWEN_EDIT,
        repo_type='model',
        local_dir=str(CKPT),
        local_dir_use_symlinks=False,
        token=token,
    )

    # Case 1: HF repo already reorganised — files landed at CKPT root.
    if (CKPT / _SENTINEL).exists():
        return CKPT

    # Case 2: files still nested at upload/.../DC-Gen-Qwen-Image-Edit/
    nested = CKPT / _HUB_MODEL_SUBDIR
    if not (nested / _SENTINEL).exists():
        raise RuntimeError(
            f'model_index.json not found at {CKPT} or {nested}. '
            f'Top-level contents: {[p.name for p in CKPT.iterdir()]}'
        )

    print(f'[QwenEdit] Moving files from {_HUB_MODEL_SUBDIR} to CKPT root ...')
    for item in nested.iterdir():
        dst = CKPT / item.name
        if dst.exists():
            shutil.rmtree(str(dst)) if dst.is_dir() else dst.unlink()
        shutil.move(str(item), str(dst))

    shutil.rmtree(str(CKPT / 'upload'))
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
