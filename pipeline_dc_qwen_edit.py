"""
DC-Qwen-Image-Edit pipeline builder.
Checkpoints auto-downloaded from HuggingFace on first use.
"""

import os
import pathlib
import shutil
import sys
import torch
from huggingface_hub import snapshot_download

HUB_REPO_QWEN_EDIT = 'dc-ai/dc-gen-checkpoints'
_HUB_SUBDIR        = 'DC-Gen-Qwen-Image-Edit-Res1K'

_repo_root = pathlib.Path(__file__).resolve().parent
CKPT = _repo_root / 'pretrained_models' / _HUB_SUBDIR

# All of these must exist for the checkpoint to be considered complete.
_REQUIRED = [
    'model_index.json',
    'scheduler/scheduler_config.json',
    'transformer/config.json',
    'vae/config.json',
]


def _ckpt_complete() -> bool:
    return all((CKPT / f).exists() for f in _REQUIRED)


def _ensure_qwen_edit_ckpt() -> pathlib.Path:
    if _ckpt_complete():
        return CKPT

    if CKPT.exists():
        print(f'[QwenEdit] Removing incomplete download at {CKPT} ...')
        shutil.rmtree(str(CKPT))

    token = os.environ.get('HF_TOKEN')
    print(f'[QwenEdit] Downloading from {HUB_REPO_QWEN_EDIT}/{_HUB_SUBDIR}/ ...')
    CKPT.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HUB_REPO_QWEN_EDIT,
        repo_type='model',
        local_dir=str(CKPT),
        allow_patterns=f'{_HUB_SUBDIR}/**',
        local_dir_use_symlinks=False,
        token=token,
    )

    # Strip extra nesting that snapshot_download adds
    nested = CKPT / _HUB_SUBDIR
    if nested.exists():
        for item in nested.iterdir():
            dst = CKPT / item.name
            if dst.exists():
                shutil.rmtree(str(dst)) if dst.is_dir() else dst.unlink()
            shutil.move(str(item), str(dst))
        nested.rmdir()

    if not _ckpt_complete():
        raise RuntimeError(
            f'model_index.json not found at {CKPT}. '
            f'Contents: {[p.name for p in CKPT.iterdir()]}'
        )
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
