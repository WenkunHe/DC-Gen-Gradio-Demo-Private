"""
DC-Qwen-Image-Edit pipeline builder.
Checkpoints auto-downloaded from HuggingFace on first use.
Files live under upload/DC-Qwen-Image-Edit/DC-Gen-Qwen-Image-Edit/ in the repo;
they are downloaded flat into pretrained_models/dc_qwen_edit/.
"""

import os
import pathlib
import sys
import torch
from huggingface_hub import HfApi, hf_hub_download

HUB_REPO_QWEN_EDIT = 'nvidia/DC-Qwen-Image-Edit'
_HUB_PREFIX = 'upload/DC-Qwen-Image-Edit/DC-Gen-Qwen-Image-Edit/'

_repo_root = pathlib.Path(__file__).resolve().parent
CKPT = _repo_root / 'pretrained_models' / 'dc_qwen_edit'

_SENTINEL = 'model_index.json'


def _ensure_qwen_edit_ckpt() -> pathlib.Path:
    if (CKPT / _SENTINEL).exists():
        return CKPT

    token = os.environ.get('HF_TOKEN')
    print(f'[QwenEdit] Downloading from {HUB_REPO_QWEN_EDIT} ...')
    CKPT.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    repo_files = [
        f for f in api.list_repo_files(HUB_REPO_QWEN_EDIT, repo_type='model')
        if f.startswith(_HUB_PREFIX)
    ]

    for repo_path in repo_files:
        local_rel = repo_path[len(_HUB_PREFIX):]   # strip the upload prefix
        local_path = CKPT / local_rel
        if local_path.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'  {local_rel}')
        hf_hub_download(
            repo_id=HUB_REPO_QWEN_EDIT,
            repo_type='model',
            filename=repo_path,
            local_dir=str(CKPT),
            token=token,
        )
        # hf_hub_download preserves the full path; move to flat location
        downloaded = CKPT / repo_path
        if downloaded.exists() and downloaded != local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(local_path)

    # Clean up the leftover upload/ directory tree
    upload_dir = CKPT / 'upload'
    if upload_dir.exists():
        import shutil
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
