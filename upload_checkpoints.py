#!/usr/bin/env python
"""Upload all DC-Gen checkpoints to dc-ai/dc-gen-checkpoints.

Each model is placed in a named subdirectory:
  DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K/
  DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K/
  DC-Gen-Wan2.1-14B-720P/
  DC-Gen-Qwen-Image-Edit-Res1K/

Run:
  HF_TOKEN=<token> python upload_checkpoints.py
"""

import os
import pathlib
import shutil
import tempfile

from huggingface_hub import HfApi, snapshot_download

DST_REPO = 'dc-ai/dc-gen-checkpoints'

SRC_FLUX      = 'nvidia/DC-Gen-FLUX.1-Krea-Dev'
SRC_VIDEOGEN  = 'nvidia/DC-VideoGen-Wan2.1-14B'
SRC_QWEN_EDIT = 'nvidia/DC-Qwen-Image-Edit'

SUBDIRS_FLUX = [
    'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K',
    'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K',
]
SUBDIR_VIDEO = 'DC-Gen-Wan2.1-14B-720P'
SUBDIR_EDIT  = 'DC-Gen-Qwen-Image-Edit-Res1K'

# Nested path inside nvidia/DC-Qwen-Image-Edit where the actual model lives
_QWEN_NESTED = pathlib.Path('upload') / 'DC-Qwen-Image-Edit' / 'DC-Gen-Qwen-Image-Edit'


def _token():
    t = os.environ.get('HF_TOKEN')
    if not t:
        raise RuntimeError('HF_TOKEN not set')
    return t


def upload_flux(api: HfApi, tmp: pathlib.Path):
    for subdir in SUBDIRS_FLUX:
        print(f'\n[FLUX] Downloading {SRC_FLUX}/{subdir} ...')
        local = tmp / subdir
        snapshot_download(
            repo_id=SRC_FLUX,
            repo_type='model',
            local_dir=str(local),
            allow_patterns=f'{subdir}/**',
            token=_token(),
        )
        # Strip the extra nesting that snapshot_download adds
        nested = local / subdir
        if nested.exists():
            for item in nested.iterdir():
                dst = local / item.name
                if dst.exists():
                    shutil.rmtree(str(dst)) if dst.is_dir() else dst.unlink()
                shutil.move(str(item), str(dst))
            shutil.rmtree(str(nested))
        print(f'[FLUX] Uploading to {DST_REPO}/{subdir}/ ...')
        api.upload_folder(
            repo_id=DST_REPO,
            repo_type='model',
            folder_path=str(local),
            path_in_repo=subdir,
            token=_token(),
        )
        print(f'[FLUX] Done: {subdir}')


def upload_videogen(api: HfApi, tmp: pathlib.Path):
    print(f'\n[VideoGen] Downloading {SRC_VIDEOGEN} ...')
    local = tmp / SUBDIR_VIDEO
    snapshot_download(
        repo_id=SRC_VIDEOGEN,
        repo_type='model',
        local_dir=str(local),
        token=_token(),
    )
    print(f'[VideoGen] Uploading to {DST_REPO}/{SUBDIR_VIDEO}/ ...')
    api.upload_folder(
        repo_id=DST_REPO,
        repo_type='model',
        folder_path=str(local),
        path_in_repo=SUBDIR_VIDEO,
        token=_token(),
    )
    print(f'[VideoGen] Done: {SUBDIR_VIDEO}')


def upload_qwen_edit(api: HfApi, tmp: pathlib.Path):
    print(f'\n[QwenEdit] Downloading {SRC_QWEN_EDIT} ...')
    local_raw = tmp / 'qwen_edit_raw'
    snapshot_download(
        repo_id=SRC_QWEN_EDIT,
        repo_type='model',
        local_dir=str(local_raw),
        local_dir_use_symlinks=False,
        token=_token(),
    )
    # Unwrap nested path if present
    nested = local_raw / _QWEN_NESTED
    if nested.exists():
        local = tmp / SUBDIR_EDIT
        shutil.copytree(str(nested), str(local))
    else:
        local = local_raw
    print(f'[QwenEdit] Uploading to {DST_REPO}/{SUBDIR_EDIT}/ ...')
    api.upload_folder(
        repo_id=DST_REPO,
        repo_type='model',
        folder_path=str(local),
        path_in_repo=SUBDIR_EDIT,
        token=_token(),
    )
    print(f'[QwenEdit] Done: {SUBDIR_EDIT}')


def main():
    api = HfApi()
    try:
        api.repo_info(repo_id=DST_REPO, repo_type='model', token=_token())
        print(f'Repo {DST_REPO} already exists.')
    except Exception:
        print(f'Creating repo {DST_REPO} ...')
        api.create_repo(repo_id=DST_REPO, repo_type='model', private=False, token=_token())

    with tempfile.TemporaryDirectory(prefix='dcgen_upload_') as tmp:
        tmp = pathlib.Path(tmp)
        upload_flux(api, tmp)
        upload_videogen(api, tmp)
        upload_qwen_edit(api, tmp)

    print(f'\nAll done. Checkpoints available at: https://huggingface.co/{DST_REPO}')


if __name__ == '__main__':
    main()
