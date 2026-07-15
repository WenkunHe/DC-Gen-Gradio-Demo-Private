#!/usr/bin/env python
"""Upload all DC-Gen checkpoints to dc-ai/dc-gen-checkpoints.

Each model is placed in a named subdirectory:
  DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K/
  DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K/
  DC-Gen-Wan2.1-14B-720P/
  DC-Gen-Qwen-Image-Edit-Res1K/

Run:
  HF_TOKEN_SRC=<nvidia_token> HF_TOKEN_DST=<dc-ai_token> python upload_checkpoints.py
"""

import os
import pathlib
import shutil
import subprocess
import sys

from huggingface_hub import HfApi

# Stage on lustre (not /tmp which is RAM-backed) so large models don't OOM
_STAGE_DIR = pathlib.Path(__file__).resolve().parent / '_upload_stage'

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

_QWEN_NESTED = pathlib.Path('upload') / 'DC-Qwen-Image-Edit' / 'DC-Gen-Qwen-Image-Edit'

_HF_CLI = shutil.which('huggingface-cli') or 'huggingface-cli'


def _token_src():
    t = os.environ.get('HF_TOKEN_SRC') or os.environ.get('HF_TOKEN')
    if not t:
        raise RuntimeError('HF_TOKEN_SRC (or HF_TOKEN) not set')
    return t

def _token_dst():
    t = os.environ.get('HF_TOKEN_DST') or os.environ.get('HF_TOKEN')
    if not t:
        raise RuntimeError('HF_TOKEN_DST (or HF_TOKEN) not set')
    return t


def _cli_download(repo_id: str, local_dir: pathlib.Path, token: str,
                  include: str | None = None):
    """Run huggingface-cli download in a subprocess with XET disabled.

    HF_HUB_DISABLE_XET=1 forces plain HTTP streaming instead of the XET
    Rust extension, which mmap's chunks and exhausts the 8 GB ulimit -v.
    Retries on failure because .incomplete files are resumed automatically.
    """
    env = os.environ.copy()
    env['HF_HUB_DISABLE_XET'] = '1'
    cmd = [
        _HF_CLI, 'download', repo_id,
        '--local-dir', str(local_dir),
        '--local-dir-use-symlinks', 'False',
        '--max-workers', '1',
        '--resume-download',
        '--token', token,
    ]
    if include:
        cmd += ['--include', include]
    print(f'  Running: {" ".join(cmd[:6])} ...')
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f'huggingface-cli download failed with exit code {result.returncode}')


def _strip_nesting(local: pathlib.Path, nested_name: str):
    """Move contents of local/nested_name/ up to local/."""
    nested = local / nested_name
    if not nested.exists():
        return
    for item in nested.iterdir():
        dst = local / item.name
        if dst.exists():
            shutil.rmtree(str(dst)) if dst.is_dir() else dst.unlink()
        shutil.move(str(item), str(dst))
    shutil.rmtree(str(nested))


def upload_flux(api: HfApi):
    for subdir in SUBDIRS_FLUX:
        local = _STAGE_DIR / subdir
        if not (local / 'model_index.json').exists():
            print(f'\n[FLUX] Downloading {SRC_FLUX}/{subdir} ...')
            local.mkdir(parents=True, exist_ok=True)
            _cli_download(
                repo_id=SRC_FLUX,
                local_dir=local,
                token=_token_src(),
                include=f'{subdir}/**',
            )
            _strip_nesting(local, subdir)
        else:
            print(f'\n[FLUX] {subdir} already staged, skipping download.')

        if not (local / 'model_index.json').exists():
            raise RuntimeError(f'model_index.json missing after download in {local}')

        print(f'[FLUX] Uploading to {DST_REPO}/{subdir}/ ...')
        api.upload_folder(
            repo_id=DST_REPO,
            repo_type='model',
            folder_path=str(local),
            path_in_repo=subdir,
            token=_token_dst(),
        )
        print(f'[FLUX] Done: {subdir}')


def upload_videogen(api: HfApi):
    local = _STAGE_DIR / SUBDIR_VIDEO
    if not (local / 'dc-ae-v-f32t4c32-1.0-bf16.pt').exists():
        print(f'\n[VideoGen] Downloading {SRC_VIDEOGEN} ...')
        local.mkdir(parents=True, exist_ok=True)
        _cli_download(
            repo_id=SRC_VIDEOGEN,
            local_dir=local,
            token=_token_src(),
        )
    else:
        print(f'\n[VideoGen] Already staged, skipping download.')
    print(f'[VideoGen] Uploading to {DST_REPO}/{SUBDIR_VIDEO}/ ...')
    api.upload_folder(
        repo_id=DST_REPO,
        repo_type='model',
        folder_path=str(local),
        path_in_repo=SUBDIR_VIDEO,
        token=_token_dst(),
    )
    print(f'[VideoGen] Done: {SUBDIR_VIDEO}')


def upload_qwen_edit(api: HfApi):
    local = _STAGE_DIR / SUBDIR_EDIT
    if not (local / 'model_index.json').exists():
        print(f'\n[QwenEdit] Downloading {SRC_QWEN_EDIT} ...')
        local_raw = _STAGE_DIR / 'qwen_edit_raw'
        local_raw.mkdir(parents=True, exist_ok=True)
        _cli_download(
            repo_id=SRC_QWEN_EDIT,
            local_dir=local_raw,
            token=_token_src(),
        )
        nested = local_raw / _QWEN_NESTED
        if nested.exists():
            shutil.copytree(str(nested), str(local))
        else:
            local.mkdir(parents=True, exist_ok=True)
            for item in local_raw.iterdir():
                shutil.move(str(item), str(local / item.name))
    else:
        print(f'\n[QwenEdit] Already staged, skipping download.')
    print(f'[QwenEdit] Uploading to {DST_REPO}/{SUBDIR_EDIT}/ ...')
    api.upload_folder(
        repo_id=DST_REPO,
        repo_type='model',
        folder_path=str(local),
        path_in_repo=SUBDIR_EDIT,
        token=_token_dst(),
    )
    print(f'[QwenEdit] Done: {SUBDIR_EDIT}')


def main():
    _STAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Staging to: {_STAGE_DIR}')

    api = HfApi()
    try:
        api.repo_info(repo_id=DST_REPO, repo_type='model', token=_token_dst())
        print(f'Repo {DST_REPO} already exists.')
    except Exception:
        print(f'Creating repo {DST_REPO} ...')
        api.create_repo(repo_id=DST_REPO, repo_type='model', private=False, token=_token_dst())

    upload_flux(api)
    upload_videogen(api)
    upload_qwen_edit(api)

    print(f'\nAll done. Checkpoints available at: https://huggingface.co/{DST_REPO}')
    print(f'Staging dir can be removed: rm -rf {_STAGE_DIR}')


if __name__ == '__main__':
    main()
