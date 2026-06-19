#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import uuid

import gradio as gr
import torch
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from PIL import Image

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
from pipeline_dc_videogen import build_t2v_pipeline, build_i2v_pipeline  # noqa: E402
from pipeline_dc_qwen_edit import build_qwen_edit_pipeline  # noqa: E402

INTRODUCTION = """
# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

DC-Gen adapts high-resolution visual generation and editing models (e.g., FLUX, Wan2.1, Qwen-Image-Edit) to deeply compressed latent spaces through efficient post-training. It enables native 4K image synthesis, and achieves up to 54x acceleration.
"""

HUB_REPO          = 'nvidia/DC-Gen-FLUX.1-Krea-Dev'
HUB_REPO_1K        = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K'
HUB_REPO_1K_ANYFLOW = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K-Anyflow'
HUB_REPO_4K        = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K'
HUB_REPO_4K_ANYFLOW = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K-Anyflow'

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument('--load_before_generation', action='store_true',
                     help='Load all models before serving requests (requires ≥6 GPUs).')
_args, _ = _parser.parse_known_args()

# ── Multi-GPU detection ───────────────────────────────────────────────────────
_NUM_GPUS = torch.cuda.device_count()
MULTI_GPU = _args.load_before_generation and _NUM_GPUS >= 6
if MULTI_GPU:
    print(f'[Multi-GPU] {_NUM_GPUS} GPUs detected — models will be loaded persistently.')
    _DEV_EXP  = 'cuda:0'   # prompt expander  (Qwen2.5-14B)
    _DEV_1K   = 'cuda:1'   # DC-Gen-FLUX 1K
    _DEV_4K   = 'cuda:2'   # DC-Gen-FLUX 4K
    _DEV_T2V  = 'cuda:3'   # Wan2.1 T2V
    _DEV_I2V  = 'cuda:4'   # Wan2.1 I2V
    _DEV_EDIT = 'cuda:5'   # Qwen-Image-Edit
elif _args.load_before_generation:
    print(f'[Warning] --load_before_generation requires ≥6 GPUs; only {_NUM_GPUS} found — using lazy load/unload.')
    _DEV_EXP = _DEV_1K = _DEV_4K = _DEV_T2V = _DEV_I2V = _DEV_EDIT = 'cuda'
else:
    _DEV_EXP = _DEV_1K = _DEV_4K = _DEV_T2V = _DEV_I2V = _DEV_EDIT = 'cuda'

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

ASPECT_RATIOS_4K = {
    '1:1  (4096×4096)': (4096, 4096),
    '4:3  (4608×3328)': (4608, 3328),
    '3:4  (3328×4608)': (3328, 4608),
    '16:9 (5376×3072)': (5376, 3072),
    '9:16 (3072×5376)': (3072, 5376),
    '9:7  (4608×3584)': (4608, 3584),
    '7:9  (3584×4608)': (3584, 4608),
}

EXAMPLES_1K = [
    "A well-groomed man with short, sleek brown hair and fashionable eyeglasses smiles warmly into the camera in this polished professional portrait.",
    "Anime style. A graceful anime girl with flowing silver hair adorned with a delicate blue ribbon stands in a serene outdoor scene, a vibrant blue butterfly perched on her shoulder.",
    "A serene landscape where rolling green mountains cradle a tranquil lake, their peaks mirrored perfectly on the glassy surface. Mist drifts gently along the slopes.",
]

EXAMPLES_4K = [
    "A serene landscape where rolling green mountains cradle a tranquil lake, mist drifting along the slopes at dawn.",
    "A well-groomed man with short brown hair and glasses smiling in a polished professional portrait, navy suit, soft lighting.",
    "Anime style. A graceful girl with flowing silver hair and a blue ribbon, pastel sky, drifting butterflies.",
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
    # Patch the cloned AnyFlow with the correct DC-Gen custom versions bundled in
    # this repo under custom_code/.  Do NOT rely on the checkpoint's custom_code/
    # subdirectory — snapshot_download with allow_patterns='{subdir}/*' only
    # matches one directory level deep and may not download nested subdirs.
    import shutil
    custom_code_dir = repo_root / 'custom_code'
    # Delete __pycache__ dirs first so Python recompiles from the freshly-copied .py
    # files rather than using stale bytecode that predates our edits.
    for _pycache in (
        anyflow_src / 'far' / 'models' / '__pycache__',
        anyflow_src / 'far' / 'schedulers' / '__pycache__',
        anyflow_src / 'far' / 'pipelines' / '__pycache__',
    ):
        if _pycache.exists():
            shutil.rmtree(str(_pycache))
    shutil.copy(custom_code_dir / 'transformer_dcgen_flux_model.py',
                anyflow_src / 'far' / 'models' / 'transformer_dcgen_flux_model.py')
    shutil.copy(custom_code_dir / 'scheduling_flowmap_euler_discrete.py',
                anyflow_src / 'far' / 'schedulers' / 'scheduling_flowmap_euler_discrete.py')
    shutil.copy(custom_code_dir / 'pipeline_dcgen_flux_anyflow.py',
                anyflow_src / 'far' / 'pipelines' / 'pipeline_dcgen_flux_anyflow.py')

    # Invalidate module cache so the freshly-copied files are picked up
    for mod in list(sys.modules.keys()):
        if mod.startswith('far.'):
            del sys.modules[mod]

    from far.pipelines.pipeline_dcgen_flux_anyflow import DCGenFluxAnyFlowPipeline  # noqa: E402
    from far.models.transformer_dcgen_flux_model import DCGenFluxFlowMapModel  # noqa: E402
    from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler  # noqa: E402
    from diffusers import AutoencoderDC
    from transformers import CLIPTextModel, PreTrainedTokenizerFast, T5EncoderModel

    dtype = torch.bfloat16

    # Load each component individually to avoid diffusers' custom-library issubclass check
    vae           = AutoencoderDC.from_pretrained(local_dir / 'vae', torch_dtype=dtype)
    # Load directly from tokenizer.json to avoid tokenizer_config.json version mismatches
    # (extra_special_tokens saved as list instead of dict in older transformers).
    # Set pad/eos/bos tokens explicitly since tokenizer_config.json is skipped.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(local_dir / 'tokenizer' / 'tokenizer.json'),
        model_max_length=77,
        pad_token='<|endoftext|>',
        eos_token='<|endoftext|>',
        bos_token='<|startoftext|>',
        unk_token='<|endoftext|>',
    )
    tokenizer_2 = PreTrainedTokenizerFast(
        tokenizer_file=str(local_dir / 'tokenizer_2' / 'tokenizer.json'),
        model_max_length=512,
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',  # id=0, same as T5TokenizerFast.from_pretrained default
    )
    text_encoder  = CLIPTextModel.from_pretrained(local_dir / 'text_encoder', torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(local_dir / 'text_encoder_2', torch_dtype=dtype)

    # Build model with correct architecture first, then load sharded weights manually.
    # from_pretrained can't be used because register_to_config drops config attrs when
    # __init__ is overridden, and setup_flowmap_model() must run before weights load.
    import json
    from safetensors.torch import load_file as _load_sf
    _t_dir = local_dir / 'transformer'
    transformer = DCGenFluxFlowMapModel.from_config(DCGenFluxFlowMapModel.load_config(str(_t_dir)))
    transformer.setup_flowmap_model(gate_value=0.25, deltatime_type='r')
    _index_path = _t_dir / 'diffusion_pytorch_model.safetensors.index.json'
    if _index_path.exists():
        with open(_index_path) as _f:
            _shards = sorted(set(json.load(_f)['weight_map'].values()))
        _state_dict = {}
        for _shard in _shards:
            _state_dict.update(_load_sf(str(_t_dir / _shard)))
    elif (_t_dir / 'diffusion_pytorch_model.safetensors').exists():
        _state_dict = _load_sf(str(_t_dir / 'diffusion_pytorch_model.safetensors'))
    else:
        _state_dict = torch.load(str(_t_dir / 'diffusion_pytorch_model.bin'), map_location='cpu', weights_only=True)
    _missing, _unexpected = transformer.load_state_dict(_state_dict, strict=False)
    if _missing:
        print(f'[transformer] {len(_missing)} missing keys')
    transformer = transformer.to(dtype).eval()

    # Instantiate scheduler directly with the correct shift (generate.py uses shift=3.0)
    scheduler = FlowMapDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

    # Keys in the .pth are 'prompt_embeds'/'pooled_prompt_embeds' (not 'null_*')
    null_embeds = torch.load(local_dir / 'flux_null_embedding.pth', map_location='cpu', weights_only=False)
    _null_pe  = null_embeds.get('prompt_embeds')
    if _null_pe is None:
        _null_pe = null_embeds.get('null_prompt_embeds')
    _null_ppe = null_embeds.get('pooled_prompt_embeds')
    if _null_ppe is None:
        _null_ppe = null_embeds.get('null_pooled_prompt_embeds')
    if _null_pe is not None:
        _null_pe  = _null_pe.to(dtype)
    if _null_ppe is not None:
        _null_ppe = _null_ppe.to(dtype)

    pipe = DCGenFluxAnyFlowPipeline(
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        scheduler=scheduler,
        null_prompt_embeds=_null_pe,
        null_pooled_prompt_embeds=_null_ppe,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ── All pipelines lazy-loaded on first use ────────────────────────────────────
_pipe_1k_anyflow  = None
_pipe_1k          = None
_pipe_4k_anyflow  = None
_pipe_4k          = None
_pipe_t2v         = None
_pipe_i2v         = None

def _get_pipe_1k_anyflow():
    global _pipe_1k_anyflow
    if _pipe_1k_anyflow is None:
        print('[Pipeline] Loading 1K-AnyFlow...')
        _pipe_1k_anyflow = load_pipeline_anyflow(HUB_REPO_1K_ANYFLOW)
    return _pipe_1k_anyflow

def _get_pipe_1k():
    global _pipe_1k
    if _pipe_1k is None:
        print('[Pipeline] Loading 1K...')
        _pipe_1k = load_pipeline_standard(HUB_REPO_1K)
        if MULTI_GPU:
            _pipe_1k.to(_DEV_1K)
    return _pipe_1k

def _get_pipe_4k_anyflow():
    global _pipe_4k_anyflow
    if _pipe_4k_anyflow is None:
        print('[Pipeline] Loading 4K-AnyFlow...')
        _pipe_4k_anyflow = load_pipeline_anyflow(HUB_REPO_4K_ANYFLOW)
    return _pipe_4k_anyflow

def _get_pipe_4k():
    global _pipe_4k
    if _pipe_4k is None:
        print('[Pipeline] Loading 4K...')
        _pipe_4k = load_pipeline_standard(HUB_REPO_4K)
        if MULTI_GPU:
            _pipe_4k.to(_DEV_4K)
    return _pipe_4k

def _get_t2v_pipe():
    global _pipe_t2v
    if _pipe_t2v is None:
        print('[Pipeline] Loading T2V...')
        _pipe_t2v = build_t2v_pipeline()
        if MULTI_GPU:
            _pipe_t2v.to(_DEV_T2V)
    return _pipe_t2v

def _get_i2v_pipe():
    global _pipe_i2v
    if _pipe_i2v is None:
        print('[Pipeline] Loading I2V...')
        _pipe_i2v = build_i2v_pipeline()
        if MULTI_GPU:
            _pipe_i2v.to(_DEV_I2V)
    return _pipe_i2v

_pipe_qwen_edit = None

def _get_qwen_edit_pipe():
    global _pipe_qwen_edit
    if _pipe_qwen_edit is None:
        print('[Pipeline] Loading DC-Qwen-Image-Edit...')
        _pipe_qwen_edit = build_qwen_edit_pipeline()
        if MULTI_GPU:
            _pipe_qwen_edit.to(_DEV_EDIT)
    return _pipe_qwen_edit


# ── Prompt expanders (lazy-loaded) ────────────────────────────────────────────
_expander_t2i  = None   # T2I 1K + 4K  (Qwen2.5-14B, text-only)
_expander_t2v  = None   # T2V           (Qwen2.5-14B, text-only)
_expander_i2v  = None   # I2V           (QwenVL2.5-7B, image-conditioned)
_expander_edit = None   # Image editing (QwenVL2.5-7B, image-conditioned)
_translator    = None   # Translate any language → English (Qwen2.5-14B, text-only)

def _get_expander_t2i():
    global _expander_t2i
    if _expander_t2i is None:
        print('[Expander] Loading T2I prompt expander...')
        from qwen_25_extend import QwenPromptExpander
        _expander_t2i = QwenPromptExpander(is_t2i=True)
        if MULTI_GPU:
            _expander_t2i.to(_DEV_EXP)
    return _expander_t2i

def _get_expander_t2v():
    global _expander_t2v
    if _expander_t2v is None:
        print('[Expander] Loading T2V prompt expander...')
        from qwen_25_extend import QwenPromptExpander
        _expander_t2v = QwenPromptExpander(is_t2v=True)
        if MULTI_GPU:
            _expander_t2v.to(_DEV_EXP)
    return _expander_t2v

def _get_expander_i2v():
    global _expander_i2v
    if _expander_i2v is None:
        print('[Expander] Loading I2V prompt expander...')
        from qwen_25_extend import QwenPromptExpander
        _expander_i2v = QwenPromptExpander(is_i2v=True)
        if MULTI_GPU:
            _expander_i2v.to(_DEV_EXP)
    return _expander_i2v

def _get_expander_edit():
    global _expander_edit
    if _expander_edit is None:
        print('[Expander] Loading image-edit prompt expander...')
        from qwen_25_extend import QwenPromptExpander
        _expander_edit = QwenPromptExpander(is_vl=True, is_edit=True)
        if MULTI_GPU:
            _expander_edit.to(_DEV_EXP)
    return _expander_edit

def _get_translator():
    global _translator
    if _translator is None:
        print('[Expander] Loading translator...')
        from qwen_25_extend import QwenPromptExpander
        _translator = QwenPromptExpander(is_t2i=True)
        if MULTI_GPU:
            _translator.to(_DEV_EXP)
    return _translator


def _preload_all_models():
    """Load all 6 models to their dedicated GPUs at startup (multi-GPU mode only)."""
    print('[Multi-GPU] Pre-loading all models...')
    _get_pipe_1k()
    _get_pipe_4k()
    _get_t2v_pipe()
    _get_i2v_pipe()
    _get_qwen_edit_pipe()
    # Expander uses Qwen2.5-14B; load t2i variant (covers t2i, t2v, translation)
    from qwen_25_extend import QwenPromptExpander
    _get_expander_t2i()
    print('[Multi-GPU] All models loaded.')

def _is_english(text):
    for char in text:
        if '一' <= char <= '鿿':   # CJK unified ideographs
            return False
        if '぀' <= char <= 'ヿ':   # Hiragana / Katakana
            return False
        if '가' <= char <= '힯':   # Hangul
            return False
    return True

def _expand_prompt(expander, prompt, image=None):
    # Step 1: translate to English if the prompt is not in English
    if not _is_english(prompt):
        from qwen_25_extend import TRANSLATE_TEXT_SYS_PROMPT
        translator = _get_translator()
        if not MULTI_GPU:
            translator.to('cuda')
        try:
            prompt = translator.extend(prompt, TRANSLATE_TEXT_SYS_PROMPT).prompt
        finally:
            if not MULTI_GPU:
                translator.to('cpu')
                torch.cuda.empty_cache()
    # Step 2: expand the (now-English) prompt
    if not MULTI_GPU:
        expander.to('cuda')
    try:
        result = expander(prompt, tar_lang='en', image=image)
        return result.prompt
    finally:
        if not MULTI_GPU:
            expander.to('cpu')
            torch.cuda.empty_cache()


VIDEO_RESOLUTIONS_T2V = {
    '720p portrait  (720×1280)':  (720,  1280),
    '720p landscape (1280×720)':  (1280,  720),
    '480p portrait  (480×832)':   (480,   832),
    '480p landscape (832×480)':   (832,   480),
}

VIDEO_NEGATIVE_PROMPT = (
    'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, '
    'static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, '
    'extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, '
    'fused fingers, still picture, messy background, three legs, many people in the background, '
    'walking backwards'
)

EXAMPLES_T2V = [
    'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. '
    'The fluffy-furred feline gazes directly at the camera with a relaxed expression. '
    'Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, '
    'and a blue sky dotted with white clouds.',
    'A lone astronaut walks across a vast red Martian landscape under a pale orange sky, '
    'dust swirling gently at their feet. Distant mountains loom on the horizon.',
]

_I2V_ASSETS = repo_root / 'assets' / 'i2v_examples'
EXAMPLES_I2V = [
    (
        str(_I2V_ASSETS / 'cat_surf.jpg'),
        'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. '
        'The fluffy-furred feline gazes directly at the camera with a relaxed expression. '
        'Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, '
        'and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, '
        'as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline\'s '
        'intricate details and the refreshing atmosphere of the seaside.',
    ),
    (
        str(_I2V_ASSETS / 'girl_cat.jpg'),
        'Hard lighting, high contrast, mid-close up shot, artificial side lighting, soft focus background, '
        'warm tones, a clean single-person frame. A blonde girl in her early twenties, wearing a light-colored '
        'knit top and jeans, gently strokes a large, pure white cat while squatting on the studio floor. '
        'The girl has delicate features and a gentle gaze, moving slowly. The cat has big round eyes that '
        'express affection as it blinks. She then hugs the cat, which rests its head on her chest, creating '
        'a natural interaction filled with a magical and dreamlike atmosphere.',
    ),
    (
        str(_I2V_ASSETS / 'robot.jpg'),
        'A towering, battle-scarred humanoid robot, reminiscent of a Transformer with powerful, segmented '
        'armor and glowing red optics, walking through the skeletal remains of a city ruin. Twisted metal '
        'and shattered concrete crunch under its heavy steps, as the robot scans the desolate, dust-choked '
        'skyline under a dark sky.',
    ),
    (
        str(_I2V_ASSETS / 'bicycle.jpg'),
        'A thoughtful teenage boy riding an old bicycle, aimlessly wandering. Parallel tracking shot captures '
        'his elongated shadow and fluttering clothes under the setting sun, evoking a sense of adolescent '
        'confusion, freedom, and gentle melancholy.',
    ),
]


_QWEN_EDIT_ASSETS = repo_root / 'assets' / 'qwen_edit_examples'
EXAMPLES_QWEN_EDIT = [
    (str(_QWEN_EDIT_ASSETS / 'cat.png'),   "Change the background to outside, and change 'CVPR' into 'ECCV'."),
    (str(_QWEN_EDIT_ASSETS / 'dog.png'),   '小狗换成粘土材质'),
    (str(_QWEN_EDIT_ASSETS / 'woman.png'), 'Replace the bracelet with a jade bangle.'),
    (str(_QWEN_EDIT_ASSETS / 'beard.png'), '让他的胡子更长'),
]


def _stream_generation(run_fn):
    """Run run_fn(tlog, ev_q, result) in a thread.
    Yields (None, timing, extended_prompt) after each stage update,
    then (path, timing, extended_prompt) at the end.
    run_fn appends timing lines to tlog and puts 'update' in ev_q after each stage.
    """
    tlog = []
    ev_q = queue.Queue()
    result = {'path': None, 'error': None, 'extended_prompt': ''}

    def _worker():
        try:
            run_fn(tlog, ev_q, result)
        except Exception as exc:
            result['error'] = exc
            tlog.append(f'[Error] {exc}')
        finally:
            ev_q.put('done')

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        msg = ev_q.get()
        if msg == 'done':
            break
        yield result['path'], '\n'.join(tlog), result['extended_prompt']

    if result['error']:
        raise result['error']
    yield result['path'], '\n'.join(tlog), result['extended_prompt']


def _tlog(tlog, ev_q, msg):
    print(msg)
    tlog.append(msg)
    ev_q.put('update')


def generate_1k(prompt: str, use_anyflow: str, aspect_ratio: str, num_steps: int, guidance: float, seed: int, use_expander: str = 'No'):
    h, w = ASPECT_RATIOS_1K[aspect_ratio]
    tag = 'anyflow' if use_anyflow == 'AnyFlow' else 'standard'
    print(f'\n[Generate 1K-{tag}] {h}x{w}, {num_steps} steps, seed={int(seed)}')
    _t0 = time.perf_counter()

    def _run(tlog, ev_q, result):
        nonlocal prompt
        if use_expander == 'Yes':
            _tlog(tlog, ev_q, '[Info]   Extending prompt...')
            prompt = _expand_prompt(_get_expander_t2i(), prompt)
            result['extended_prompt'] = prompt
            _tlog(tlog, ev_q, '[Info]   Prompt extended.')
        pipe = _get_pipe_1k_anyflow() if use_anyflow == 'AnyFlow' else _get_pipe_1k()
        dev = _DEV_1K
        if not MULTI_GPU:
            pipe.to(dev)
        torch.cuda.synchronize(dev)
        _t_load = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] GPU load (to CUDA) : {_t_load - _t0:.3f}s  (total {_t_load - _t0:.3f}s)')
        with torch.no_grad():
            kwargs = dict(
                height=h, width=w,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=torch.Generator(dev).manual_seed(int(seed)),
            )
            kwargs['timing_log'] = tlog
            kwargs['timing_event'] = ev_q
            if use_anyflow != 'AnyFlow':
                kwargs['use_flux_2'] = True
            out = pipe(prompt.strip(), **kwargs).images[0]
            torch.cuda.synchronize(dev)
            _t_gen = time.perf_counter()
        path = output_dir / f'1k_{tag}_{uuid.uuid4().hex[:8]}.jpg'
        out.save(str(path))
        _t_save = time.perf_counter()
        result['path'] = str(path)
        _tlog(tlog, ev_q, f'[Timing] Storing            : {_t_save - _t_gen:.3f}s  (total {_t_save - _t0:.3f}s)')
        if not MULTI_GPU:
            pipe.to('cpu')
            torch.cuda.empty_cache()
            _t_offload = time.perf_counter()
            _tlog(tlog, ev_q, f'[Timing] GPU unload (to CPU): {_t_offload - _t_save:.3f}s  (total {_t_offload - _t0:.3f}s)')

    yield from _stream_generation(_run)


def generate_4k(prompt: str, use_anyflow: str, aspect_ratio: str, num_steps: int, guidance: float, seed: int, use_expander: str = 'No'):
    h, w = ASPECT_RATIOS_4K[aspect_ratio]
    tag = 'anyflow' if use_anyflow == 'AnyFlow' else 'standard'
    print(f'\n[Generate 4K-{tag.upper()}] {h}x{w}, {num_steps} steps, seed={int(seed)}')
    _t0 = time.perf_counter()

    def _run(tlog, ev_q, result):
        nonlocal prompt
        if use_expander == 'Yes':
            _tlog(tlog, ev_q, '[Info]   Extending prompt...')
            prompt = _expand_prompt(_get_expander_t2i(), prompt)
            result['extended_prompt'] = prompt
            _tlog(tlog, ev_q, '[Info]   Prompt extended.')
        if use_anyflow == 'AnyFlow':
            pipe_4k = _get_pipe_4k_anyflow()
        else:
            pipe_4k = _get_pipe_4k()
        dev = _DEV_4K
        if not MULTI_GPU:
            pipe_4k.to(dev)
        torch.cuda.synchronize(dev)
        _t_load = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] GPU load (to CUDA) : {_t_load - _t0:.3f}s  (total {_t_load - _t0:.3f}s)')
        with torch.no_grad():
            if use_anyflow == 'AnyFlow':
                _timing = {}
                def _step_cb(pipeline, i, t, cb_kwargs):
                    if i == 0:
                        torch.cuda.synchronize(dev)
                        _timing['cond_end'] = time.perf_counter()
                    return cb_kwargs
                latents = pipe_4k(
                    prompt.strip(),
                    height=h, width=w,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    generator=torch.Generator(dev).manual_seed(int(seed)),
                    output_type='latent',
                    callback_on_step_end=_step_cb,
                    callback_on_step_end_tensor_inputs=['latents'],
                ).images
                torch.cuda.synchronize(dev)
                _t_denoise_end = time.perf_counter()
                _t_cond_end = _timing.get('cond_end', _t_load)
                _tlog(tlog, ev_q, f'[Timing] Condition encoding : {_t_cond_end - _t_load:.3f}s  (total {_t_cond_end - _t0:.3f}s)')
                _tlog(tlog, ev_q, f'[Timing] Denoising          : {_t_denoise_end - _t_cond_end:.3f}s  (total {_t_denoise_end - _t0:.3f}s)')
                scaling_factor = getattr(pipe_4k.vae.config, 'scaling_factor', 1.0)
                shift_factor = getattr(pipe_4k.vae.config, 'shift_factor', 0.0)
                decoded = pipe_4k.vae.decode((latents / scaling_factor) + shift_factor, return_dict=False)[0]
                out = pipe_4k.image_processor.postprocess(decoded, output_type='pil')[0]
                torch.cuda.synchronize(dev)
                _t_vae = time.perf_counter()
                _tlog(tlog, ev_q, f'[Timing] VAE decoding       : {_t_vae - _t_denoise_end:.3f}s  (total {_t_vae - _t0:.3f}s)')
            else:
                out = pipe_4k(
                    prompt.strip(),
                    height=h, width=w,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    generator=torch.Generator(dev).manual_seed(int(seed)),
                    use_flux_2=False,
                    timing_log=tlog,
                    timing_event=ev_q,
                ).images[0]
                torch.cuda.synchronize(dev)
                _t_vae = time.perf_counter()
        path = output_dir / f'4k_{tag}_{uuid.uuid4().hex[:8]}.jpg'
        out.save(str(path))
        _t_save = time.perf_counter()
        result['path'] = str(path)
        _tlog(tlog, ev_q, f'[Timing] Storing            : {_t_save - _t_vae:.3f}s  (total {_t_save - _t0:.3f}s)')
        if not MULTI_GPU:
            pipe_4k.to('cpu')
            torch.cuda.empty_cache()
            _t_offload = time.perf_counter()
            _tlog(tlog, ev_q, f'[Timing] GPU unload (to CPU): {_t_offload - _t_save:.3f}s  (total {_t_offload - _t0:.3f}s)')

    yield from _stream_generation(_run)


def generate_t2v(prompt: str, num_frames: int, num_steps: int, guidance: float, seed: int, use_expander: str = 'No'):
    h, w = 720, 1280
    print(f'\n[Generate T2V] {h}x{w}, {num_frames} frames, {num_steps} steps, seed={int(seed)}')
    _t0 = time.perf_counter()

    def _run(tlog, ev_q, result):
        nonlocal prompt
        if use_expander == 'Yes':
            _tlog(tlog, ev_q, '[Info]   Extending prompt...')
            prompt = _expand_prompt(_get_expander_t2v(), prompt)
            result['extended_prompt'] = prompt
            _tlog(tlog, ev_q, '[Info]   Prompt extended.')
        pipe = _get_t2v_pipe()
        dev = _DEV_T2V
        if not MULTI_GPU:
            pipe.to(dev)
        torch.cuda.synchronize(dev)
        _t_load = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] GPU load            : {_t_load - _t0:.3f}s  (total {_t_load - _t0:.3f}s)')
        _tlog(tlog, ev_q, f'[Info]   Generating {h}×{w}, {num_frames} frames ({num_steps} steps)...')

        _timing = {}
        def _step_cb(pipeline, i, t, cb_kwargs):
            if i == 0:
                torch.cuda.synchronize(dev)
                _timing['cond_end'] = time.perf_counter()
            return cb_kwargs

        with torch.no_grad():
            out = pipe(
                prompt=prompt.strip(),
                negative_prompt=VIDEO_NEGATIVE_PROMPT,
                height=h, width=w,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=torch.Generator(dev).manual_seed(int(seed)),
                output_type='latent',
                callback_on_step_end=_step_cb,
                callback_on_step_end_tensor_inputs=['latents'],
            )
        torch.cuda.synchronize(dev)
        _t_denoise_end = time.perf_counter()
        _t_cond_end = _timing.get('cond_end', _t_load)
        _tlog(tlog, ev_q, f'[Timing] Condition encoding : {_t_cond_end - _t_load:.3f}s  (total {_t_cond_end - _t0:.3f}s)')
        _tlog(tlog, ev_q, f'[Timing] Denoising          : {_t_denoise_end - _t_cond_end:.3f}s  (total {_t_denoise_end - _t0:.3f}s)')

        with torch.no_grad():
            latents = out.frames.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
            video_tensor = pipe.vae.decode(latents, return_dict=False)[0]
            output = pipe.video_processor.postprocess_video(video_tensor, output_type='np')
        torch.cuda.synchronize(dev)
        _t_vae = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] VAE decoding       : {_t_vae - _t_denoise_end:.3f}s  (total {_t_vae - _t0:.3f}s)')

        path = output_dir / f't2v_{uuid.uuid4().hex[:8]}.mp4'
        export_to_video(output[0], str(path), fps=16)
        _t_save = time.perf_counter()
        result['path'] = str(path)
        _tlog(tlog, ev_q, f'[Timing] Save video          : {_t_save - _t_vae:.3f}s  (total {_t_save - _t0:.3f}s)')
        if not MULTI_GPU:
            pipe.to('cpu')
            torch.cuda.empty_cache()
            _t_offload = time.perf_counter()
            _tlog(tlog, ev_q, f'[Timing] GPU unload          : {_t_offload - _t_save:.3f}s  (total {_t_offload - _t0:.3f}s)')

    yield from _stream_generation(_run)


def generate_i2v(image, prompt: str, num_frames: int, num_steps: int, guidance: float, seed: int, use_expander: str = 'No'):
    if image is None:
        yield None, '[Error] Please upload an input image.'
        return
    print(f'\n[Generate I2V] {num_frames} frames, {num_steps} steps, seed={int(seed)}')
    _t0 = time.perf_counter()

    def _run(tlog, ev_q, result):
        nonlocal prompt
        if use_expander == 'Yes':
            _tlog(tlog, ev_q, '[Info]   Extending prompt...')
            img_for_expand = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
            prompt = _expand_prompt(_get_expander_i2v(), prompt, image=img_for_expand)
            result['extended_prompt'] = prompt
            _tlog(tlog, ev_q, '[Info]   Prompt extended.')
        pipe = _get_i2v_pipe()

        # Compute resolution from image, snapped to model's patch grid
        img = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
        mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        aspect = img.height / img.width
        h = round(math.sqrt(720 * 1280 * aspect)) // mod * mod
        w = round(math.sqrt(720 * 1280 / aspect)) // mod * mod
        img = img.resize((w, h))

        dev = _DEV_I2V
        if not MULTI_GPU:
            pipe.to(dev)
        torch.cuda.synchronize(dev)
        _t_load = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] GPU load            : {_t_load - _t0:.3f}s  (total {_t_load - _t0:.3f}s)')
        _tlog(tlog, ev_q, f'[Info]   Generating {h}×{w}, {num_frames} frames ({num_steps} steps)...')

        _timing = {}
        def _step_cb(pipeline, i, t, cb_kwargs):
            if i == 0:
                torch.cuda.synchronize(dev)
                _timing['cond_end'] = time.perf_counter()
            return cb_kwargs

        with torch.no_grad():
            out = pipe(
                image=img,
                prompt=prompt.strip(),
                negative_prompt=VIDEO_NEGATIVE_PROMPT,
                height=h, width=w,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=torch.Generator(dev).manual_seed(int(seed)),
                output_type='latent',
                callback_on_step_end=_step_cb,
                callback_on_step_end_tensor_inputs=['latents'],
            )
        torch.cuda.synchronize(dev)
        _t_denoise_end = time.perf_counter()
        _t_cond_end = _timing.get('cond_end', _t_load)
        _tlog(tlog, ev_q, f'[Timing] Condition encoding : {_t_cond_end - _t_load:.3f}s  (total {_t_cond_end - _t0:.3f}s)')
        _tlog(tlog, ev_q, f'[Timing] Denoising          : {_t_denoise_end - _t_cond_end:.3f}s  (total {_t_denoise_end - _t0:.3f}s)')

        with torch.no_grad():
            latents = out.frames.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
            video_tensor = pipe.vae.decode(latents, return_dict=False)[0]
            output = pipe.video_processor.postprocess_video(video_tensor, output_type='np')
        torch.cuda.synchronize(dev)
        _t_vae = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] VAE decoding       : {_t_vae - _t_denoise_end:.3f}s  (total {_t_vae - _t0:.3f}s)')

        path = output_dir / f'i2v_{uuid.uuid4().hex[:8]}.mp4'
        export_to_video(output[0], str(path), fps=16)
        _t_save = time.perf_counter()
        result['path'] = str(path)
        _tlog(tlog, ev_q, f'[Timing] Save video          : {_t_save - _t_vae:.3f}s  (total {_t_save - _t0:.3f}s)')
        if not MULTI_GPU:
            pipe.to('cpu')
            torch.cuda.empty_cache()
            _t_offload = time.perf_counter()
            _tlog(tlog, ev_q, f'[Timing] GPU unload          : {_t_offload - _t_save:.3f}s  (total {_t_offload - _t0:.3f}s)')

    yield from _stream_generation(_run)


def generate_qwen_edit(image, prompt: str, num_steps: int, cfg_scale: float, seed: int, use_expander: str = 'No'):
    if image is None:
        yield None, '[Error] Please upload an input image.'
        return
    print(f'\n[Generate QwenEdit] {num_steps} steps, cfg={cfg_scale}, seed={int(seed)}')
    _t0 = time.perf_counter()

    def _run(tlog, ev_q, result):
        nonlocal prompt
        from PIL import Image as PILImage
        from pipeline_qwen_image_edit import calculate_dimensions
        img = PILImage.open(image).convert('RGB')
        if use_expander == 'Yes':
            _tlog(tlog, ev_q, '[Info]   Extending prompt...')
            prompt = _expand_prompt(_get_expander_edit(), prompt, image=img)
            result['extended_prompt'] = prompt
            _tlog(tlog, ev_q, '[Info]   Prompt extended.')
        pipe = _get_qwen_edit_pipe()

        # Pre-compute h/w so we can unpack latents after decoupled VAE decode
        _cw, _ch, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
        _mul = pipe.vae_scale_factor * 2
        _w = _cw // _mul * _mul
        _h = _ch // _mul * _mul

        dev = _DEV_EDIT
        if not MULTI_GPU:
            pipe.to(dev)
        torch.cuda.synchronize(dev)
        _t_load = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] GPU load            : {_t_load - _t0:.3f}s  (total {_t_load - _t0:.3f}s)')

        _timing = {}
        def _step_cb(pipeline, i, t, cb_kwargs):
            if i == 0:
                torch.cuda.synchronize(dev)
                _timing['cond_end'] = time.perf_counter()
            return cb_kwargs

        with torch.inference_mode():
            out = pipe(
                image=img,
                prompt=prompt.strip(),
                negative_prompt=' ',
                true_cfg_scale=cfg_scale,
                num_inference_steps=num_steps,
                height=_h, width=_w,
                generator=torch.Generator(dev).manual_seed(int(seed)),
                output_type='latent',
                callback_on_step_end=_step_cb,
                callback_on_step_end_tensor_inputs=['latents'],
            )
        torch.cuda.synchronize(dev)
        _t_denoise_end = time.perf_counter()
        _t_cond_end = _timing.get('cond_end', _t_load)
        _tlog(tlog, ev_q, f'[Timing] Condition encoding : {_t_cond_end - _t_load:.3f}s  (total {_t_cond_end - _t0:.3f}s)')
        _tlog(tlog, ev_q, f'[Timing] Denoising          : {_t_denoise_end - _t_cond_end:.3f}s  (total {_t_denoise_end - _t0:.3f}s)')

        with torch.inference_mode():
            latents = pipe._unpack_latents(out.images, _h, _w, pipe.vae_scale_factor)
            latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
            decoded = pipe.vae.decode(latents, return_dict=False)[0]
            out_img = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
        torch.cuda.synchronize(dev)
        _t_vae = time.perf_counter()
        _tlog(tlog, ev_q, f'[Timing] VAE decoding       : {_t_vae - _t_denoise_end:.3f}s  (total {_t_vae - _t0:.3f}s)')

        path = output_dir / f'qwen_edit_{uuid.uuid4().hex[:8]}.png'
        out_img.save(str(path))
        _t_save = time.perf_counter()
        result['path'] = str(path)
        _tlog(tlog, ev_q, f'[Timing] Storing             : {_t_save - _t_vae:.3f}s  (total {_t_save - _t0:.3f}s)')
        if not MULTI_GPU:
            pipe.to('cpu')
            torch.cuda.empty_cache()
            _t_offload = time.perf_counter()
            _tlog(tlog, ev_q, f'[Timing] GPU unload          : {_t_offload - _t_save:.3f}s  (total {_t_offload - _t0:.3f}s)')

    yield from _stream_generation(_run)


if MULTI_GPU:
    _preload_all_models()

with gr.Blocks(title='DC-Gen') as demo:
    gr.Markdown(INTRODUCTION)

    with gr.Tabs():
        with gr.Tab('DC-Gen-FLUX-1K'):
            gr.Markdown('### DC-Gen-FLUX - 1K Generation (DC-AE-f32c32)')
            with gr.Row():
                with gr.Column():
                    prompt_1k = gr.Textbox(label='Prompt', lines=4)
                    model_toggle = gr.Radio(
                        choices=['AnyFlow', 'Standard'],
                        value='Standard',
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
                    expand_1k = gr.Radio(['No', 'Yes'], value='No', label='Extend Prompt')
                    model_toggle.change(
                        lambda m: gr.update(value=4 if m == 'AnyFlow' else 20),
                        inputs=[model_toggle], outputs=[steps_1k],
                    )
                with gr.Column():
                    out_1k = gr.Image(label='Output', type='filepath')
                    timing_1k = gr.Textbox(label='Timing', lines=6, interactive=False)
                    extended_prompt_1k = gr.Textbox(label='Extended Prompt', lines=4, interactive=False, visible=True)
                    btn_1k = gr.Button('Generate', variant='primary')

            gr.Markdown('### Examples')
            for ex in EXAMPLES_1K:
                with gr.Row():
                    gr.Textbox(value=ex, show_label=False, interactive=False, lines=2)
                    use_btn = gr.Button('Use', scale=0, min_width=60)
                    use_btn.click(lambda p=ex: p, outputs=[prompt_1k])
            with gr.Row():
                gr.Textbox(value='一只猫慵懒地躺在一只狗的怀里晒太阳。', show_label=False, interactive=False, lines=2)
                use_btn_cn = gr.Button('Use', scale=0, min_width=60)
                use_btn_cn.click(
                    lambda: ('一只猫慵懒地躺在一只狗的怀里晒太阳。', 'Yes'),
                    outputs=[prompt_1k, expand_1k],
                )

            btn_1k.click(
                generate_1k,
                inputs=[prompt_1k, model_toggle, aspect_1k, steps_1k, guidance_1k, seed_1k, expand_1k],
                outputs=[out_1k, timing_1k, extended_prompt_1k],
            )

        with gr.Tab('DC-Gen-FLUX-4K'):
            gr.Markdown('### DC-Gen-FLUX - 4K Generation (DC-AE-1.5-f64c128)')
            with gr.Row():
                with gr.Column():
                    prompt_4k = gr.Textbox(label='Prompt', lines=4)
                    model_toggle_4k = gr.Radio(
                        choices=['AnyFlow', 'Standard'],
                        value='Standard',
                        label='Model',
                        info='AnyFlow: on-policy distilled model (faster, any-step); Standard: base 4K model',
                    )
                    aspect_ratio_4k = gr.Dropdown(
                        list(ASPECT_RATIOS_4K.keys()),
                        value='1:1  (4096×4096)',
                        label='Aspect Ratio',
                    )
                    with gr.Row():
                        steps_4k = gr.Slider(1, 50, value=20, step=1, label='Steps')
                        guidance_4k = gr.Slider(1.0, 10.0, value=3.5, step=0.1, label='Guidance Scale')
                    seed_4k = gr.Number(0, label='Seed', precision=0)
                    expand_4k = gr.Radio(['No', 'Yes'], value='No', label='Extend Prompt')
                    model_toggle_4k.change(
                        lambda m: gr.update(value=4 if m == 'AnyFlow' else 20),
                        inputs=[model_toggle_4k], outputs=[steps_4k],
                    )
                with gr.Column():
                    out_4k = gr.Image(label='Output', type='filepath')
                    timing_4k = gr.Textbox(label='Timing', lines=6, interactive=False)
                    extended_prompt_4k = gr.Textbox(label='Extended Prompt', lines=4, interactive=False, visible=True)
                    btn_4k = gr.Button('Generate', variant='primary')

            gr.Markdown('### Examples')
            for ex in EXAMPLES_4K:
                with gr.Row():
                    gr.Textbox(value=ex, show_label=False, interactive=False, lines=2)
                    use_btn = gr.Button('Use', scale=0, min_width=60)
                    use_btn.click(lambda p=ex: p, outputs=[prompt_4k])

            btn_4k.click(generate_4k, inputs=[prompt_4k, model_toggle_4k, aspect_ratio_4k, steps_4k, guidance_4k, seed_4k, expand_4k], outputs=[out_4k, timing_4k, extended_prompt_4k])

        with gr.Tab('DC-Gen-Wan2.1-T2V-14B-720P'):
            gr.Markdown('### DC-Gen-Wan2.1-14B - Text-to-Video (DC-AE-V-f32t4c32)')
            with gr.Row():
                with gr.Column():
                    prompt_t2v = gr.Textbox(label='Prompt', lines=4)
                    with gr.Row():
                        frames_t2v = gr.Slider(17, 121, value=81, step=8, label='Frames')
                        steps_t2v  = gr.Slider(1, 100, value=50, step=1, label='Steps')
                    with gr.Row():
                        guidance_t2v = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label='Guidance Scale')
                        seed_t2v     = gr.Number(42, label='Seed', precision=0)
                    expand_t2v = gr.Radio(['No', 'Yes'], value='No', label='Extend Prompt')
                with gr.Column():
                    out_t2v    = gr.Video(label='Output', autoplay=True)
                    timing_t2v = gr.Textbox(label='Timing', lines=6, interactive=False)
                    extended_prompt_t2v = gr.Textbox(label='Extended Prompt', lines=4, interactive=False, visible=True)
                    btn_t2v    = gr.Button('Generate', variant='primary')

            gr.Markdown('### Examples')
            for ex in EXAMPLES_T2V:
                with gr.Row():
                    gr.Textbox(value=ex, show_label=False, interactive=False, lines=2)
                    use_btn = gr.Button('Use', scale=0, min_width=60)
                    use_btn.click(lambda p=ex: p, outputs=[prompt_t2v])

            btn_t2v.click(
                generate_t2v,
                inputs=[prompt_t2v, frames_t2v, steps_t2v, guidance_t2v, seed_t2v, expand_t2v],
                outputs=[out_t2v, timing_t2v, extended_prompt_t2v],
            )

        with gr.Tab('DC-Gen-Wan2.1-I2V-14B-720P'):
            gr.Markdown('### DC-Gen-Wan2.1-14B - Image-to-Video (DC-AE-V-f32t4c32)')
            with gr.Row():
                with gr.Column():
                    image_i2v  = gr.Image(label='Input Image', type='filepath')
                    prompt_i2v = gr.Textbox(label='Prompt', lines=4)
                    with gr.Row():
                        frames_i2v = gr.Slider(17, 121, value=81, step=8, label='Frames')
                        steps_i2v  = gr.Slider(1, 100, value=50, step=1, label='Steps')
                    with gr.Row():
                        guidance_i2v = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label='Guidance Scale')
                        seed_i2v     = gr.Number(42, label='Seed', precision=0)
                    expand_i2v = gr.Radio(['No', 'Yes'], value='No', label='Extend Prompt')
                with gr.Column():
                    out_i2v    = gr.Video(label='Output', autoplay=True)
                    timing_i2v = gr.Textbox(label='Timing', lines=6, interactive=False)
                    extended_prompt_i2v = gr.Textbox(label='Extended Prompt', lines=4, interactive=False, visible=True)
                    btn_i2v    = gr.Button('Generate', variant='primary')

            gr.Markdown('### Examples')
            for img_path, prompt in EXAMPLES_I2V:
                with gr.Row():
                    gr.Image(value=img_path, show_label=False, interactive=False,
                             width=120, height=80, scale=0, min_width=120)
                    gr.Textbox(value=prompt, show_label=False, interactive=False, lines=3)
                    use_btn = gr.Button('Use', scale=0, min_width=60)
                    use_btn.click(lambda i=img_path, p=prompt: (i, p),
                                  outputs=[image_i2v, prompt_i2v])

            btn_i2v.click(
                generate_i2v,
                inputs=[image_i2v, prompt_i2v, frames_i2v, steps_i2v, guidance_i2v, seed_i2v, expand_i2v],
                outputs=[out_i2v, timing_i2v, extended_prompt_i2v],
            )

        with gr.Tab('DC-Gen-Qwen-Image-Edit-1K'):
            gr.Markdown('### DC-Gen-Qwen-Image-Edit - Instruction-based Image Editing (DC-AE-f32c32)')
            with gr.Row():
                with gr.Column():
                    image_qe   = gr.Image(label='Input Image', type='filepath')
                    prompt_qe  = gr.Textbox(label='Edit Instruction', lines=3,
                                            placeholder='e.g. Change the background to outside')
                    with gr.Row():
                        steps_qe   = gr.Slider(1, 100, value=50, step=1, label='Steps')
                        cfg_qe     = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label='CFG Scale')
                    seed_qe    = gr.Number(0, label='Seed', precision=0)
                    expand_qe  = gr.Radio(['No', 'Yes'], value='No', label='Extend Prompt')
                with gr.Column():
                    out_qe     = gr.Image(label='Output', type='filepath')
                    timing_qe  = gr.Textbox(label='Timing', lines=6, interactive=False)
                    extended_prompt_qe = gr.Textbox(label='Extended Prompt', lines=4, interactive=False, visible=True)
                    btn_qe     = gr.Button('Generate', variant='primary')

            gr.Markdown('### Examples')
            for img_path, instruction in EXAMPLES_QWEN_EDIT:
                with gr.Row():
                    gr.Image(value=img_path, show_label=False, interactive=False,
                             width=120, height=80, scale=0, min_width=120)
                    gr.Textbox(value=instruction, show_label=False, interactive=False, lines=2)
                    use_btn = gr.Button('Use', scale=0, min_width=60)
                    use_btn.click(lambda i=img_path, p=instruction: (i, p),
                                  outputs=[image_qe, prompt_qe])

            btn_qe.click(
                generate_qwen_edit,
                inputs=[image_qe, prompt_qe, steps_qe, cfg_qe, seed_qe, expand_qe],
                outputs=[out_qe, timing_qe, extended_prompt_qe],
            )

demo.queue(default_concurrency_limit=1)

if __name__ == '__main__':
    demo.launch(show_error=True, share=True, server_name='0.0.0.0', server_port=7861)
