"""
Minimal debug: load exactly as load_pipeline_anyflow, generate ONE 4-step image.
Run with: CUDA_VISIBLE_DEVICES=X python debug_test_minimal.py
"""
import json, pathlib, shutil, sys, torch
repo_root = pathlib.Path(__file__).resolve().parent
anyflow_src = repo_root.parent / 'anyflow-gradio-demo' / 'AnyFlow'
sys.path.insert(0, str(anyflow_src))

custom_code_dir = repo_root / 'custom_code'
local_dir = repo_root / 'pretrained_models' / 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K-Anyflow'

shutil.copy(custom_code_dir / 'transformer_dcgen_flux_model.py',
            anyflow_src / 'far' / 'models' / 'transformer_dcgen_flux_model.py')
shutil.copy(custom_code_dir / 'scheduling_flowmap_euler_discrete.py',
            anyflow_src / 'far' / 'schedulers' / 'scheduling_flowmap_euler_discrete.py')
shutil.copy(custom_code_dir / 'pipeline_dcgen_flux_anyflow.py',
            anyflow_src / 'far' / 'pipelines' / 'pipeline_dcgen_flux_anyflow.py')
for mod in list(sys.modules.keys()):
    if mod.startswith('far.'):
        del sys.modules[mod]

from far.pipelines.pipeline_dcgen_flux_anyflow import DCGenFluxAnyFlowPipeline
from far.models.transformer_dcgen_flux_model import DCGenFluxFlowMapModel
from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
from diffusers import AutoencoderDC
from transformers import CLIPTextModel, PreTrainedTokenizerFast, T5EncoderModel
from safetensors.torch import load_file as _load_sf

dtype = torch.bfloat16
vae = AutoencoderDC.from_pretrained(local_dir / 'vae', torch_dtype=dtype)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(local_dir / 'tokenizer' / 'tokenizer.json'),
    model_max_length=77, pad_token='<|endoftext|>', eos_token='<|endoftext|>',
    bos_token='<|startoftext|>', unk_token='<|endoftext|>',
)
tokenizer_2 = PreTrainedTokenizerFast(
    tokenizer_file=str(local_dir / 'tokenizer_2' / 'tokenizer.json'),
    model_max_length=512, eos_token='</s>', unk_token='<unk>', pad_token='<pad>',
)
text_encoder = CLIPTextModel.from_pretrained(local_dir / 'text_encoder', torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(local_dir / 'text_encoder_2', torch_dtype=dtype)

_t_dir = local_dir / 'transformer'
transformer = DCGenFluxFlowMapModel.from_config(DCGenFluxFlowMapModel.load_config(str(_t_dir)))
transformer.setup_flowmap_model(gate_value=0.25, deltatime_type='r')
with open(_t_dir / 'diffusion_pytorch_model.safetensors.index.json') as f:
    _shards = sorted(set(json.load(f)['weight_map'].values()))
_sd = {}
for s in _shards:
    _sd.update(_load_sf(str(_t_dir / s)))
missing, _ = transformer.load_state_dict(_sd, strict=False)
print(f'missing keys: {len(missing)}')
transformer = transformer.to(dtype).eval()

scheduler = FlowMapDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
nd = torch.load(local_dir / 'flux_null_embedding.pth', map_location='cpu', weights_only=False)
print('null_embeds keys:', list(nd.keys()))
null_pe  = nd.get('prompt_embeds',       nd.get('null_prompt_embeds')).to(dtype)
null_ppe = nd.get('pooled_prompt_embeds', nd.get('null_pooled_prompt_embeds')).to(dtype)

pipe = DCGenFluxAnyFlowPipeline(
    vae=vae, tokenizer=tokenizer, tokenizer_2=tokenizer_2,
    text_encoder=text_encoder, text_encoder_2=text_encoder_2,
    transformer=transformer, scheduler=scheduler,
    null_prompt_embeds=null_pe, null_pooled_prompt_embeds=null_ppe,
)
pipe.to('cuda')
prompt = "A serene landscape where rolling green mountains cradle a tranquil lake."
with torch.no_grad():
    img = pipe(prompt, height=1024, width=1024, num_inference_steps=4,
               guidance_scale=3.5, generator=torch.Generator('cuda').manual_seed(0)).images[0]
out = local_dir / 'debug_minimal_4steps.jpg'
img.save(str(out))
print('saved', out)
