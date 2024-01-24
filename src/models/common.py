import importlib
import requests
from pathlib import Path
from os.path import dirname

import torch
import safetensors
import safetensors.torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.smplfusion import DDIM, share, scheduler


PROJECT_DIR = dirname(dirname(dirname(__file__)))
CONFIG_FOLDER =  f'{PROJECT_DIR}/config'
MODEL_FOLDER =  f'{PROJECT_DIR}/checkpoints'


def download_file(url, save_path, chunk_size=1024):
    try:
        save_path = Path(save_path)
        if save_path.exists():
            print(f'{save_path.name} exists')
            return
        save_path.parent.mkdir(exist_ok=True, parents=True)
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path.name,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
        print(f'{save_path.name} download finished')
    except Exception as e:
        raise Exception(f"Download failed: {e}")


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except:
        return getattr(importlib.import_module('src.' + module, package=None), cls)


def load_obj(path):
    objyaml = OmegaConf.load(path)
    return get_obj_from_str(objyaml['__class__'])(**objyaml.get("__init__", {}))


def load_sd_inpainting_model(
    download_url,
    model_path,
    sd_version,
    dtype=torch.float16,
    device='cuda:0'
):
    model_path = f'{MODEL_FOLDER}/{model_path}'

    download_file(download_url, model_path)
    
    model_ext = Path(model_path).suffix
    
    if model_ext == '.safetensors':
        state_dict = safetensors.torch.load_file(model_path)
    elif model_ext == '.ckpt':
        state_dict = torch.load(model_path)['state_dict']
    else:
        raise Exception(f'Unsupported model extension {model_ext}')

    # Load common config files
    config = OmegaConf.load(f'{CONFIG_FOLDER}/ddpm/v1.yaml')
    vae = load_obj(f'{CONFIG_FOLDER}/vae.yaml').eval().cuda()

    # Load version specific config files
    if sd_version == 1:
        encoder = load_obj(f'{CONFIG_FOLDER}/encoders/clip.yaml').eval().cuda()
        unet = load_obj(f'{CONFIG_FOLDER}/unet/inpainting/v1.yaml').eval().cuda()
    elif sd_version == 2:
        encoder = load_obj(f'{CONFIG_FOLDER}/encoders/openclip.yaml').eval().cuda()
        unet = load_obj(f'{CONFIG_FOLDER}/unet/inpainting/v2.yaml').eval().cuda()
    else:
        raise Exception(f'Unsupported SD version {sd_version}.')
    
    ddim = DDIM(config, vae, encoder, unet)

    extract = lambda state_dict, model: {x[len(model)+1:]:y for x,y in state_dict.items() if model in x}
    unet_state = extract(state_dict, 'model.diffusion_model')
    encoder_state = extract(state_dict, 'cond_stage_model')
    vae_state = extract(state_dict, 'first_stage_model')

    unet.load_state_dict(unet_state)
    encoder.load_state_dict(encoder_state)
    vae.load_state_dict(vae_state)

    if dtype == torch.float16:
        unet.convert_to_fp16()
    unet.to(device=device)
    vae.to(dtype=dtype, device=device)
    encoder.to(dtype=dtype, device=device)
    encoder.device = device

    unet = unet.requires_grad_(False)
    encoder = encoder.requires_grad_(False)
    vae = vae.requires_grad_(False)

    ddim = DDIM(config, vae, encoder, unet)
    share.schedule = scheduler.linear(config.timesteps, config.linear_start, config.linear_end)

    return ddim
