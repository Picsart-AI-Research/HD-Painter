from collections import OrderedDict

import torch
from .common import MODEL_FOLDER, load_sd_inpainting_model, download_file

model_dict = {
    'sd15_inp': {
        'sd_version': 1,
        'diffusers_ckpt': True,
        'model_path': OrderedDict([
            ('unet', 'sd-1-5-inpainting/unet.fp16.safetensors'),
            ('encoder', 'sd-1-5-inpainting/encoder.fp16.safetensors'),
            ('vae', 'sd-1-5-inpainting/vae.fp16.safetensors')
        ]),
        'download_url': OrderedDict([
            ('unet', 'https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors?download=true'),
            ('encoder', 'https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/text_encoder/model.fp16.safetensors?download=true'),
            ('vae', 'https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors?download=true')
        ])
    },
    'ds8_inp': {
        'sd_version': 1,
        'diffusers_ckpt': True,
        'model_path': OrderedDict([
            ('unet', 'ds-8-inpainting/unet.fp16.safetensors'),
            ('encoder', 'ds-8-inpainting/encoder.fp16.safetensors'),
            ('vae', 'ds-8-inpainting/vae.fp16.safetensors')
        ]),
        'download_url': OrderedDict([
            ('unet', 'https://huggingface.co/Lykon/dreamshaper-8-inpainting/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors?download=true'),
            ('encoder', 'https://huggingface.co/Lykon/dreamshaper-8-inpainting/resolve/main/text_encoder/model.fp16.safetensors?download=true'),
            ('vae', 'https://huggingface.co/Lykon/dreamshaper-8-inpainting/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors?download=true')
        ])
    },
    'sd2_inp': {
        'sd_version': 2,
        'diffusers_ckpt': False,
        'model_path': 'sd-2-0-inpainting/512-inpainting-ema.safetensors',
        'download_url': 'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors?download=true'
    }
}

model_cache = {}


def pre_download_inpainting_models():
    for model_id, model_details in model_dict.items():
        download_url = model_details['download_url']
        model_path = model_details["model_path"]

        if type(download_url) == str and type(model_path) == str:
            download_file(download_url, f'{MODEL_FOLDER}/{model_path}')
        elif type(download_url) == OrderedDict and type(model_path) == OrderedDict:
            for key in download_url.keys():
                download_file(download_url[key], f'{MODEL_FOLDER}/{model_path[key]}')
        else:
            raise Exception('download_url definition type is not supported')


def load_inpainting_model(model_id, dtype=torch.float16, device='cuda:0', cache=False):
    if cache and model_id in model_cache:
        return model_cache[model_id]
    else:
        if model_id not in model_dict:
            raise Exception(f'Unsupported model-id. Choose one from {list(model_dict.keys())}.')

        model = load_sd_inpainting_model(
            **model_dict[model_id],
            dtype=dtype,
            device=device
        )
        if cache:
            model_cache[model_id] = model
        return model
