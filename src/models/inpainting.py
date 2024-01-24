import torch
from .common import MODEL_FOLDER, load_sd_inpainting_model, download_file

model_dict = {
    'sd15_inp': {
        'sd_version': 1,
        'model_path': 'sd-1-5-inpainting/sd-v1-5-inpainting.ckpt',
        'download_url': 'https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt?download=true'
    },
    'ds8_inp': {
        'sd_version': 1,
        'model_path': 'dreamshaper/dreamshaper_8Inpainting.safetensors',
        'download_url': 'https://civitai.com/api/download/models/131004'
    },
    'sd2_inp': {
        'sd_version': 2,
        'model_path': 'sd-2-0-inpainting/512-inpainting-ema.safetensors',
        'download_url': 'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors?download=true'
    }
}

model_cache = {}


def pre_download_inpainting_models():
    for model_id, model_details in model_dict.items():
        download_file(
            model_details['download_url'],
            f'{MODEL_FOLDER}/{model_details["model_path"]}'
        )


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
