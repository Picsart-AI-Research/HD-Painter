import importlib
from functools import partial

import cv2
import numpy as np
import safetensors
import safetensors.torch
import torch
import torch.nn as nn
from inspect import isfunction
from omegaconf import OmegaConf

from src.smplfusion import DDIM, share, scheduler
from .common import *


DOWNLOAD_URL = 'https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.safetensors?download=true'
MODEL_PATH = f'{MODEL_FOLDER}/sd-2-0-upsample/x4-upscaler-ema.safetensors'

# pre-download
# download_file(DOWNLOAD_URL, MODEL_PATH)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predict_eps_from_z_and_v(schedule, x_t, t, v):
    return (
        extract_into_tensor(schedule.sqrt_alphas.to(x_t.device), t, x_t.shape) * v +
        extract_into_tensor(schedule.sqrt_one_minus_alphas.to(x_t.device), t, x_t.shape) * x_t
    )


def predict_start_from_z_and_v(schedule, x_t, t, v):
    return (
        extract_into_tensor(schedule.sqrt_alphas.to(x_t.device), t, x_t.shape) * x_t -
        extract_into_tensor(schedule.sqrt_one_minus_alphas.to(x_t.device), t, x_t.shape) * v
    )


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class AbstractLowScaleModel(nn.Module):
    # for concatenating a downsampled image to the latent representation
    def __init__(self, noise_schedule_config=None):
        super(AbstractLowScaleModel, self).__init__()
        if noise_schedule_config is not None:
            self.register_schedule(**noise_schedule_config)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x):
        return x, None

    def decode(self, x):
        return x


class ImageConcatWithNoiseAugmentation(AbstractLowScaleModel):
    def __init__(self, noise_schedule_config, max_noise_level=1000, to_cuda=False):
        super().__init__(noise_schedule_config=noise_schedule_config)
        self.max_noise_level = max_noise_level

    def forward(self, x, noise_level=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        z = self.q_sample(x, noise_level)
        return z, noise_level


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except:
        return getattr(importlib.import_module('src.' + module, package=None), cls)
def load_obj(path):
    objyaml = OmegaConf.load(path)
    return get_obj_from_str(objyaml['__class__'])(**objyaml.get("__init__", {}))
    

def load_model(dtype=torch.bfloat16, device='cuda:0'):
    download_file(DOWNLOAD_URL, MODEL_PATH)

    state_dict = safetensors.torch.load_file(MODEL_PATH)

    config = OmegaConf.load(f'{CONFIG_FOLDER}/ddpm/v2-upsample.yaml')

    unet = load_obj(f'{CONFIG_FOLDER}/unet/upsample/v2.yaml').eval().cuda()
    vae = load_obj(f'{CONFIG_FOLDER}/vae-upsample.yaml').eval().cuda()
    encoder = load_obj(f'{CONFIG_FOLDER}/encoders/openclip.yaml').eval().cuda()
    ddim = DDIM(config, vae, encoder, unet)

    extract = lambda state_dict, model: {x[len(model)+1:]:y for x,y in state_dict.items() if model in x}
    unet_state = extract(state_dict, 'model.diffusion_model')
    encoder_state = extract(state_dict, 'cond_stage_model')
    vae_state = extract(state_dict, 'first_stage_model')

    unet.load_state_dict(unet_state)
    encoder.load_state_dict(encoder_state)
    vae.load_state_dict(vae_state)

    unet = unet.requires_grad_(False)
    encoder = encoder.requires_grad_(False)
    vae = vae.requires_grad_(False)
    
    unet.to(dtype=dtype, device=device)
    vae.to(dtype=dtype, device=device)
    encoder.to(dtype=dtype, device=device)
    encoder.device = device

    ddim = DDIM(config, vae, encoder, unet)

    params = {
        'noise_schedule_config': {
            'linear_start': 0.0001,
            'linear_end': 0.02
        },
        'max_noise_level': 350
    }

    low_scale_model = ImageConcatWithNoiseAugmentation(**params).eval().to('cuda')
    low_scale_model.train = disabled_train
    for param in low_scale_model.parameters():
        param.requires_grad = False

    low_scale_model = low_scale_model.to(dtype=dtype, device=device)

    ddim.low_scale_model = low_scale_model
    return ddim
