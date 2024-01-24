import os
from functools import partial
from glob import glob
from pathlib import Path as PythonPath

import cv2
import torchvision.transforms.functional as TvF
import torch
import torch.nn as nn
import numpy as np
from inspect import isfunction
from PIL import Image

from src import smplfusion
from src.smplfusion import share, router, attentionpatch, transformerpatch
from src.utils.iimage import IImage
from src.utils import poisson_blend
from src.models.sd2_sr import predict_eps_from_z_and_v, predict_start_from_z_and_v


def refine_mask(hr_image, hr_mask, lr_image, sam_predictor):
    lr_mask = hr_mask.resize(512)

    x_min, y_min, rect_w, rect_h = cv2.boundingRect(lr_mask.data[0][:, :, 0])
    x_min = max(x_min - 1, 0)
    y_min = max(y_min - 1, 0)
    x_max = x_min + rect_w + 1
    y_max = y_min + rect_h + 1

    input_box = np.array([x_min, y_min, x_max, y_max])

    sam_predictor.set_image(hr_image.resize(512).data[0])
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )
    dilation_kernel = np.ones((13, 13))
    original_object_mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)
    original_object_mask = cv2.dilate(original_object_mask, dilation_kernel)

    sam_predictor.set_image(lr_image.resize(512).data[0])
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )
    dilation_kernel = np.ones((3, 3))
    inpainted_object_mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)
    inpainted_object_mask = cv2.dilate(inpainted_object_mask, dilation_kernel)

    lr_mask_masking = ((original_object_mask + inpainted_object_mask ) > 0).astype(np.uint8)
    new_mask = lr_mask.data[0] * lr_mask_masking[:, :, np.newaxis]
    new_mask = IImage(new_mask).resize(2048, resample = Image.BICUBIC)
    return new_mask


def run(
    ddim,
    sam_predictor,
    lr_image,
    hr_image,
    hr_mask,
    prompt = 'high resolution professional photo',
    noise_level=20,
    blend_output = True,
    blend_trick = True,
    dt = 50,
    seed = 1,
    guidance_scale = 7.5,
    negative_prompt = '',
    use_sam_mask = False
):
    hr_image_info = hr_image.info
    lr_image = IImage(lr_image)
    hr_image = IImage(hr_image).resize(2048)
    hr_mask = IImage(hr_mask).resize(2048)

    torch.manual_seed(seed)
    dtype = ddim.vae.encoder.conv_in.weight.dtype
    device = ddim.vae.encoder.conv_in.weight.device

    router.attention_forward = attentionpatch.default.forward_xformers
    router.basic_transformer_forward = transformerpatch.default.forward

    hr_image_orig = hr_image
    hr_mask_orig = hr_mask

    if use_sam_mask:
        with torch.no_grad():
            hr_mask = refine_mask(hr_image, hr_mask, lr_image, sam_predictor)

    orig_h, orig_w = hr_image.torch().shape[2], hr_image.torch().shape[3]
    hr_image = hr_image.padx(256, padding_mode='reflect')
    hr_mask = hr_mask.padx(256, padding_mode='reflect').dilate(19)
   
    lr_image = lr_image.padx(64, padding_mode='reflect').torch()
    lr_mask = hr_mask.resize((lr_image.shape[2:]), resample = Image.BICUBIC)
    lr_mask = lr_mask.alpha().torch(vmin=0).to(device)
    lr_mask = TvF.gaussian_blur(lr_mask, kernel_size=19)
                                     
    # encode hr image
    with torch.no_grad():
        hr_image = hr_image.torch().to(dtype=dtype, device=device)
        hr_z0 = ddim.vae.encode(hr_image).mean * ddim.config.scale_factor

    assert hr_z0.shape[2] == lr_image.shape[2]
    assert hr_z0.shape[3] == lr_image.shape[3]                                  
                                     
    with torch.no_grad():
        context = ddim.encoder.encode([negative_prompt, prompt])
    
    noise_level = torch.Tensor(1 * [noise_level]).to(device=device).long()
    unet_condition = lr_image.to(dtype=dtype, device=device, memory_format=torch.contiguous_format)
    unet_condition, noise_level = ddim.low_scale_model(unet_condition, noise_level=noise_level)

    with torch.autocast('cuda'), torch.no_grad():
        zt = torch.randn((1,4,unet_condition.shape[2], unet_condition.shape[3]))
        zt = zt.cuda().to(dtype=dtype, device=device)
        for index,t in enumerate(range(999, 0, -dt)):
            _zt = zt if unet_condition is None else torch.cat([zt, unet_condition], 1)
            eps_uncond, eps = ddim.unet(
                torch.cat([_zt, _zt]).to(dtype=dtype, device=device), 
                timesteps = torch.tensor([t, t]).to(device=device), 
                context = context,
                y=torch.cat([noise_level]*2)
            ).chunk(2)
            ts = torch.full((zt.shape[0],), t, device=device, dtype=torch.long)
            model_output = (eps_uncond + guidance_scale * (eps - eps_uncond))
            eps = predict_eps_from_z_and_v(ddim.schedule, zt, ts, model_output).to(dtype)
            z0 = predict_start_from_z_and_v(ddim.schedule, zt, ts, model_output).to(dtype)
            if blend_trick:
                z0 = z0 * lr_mask + hr_z0 * (1-lr_mask)
            zt = ddim.schedule.sqrt_alphas[t - dt] * z0 + ddim.schedule.sqrt_one_minus_alphas[t - dt] * eps

    with torch.no_grad():
        hr_result = ddim.vae.decode(z0.to(dtype) / ddim.config.scale_factor)
        # postprocess
        hr_result = (255 * ((hr_result + 1) / 2).clip(0, 1)).to(torch.uint8)
        hr_result = hr_result.cpu().permute(0, 2, 3, 1)[0].numpy()
        hr_result = hr_result[:orig_h, :orig_w, :]

    if blend_output:
        hr_result = poisson_blend(
            orig_img=hr_image_orig.data[0],
            fake_img=hr_result,
            mask=hr_mask_orig.alpha().data[0]
        )
    hr_result = Image.fromarray(hr_result)
    hr_result.info = hr_image_info  # save metadata
    return hr_result
