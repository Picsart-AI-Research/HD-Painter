import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from src.utils.iimage import IImage
from src.smplfusion import share, router, attentionpatch, transformerpatch
from src.smplfusion.patches.attentionpatch import painta
from src.utils import tokenize

verbose = False


def init_painta(token_idx):
    # Initialize painta
    router.attention_forward = attentionpatch.painta.forward
    router.basic_transformer_forward = transformerpatch.painta.forward
    painta.painta_on = True
    painta.painta_res = [16, 32]
    painta.token_idx = token_idx

def run(
    ddim,
    method,
    prompt,
    image,
    mask,
    seed=0,
    eta=0.0,
    negative_prompt='',
    positive_prompt='',
    num_steps=50,
    guidance_scale=7.5
):
    image = image.padx(64)
    mask = mask.dilate(1).alpha().padx(64)
    full_prompt = prompt
    if positive_prompt != '':
        full_prompt = f'{prompt}, {positive_prompt}'
    dt = 1000 // num_steps

    # Text condition
    context = ddim.encoder.encode([negative_prompt, full_prompt])
    token_idx = list(range(1, tokenize(prompt).index('<end_of_text>')))
    token_idx += [tokenize(full_prompt).index('<end_of_text>')]

    # Setup painta if needed
    if 'painta' in method: init_painta(token_idx)
    else: router.reset() 
    
    # Image condition    
    unet_condition = ddim.get_inpainting_condition(image, mask)
    dtype = unet_condition.dtype
    share.set_mask(mask)
    
    # Starting latent
    seed_everything(seed)
    zt = torch.randn((1,4) + unet_condition.shape[2:]).cuda().to(dtype)

    # Turn off gradients
    ddim.unet.requires_grad_(False)
    
    pbar = tqdm(range(999, 0, -dt)) if verbose else range(999, 0, -dt)

    for timestep in share.DDIMIterator(pbar):
        if share.timestep <= 500: router.reset()

        _zt = zt if unet_condition is None else torch.cat([zt, unet_condition], 1)
        with torch.autocast('cuda'):
            eps_uncond, eps = ddim.unet(
                torch.cat([_zt, _zt]).to(dtype), 
                timesteps = torch.tensor([timestep, timestep]).cuda(), 
                context = context
            ).chunk(2)

        eps = (eps_uncond + guidance_scale * (eps - eps_uncond))
        z0 = (zt - share.schedule.sqrt_one_minus_alphas[timestep] * eps) / share.schedule.sqrt_alphas[timestep]
        zt = share.schedule.sqrt_alphas[timestep - dt] * z0 + share.schedule.sqrt_one_minus_alphas[timestep - dt] * eps

    with torch.no_grad():
        output_image = IImage(ddim.vae.decode(z0 / ddim.config.scale_factor))
    
    return output_image
