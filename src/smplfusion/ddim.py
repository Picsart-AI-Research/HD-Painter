import torch
from tqdm.notebook import tqdm 
from . import scheduler
from . import share

from src.utils.iimage import IImage

class DDIM:
    def __init__(self, config, vae, encoder, unet):
        self.vae = vae
        self.encoder = encoder
        self.unet = unet
        self.config = config
        self.schedule = scheduler.linear(1000, config.linear_start, config.linear_end)
    
    def __call__(
            self, prompt = '', dt = 50, shape = (1,4,64,64), seed = None, negative_prompt = '', unet_condition = None, 
            context = None, verbose = True):
        if seed is not None: torch.manual_seed(seed)
        if unet_condition is not None:
            zT = torch.randn((1,4) + unet_condition.shape[2:]).cuda()
        else:
            zT = torch.randn(shape).cuda()

        with torch.autocast('cuda'), torch.no_grad():
            if context is None: context = self.encoder.encode([negative_prompt, prompt])

            zt = zT
            pbar = tqdm(range(999, 0, -dt)) if verbose else range(999, 0, -dt)
            for timestep in share.DDIMIterator(pbar):
                _zt = zt if unet_condition is None else torch.cat([zt, unet_condition], 1)
                eps_uncond, eps = self.unet(
                    torch.cat([_zt, _zt]), 
                    timesteps = torch.tensor([timestep, timestep]).cuda(), 
                    context = context
                ).chunk(2)
                
                eps = (eps_uncond + 7.5 * (eps - eps_uncond))
                
                z0 = (zt - self.schedule.sqrt_one_minus_alphas[timestep] * eps) / self.schedule.sqrt_alphas[timestep]
                zt = self.schedule.sqrt_alphas[timestep - dt] * z0 + self.schedule.sqrt_one_minus_alphas[timestep - dt] * eps
        return IImage(self.vae.decode(z0 / self.config.scale_factor))

    def get_inpainting_condition(self, image, mask):
        latent_size = [x//8 for x in image.size]
        dtype = self.vae.encoder.conv_in.weight.dtype
        with torch.no_grad():
            masked_image = image.torch().cuda() * ~mask.torch(0).bool().cuda()
            masked_image = masked_image.to(dtype)
            condition_x0 = self.vae.encode(masked_image).mean * self.config.scale_factor
        condition_mask = mask.resize(latent_size[::-1]).cuda().torch(0).bool().to(dtype)
        return torch.cat([condition_mask, condition_x0], 1)

    inpainting_condition = get_inpainting_condition

