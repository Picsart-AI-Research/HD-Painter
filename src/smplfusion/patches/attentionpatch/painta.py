import cv2
import math
import numbers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, einsum
from einops import rearrange, repeat

from ... import share
from src.utils.iimage import IImage

# params
painta_res = [16, 32]
painta_on = True
token_idx = [1,2]


# GaussianSmoothing is taken from https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/gaussian_smoothing.py
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups, padding='same')


def forward(self, x, context=None, mask=None):
    is_cross = context is not None
    att_type = "self" if context is None else "cross"

    h = self.heads

    q = self.to_q(x)
    context =  x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    sim_before = sim
    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    if hasattr(share, '_crossattn_similarity_res8') and x.shape[1] == share.input_shape.res8 and att_type == 'cross':
        share._crossattn_similarity_res8.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res16') and x.shape[1] == share.input_shape.res16 and att_type == 'cross':
        share._crossattn_similarity_res16.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res32') and x.shape[1] == share.input_shape.res32 and att_type == 'cross':
        share._crossattn_similarity_res32.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res64') and x.shape[1] == share.input_shape.res64 and att_type == 'cross':
        share._crossattn_similarity_res64.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts

    sim = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    
    if is_cross:
        return self.to_out(out)
    
    return self.to_out(out), v, sim_before


def painta_rescale(y, self_v, self_sim, cross_sim, self_h, to_out):   
    mask = share.painta_mask.get_res(self_v)
    shape = share.painta_mask.get_shape(self_v)
    res = share.painta_mask.get_res_val(self_v)
    
    mask = (mask > 0.5).to(y.dtype)
    m = mask.to(self_v.device)
    m = rearrange(m, 'b c h w -> b (h w) c').contiguous()
    m = torch.matmul(m, m.permute(0, 2, 1)) + (1-m) 
    
    cross_sim = cross_sim[:, token_idx].sum(dim=1)
    cross_sim = cross_sim.reshape(shape)
    gaussian_smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
    cross_sim = gaussian_smoothing(cross_sim.unsqueeze(0))[0]  # optional smoothing
    cross_sim = cross_sim.reshape(-1)
    cross_sim = ((cross_sim - torch.median(cross_sim.ravel())) / torch.max(cross_sim.ravel())).clip(0, 1)
  
    if painta_on and res in painta_res:
        c = (1 - m) * cross_sim.reshape(1, 1, -1) + m
        self_sim = self_sim * c
        self_sim = self_sim.softmax(dim=-1)        
        out = einsum('b i j, b j d -> b i d', self_sim, self_v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self_h)
        out = to_out(out)
    else: 
        out = y
    return out
    
