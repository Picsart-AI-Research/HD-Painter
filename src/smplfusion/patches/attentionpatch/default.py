from ... import share

import xformers
import xformers.ops


import torch
from torch import nn, einsum
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
 
_ATTN_PRECISION = None

def forward_sd2(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        if _ATTN_PRECISION =="fp32": # force cast to fp32 to avoid overflowing 
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k
    
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

def forward_xformers(self, x, context=None, mask=None):
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    b, _, _ = q.shape
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, t.shape[1], self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * self.heads, t.shape[1], self.dim_head)
        .contiguous(),
        (q, k, v),
    )

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

    if mask is not None:
        raise NotImplementedError
    out = (
        out.unsqueeze(0)
        .reshape(b, self.heads, out.shape[1], self.dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, out.shape[1], self.heads * self.dim_head)
    )
    return self.to_out(out)

forward = forward_xformers

import traceback

def forward_and_save(self, x, context=None, mask=None):
    att_type = "self" if context is None else "cross"

    h = self.heads
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

    sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
    
    if hasattr(share, '_crossattn_similarity_res8') and x.shape[1] == share.input_shape.res8 and att_type == 'cross':
        share._crossattn_similarity_res8.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res16') and x.shape[1] == share.input_shape.res16 and att_type == 'cross':
        share._crossattn_similarity_res16.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res32') and x.shape[1] == share.input_shape.res32 and att_type == 'cross':
        share._crossattn_similarity_res32.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts
    if hasattr(share, '_crossattn_similarity_res64') and x.shape[1] == share.input_shape.res64 and att_type == 'cross':
        share._crossattn_similarity_res64.append(torch.stack(share.reshape(sim).chunk(2))) # Chunk into 2 parts to differentiate the unconditional and conditional parts

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", sim, v)
    out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
    return self.to_out(out)