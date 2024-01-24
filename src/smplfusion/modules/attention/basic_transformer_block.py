import torch
from torch import nn
from .feed_forward import FeedForward

try:
    from .cross_attention import PatchedCrossAttention as CrossAttention
except:
    try:
        from .memory_efficient_cross_attention import MemoryEfficientCrossAttention as CrossAttention
    except:
        from .cross_attention import CrossAttention
from ..util import checkpoint
from ...patches import router

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,dim,n_heads,d_head,dropout=0.0,context_dim=None,
        gated_ff=True,checkpoint=True,disable_self_attn=False,
    ):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        # is a self-attention if not self.disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim,heads=n_heads,dim_head=d_head,dropout=dropout,context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # is self-attn if context is none
        self.attn2 = CrossAttention(query_dim=dim,context_dim=context_dim,heads=n_heads,dim_head=d_head,dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = x + self.attn1(self.norm1(x), context=context if self.disable_self_attn else None)
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x

class PatchedBasicTransformerBlock(nn.Module):
    def __init__(
        self,dim,n_heads,d_head,dropout=0.0,context_dim=None,
        gated_ff=True,checkpoint=True,disable_self_attn=False,
    ):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        # is a self-attention if not self.disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim,heads=n_heads,dim_head=d_head,dropout=dropout,context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # is self-attn if context is none
        self.attn2 = CrossAttention(query_dim=dim,context_dim=context_dim,heads=n_heads,dim_head=d_head,dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        return router.basic_transformer_forward(self, x, context)
