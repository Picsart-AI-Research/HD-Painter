import torch 
from ... import share

def forward(self, x, context=None):
    x = x + self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) # Self Attn.
    x = x + self.attn2(self.norm2(x), context=context) # Cross Attn.
    x = x + self.ff(self.norm3(x))
    return x
