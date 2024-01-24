import torchvision.transforms.functional as TF
from src.utils.iimage import IImage
import torch 
import sys
from .utils import *

input_mask = None
input_shape = None
timestep = None
timestep_index = None

class Seed:
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = list(range(*idx.indices(idx.stop)))
        if isinstance(idx, list) or isinstance(idx, tuple):
            return [self[_idx] for _idx in idx]
        return 12345 ** idx % 54321

class DDIMIterator:
    def __init__(self, iterator):
        self.iterator = iterator
    def __iter__(self):
        self.iterator = iter(self.iterator)
        global timestep_index
        timestep_index = 0
        return self
    def __next__(self):
        global timestep, timestep_index
        timestep = next(self.iterator)
        timestep_index += 1
        return timestep
seed = Seed()
self = sys.modules[__name__]
    
def reshape(x):
    return input_shape.reshape(x)

def set_shape(image_or_shape):
    global input_shape
    # if isinstance(image_or_shape, IImage):
    if hasattr(image_or_shape, 'size'):
        input_shape = InputShape(image_or_shape.size)
    if isinstance(image_or_shape, torch.Tensor):
        input_shape = InputShape(image_or_shape.shape[-2:][::-1])
    elif isinstance(image_or_shape, list) or isinstance(image_or_shape, tuple):
        input_shape = InputShape(image_or_shape)

def set_mask(mask):
    global input_mask, mask64, mask32, mask16, mask8, painta_mask
    input_mask = InputMask(mask)
    painta_mask = InputMask(mask)

    mask64 = input_mask.val64[0,0]
    mask32 = input_mask.val32[0,0]
    mask16 = input_mask.val16[0,0]
    mask8  = input_mask.val8[0,0]
    set_shape(mask)

def exists(name):
    return hasattr(self, name) and getattr(self, name) is not None