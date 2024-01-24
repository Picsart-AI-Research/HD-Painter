import io
import math
import os
import warnings

import PIL.Image
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tvF
from scipy.ndimage import binary_dilation


def stack(images, axis = 0):
    return IImage(np.concatenate([x.data for x in images], axis))


def torch2np(x, vmin=-1, vmax=1):
    if x.ndim != 4:
        # raise Exception("Please only use (B,C,H,W) torch tensors!")
        warnings.warn(
            "Warning! Shape of the image was not provided in (B,C,H,W) format, the shape was inferred automatically!")
        if x.ndim == 3:
            x = x[None]
        if x.ndim == 2:
            x = x[None, None]
    x = x.detach().cpu().float()
    if x.dtype == torch.uint8:
        return x.numpy().astype(np.uint8)
    elif vmin is not None and vmax is not None:
        x = (255 * (x.clip(vmin, vmax) - vmin) / (vmax - vmin))
        x = x.permute(0, 2, 3, 1).to(torch.uint8)
        return x.numpy()
    else:
        raise NotImplementedError()


class IImage:
    @staticmethod
    def open(path):
        data = np.array(PIL.Image.open(path))
        if data.ndim == 3:
            data = data[..., None]
        image = IImage(data)
        return image

    @staticmethod
    def normalized(x, dims=[-1, -2]):
        x = (x - x.amin(dims, True)) / \
            (x.amax(dims, True) - x.amin(dims, True))
        return IImage(x, 0)

    def numpy(self): return self.data

    def torch(self, vmin=-1, vmax=1):
        if self.data.ndim == 3:
            data = self.data.transpose(2, 0, 1) / 255.
        else:
            data = self.data.transpose(0, 3, 1, 2) / 255.
        return vmin + torch.from_numpy(data).float().to(self.device) * (vmax - vmin)

    def to(self, device):
        self.device = device
        return self

    def cuda(self):
        self.device = 'cuda'
        return self

    def cpu(self):
        self.device = 'cpu'
        return self

    def pil(self):
        ans = []
        for x in self.data:
            if x.shape[-1] == 1:
                x = x[..., 0]

            ans.append(PIL.Image.fromarray(x))
        if len(ans) == 1:
            return ans[0]
        return ans

    def is_iimage(self):
        return True

    @property
    def shape(self): return self.data.shape
    @property
    def size(self): return (self.data.shape[-2], self.data.shape[-3])

    def __init__(self, x, vmin=-1, vmax=1):
        if isinstance(x, PIL.Image.Image):
            self.data = np.array(x)
            if self.data.ndim == 2:
                self.data = self.data[..., None]  # (H,W,C)
            self.data = self.data[None]  # (B,H,W,C)
        elif isinstance(x, IImage):
            self.data = x.data.copy()  # Simple Copy
        elif isinstance(x, np.ndarray):
            self.data = x.copy().astype(np.uint8)
            if self.data.ndim == 2:
                self.data = self.data[None, ..., None]
            if self.data.ndim == 3:
                warnings.warn(
                    "Inferred dimensions for a 3D array as (H,W,C), but could've been (B,H,W)")
                self.data = self.data[None]
        elif isinstance(x, torch.Tensor):
            self.data = torch2np(x, vmin, vmax)
        self.device = 'cpu'

    def resize(self, size, *args, **kwargs):
        if size is None:
            return self
        use_small_edge_when_int = kwargs.pop('use_small_edge_when_int', False)
        
        resample = kwargs.pop('filter', PIL.Image.BICUBIC) # Backward compatibility
        resample = kwargs.pop('resample', resample)
        
        if isinstance(size, int):
            if use_small_edge_when_int:
                h, w = self.data.shape[1:3]
                aspect_ratio = h / w
                size = (max(size, int(size * aspect_ratio)),
                        max(size, int(size / aspect_ratio)))
            else:
                h, w = self.data.shape[1:3]
                aspect_ratio = h / w
                size = (min(size, int(size * aspect_ratio)),
                        min(size, int(size / aspect_ratio)))

        if self.size == size[::-1]:
            return self
        return stack([IImage(x.pil().resize(size[::-1], *args, resample=resample, **kwargs)) for x in self])
 
    def pad(self, padding, *args, **kwargs):
        return IImage(tvF.pad(self.torch(0), padding=padding, *args, **kwargs), 0)

    def padx(self, multiplier, *args, **kwargs):
        size = np.array(self.size)
        padding = np.concatenate(
            [[0, 0], np.ceil(size / multiplier).astype(int) * multiplier - size])
        return self.pad(list(padding), *args, **kwargs)

    def pad2wh(self, w=0, h=0, **kwargs):
        cw, ch = self.size
        return self.pad([0, 0, max(0, w - cw), max(0, h-ch)], **kwargs)

    def pad2square(self, *args, **kwargs):
        if self.size[0] > self.size[1]:
            dx = self.size[0] - self.size[1]
            return self.pad([0, dx//2, 0, dx-dx//2], *args, **kwargs)
        elif self.size[0] < self.size[1]:
            dx = self.size[1] - self.size[0]
            return self.pad([dx//2, 0, dx-dx//2, 0], *args, **kwargs)
        return self

    def alpha(self):
        return IImage(self.data[..., -1, None])

    def rgb(self):
        return IImage(self.pil().convert('RGB'))

    def dilate(self, iterations=1, *args, **kwargs):
        return IImage((binary_dilation(self.data, iterations=iterations, *args, *kwargs)*255.).astype(np.uint8))

    def save(self, path):
        _, ext = os.path.splitext(path)
        data = self.data if self.data.ndim == 3 else self.data[0]
        PIL.Image.fromarray(data).save(path)
        return self

    def crop(self, bbox):
        assert len(bbox) in [2,4]
        if len(bbox) == 2:
            x,y = 0,0
            w,h = bbox
        elif len(bbox) == 4:
            x, y, w, h = bbox
        return IImage(self.data[:, y:y+h, x:x+w, :])

    def __getitem__(self, idx):
        return IImage(self.data[None, idx])
