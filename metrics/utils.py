import requests
from typing import Tuple, Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    if len(mask.shape) == 3:
        mask = np.sum(mask, axis=2)
    xm = np.argwhere(np.sum(mask, axis=0) > 0).ravel()
    ym = np.argwhere(np.sum(mask, axis=1) > 0).ravel()
    x_min, x_max = np.min(xm), np.max(xm) + 1
    y_min, y_max = np.min(ym), np.max(ym) + 1
    return x_min, y_min, x_max, y_max


def pad2square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    dx = abs(w - h)
    padding = ((0, 0), (dx//2, dx-dx//2))
    padding = padding[::-1] if w > h else padding
    if len(image.shape) == 3:
        padding = (*padding, (0, 0)) 
    return np.pad(image, padding)


def resize(image: Image, size: Union[int, Tuple[int, int]], resample=Image.BICUBIC):
    if isinstance(size, int):
        w, h = image.size
        aspect_ratio = w / h
        size = (min(size, int(size * aspect_ratio)),
                min(size, int(size / aspect_ratio)))
    return image.resize(size, resample=resample)


def download_file(url: str, save_path: str, chunk_size: int=1024) -> None:
    try:
        save_path = Path(save_path)
        if save_path.exists():
            print(f'{save_path.name} exists')
            return
        save_path.parent.mkdir(exist_ok=True, parents=True)
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path.name,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
        print(f'{save_path.name} download finished')
    except Exception as e:
        raise Exception(f'Download failed: {e}')
