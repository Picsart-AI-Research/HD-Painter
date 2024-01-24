import base64
from typing import Tuple, Union

import cv2
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm

from .iimage import IImage


def tokenize(prompt):
    tokens = open_clip.tokenize(prompt)[0]
    return [open_clip.tokenizer._tokenizer.decoder[x.item()] for x in tokens]


def poisson_blend(
    orig_img: np.ndarray,
    fake_img: np.ndarray,
    mask: np.ndarray,
    pad_width: int = 32,
    dilation: int = 48
) -> np.ndarray:
    """Does poisson blending with some tricks.

    Args:
        orig_img (np.ndarray): Original image.
        fake_img (np.ndarray): Generated fake image to blend.
        mask (np.ndarray): Binary 0-1 mask to use for blending.
        pad_width (np.ndarray): Amount of padding to add before blending (useful to avoid some issues).
        dilation (np.ndarray): Amount of dilation to add to the mask before blending (useful to avoid some issues).

    Returns:
        np.ndarray: Blended image.
    """
    mask = mask[:, :, 0]
    padding_config = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    padded_fake_img = np.pad(fake_img, pad_width=padding_config, mode="reflect")
    padded_orig_img = np.pad(orig_img, pad_width=padding_config, mode="reflect")
    padded_orig_img[:pad_width, :, :] = padded_fake_img[:pad_width, :, :]
    padded_orig_img[:, :pad_width, :] = padded_fake_img[:, :pad_width, :]
    padded_orig_img[-pad_width:, :, :] = padded_fake_img[-pad_width:, :, :]
    padded_orig_img[:, -pad_width:, :] = padded_fake_img[:, -pad_width:, :]
    padded_mask = np.pad(mask, pad_width=padding_config[:2], mode="constant")
    padded_dmask = cv2.dilate(padded_mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    x_min, y_min, rect_w, rect_h = cv2.boundingRect(padded_dmask)
    center = (x_min + rect_w // 2, y_min + rect_h // 2)
    output = cv2.seamlessClone(padded_fake_img, padded_orig_img, padded_dmask, center, cv2.NORMAL_CLONE)
    output = output[pad_width:-pad_width, pad_width:-pad_width]
    return output


def image_from_url_text(filedata):
    if filedata is None:
        return None

    if type(filedata) == list and filedata and type(filedata[0]) == dict and filedata[0].get("is_file", False):
        filedata = filedata[0]

    if type(filedata) == dict and filedata.get("is_file", False):
        filename = filedata["name"]
        filename = filename.rsplit('?', 1)[0]
        return Image.open(filename)

    if type(filedata) == list:
        if len(filedata) == 0:
            return None

        filedata = filedata[0]

    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]

    filedata = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filedata))
    return image


def resize(image: Image, size: Union[int, Tuple[int, int]], resample=Image.BICUBIC):
    if isinstance(size, int):
        w, h = image.size
        aspect_ratio = w / h
        size = (min(size, int(size * aspect_ratio)),
                min(size, int(size / aspect_ratio)))
    return image.resize(size, resample=resample)

