import cv2
import torch
import numpy as np
from PIL import Image
from torchmetrics.multimodal import CLIPScore

from utils import get_mask_bbox, pad2square

_clip_score = CLIPScore(model_name_or_path='openai/clip-vit-base-patch16').cuda()


def get_score(image: Image, prompt: str, mask: Image = None, device='cuda') -> float:
    image = np.array(image)
    if mask is not None:
        mask = np.array(mask)
        x_min, y_min, x_max, y_max = get_mask_bbox(mask)
        image = image[y_min:y_max, x_min:x_max]
    image = pad2square(image)
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    image = torch.from_numpy(image).to(dtype=torch.float32, device=device)
    with torch.no_grad():
        return _clip_score(image, prompt).item()


if __name__ == '__main__':
    image = Image.open('sample_image.png')
    mask = Image.open('sample_mask.png')
    prompt = 'dog'
    print(get_score(image, prompt, mask))
