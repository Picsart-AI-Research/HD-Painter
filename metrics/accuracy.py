import os
from pathlib import Path

import cv2
import numpy as np
from mim import download
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from PIL import Image

from utils import get_mask_bbox


class MMDetClassifier:
    def __init__(self, device='cuda', dest_root='./checkpoints/eval/mmdetection'):
        register_all_modules()
        Path(dest_root).mkdir(parents=True, exist_ok=True)
        config_file = f'{dest_root}/rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = f'{dest_root}/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            download('mmdet', ['rtmdet_tiny_8xb32-300e_coco'], dest_root=dest_root)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        
    def _predict(self, img: np.ndarray):
        instances = inference_detector(self.model, img).pred_instances
        labels = [self.model.dataset_meta['classes'][x.labels.item()] for x in instances]
        bboxes = [x.bboxes.cpu().numpy() for x in instances]
        scores = [x.scores.item() for x in instances]
        bboxes, labels, scores = self._score_threshold(bboxes, labels, scores, 0.3)
        return bboxes, labels, scores
    
    def _score_threshold(self, bboxes, labels, scores, score_threshold = 0.):
        bboxes = [bboxes[i] for i in range(len(bboxes)) if scores[i] > score_threshold]
        labels = [labels[i] for i in range(len(labels)) if scores[i] > score_threshold]
        scores = [labels[i] for i in range(len(scores)) if scores[i] > score_threshold]
        return bboxes, labels, scores
    
    def _predict_mask(self, image: np.ndarray, mask: np.ndarray):
        x_min, y_min, x_max, y_max = get_mask_bbox(mask)
        image = image[y_min:y_max, x_min:x_max]
        bboxes, labels, scores = self._predict(image)
        return labels
    
    def eval(self, image: np.ndarray, mask: np.ndarray, label: str, top_k: int = 3) -> float:
        return (1.0 if label in self._predict_mask(image, mask) else 0.0)


mmdet = MMDetClassifier()


def get_score(image: Image, mask: Image, label: str) -> float:
    image = np.array(image)
    mask = np.array(mask)
    return mmdet.eval(image, mask, label)


if __name__ == '__main__':
    image = Image.open('sample_image.png')
    mask = Image.open('sample_mask.png')
    prompt = 'cow'
    print(get_score(image, mask, prompt))
