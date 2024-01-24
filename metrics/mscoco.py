
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from PIL import Image

    
class MSCOCO:
    def __init__(self, root_folder, subset='val2017', use_convex_hull_masks=False, verbose=False):
        self.root_folder = Path(root_folder)
        self.subset = subset
        self.use_convex_hull_masks = use_convex_hull_masks
        self.verbose = verbose

        captions_path = self.root_folder / 'annotations' / f'captions_{subset}.json'
        instances_path = self.root_folder / 'annotations' / f'instances_{subset}.json'

        with open(captions_path) as f:
            captions = json.load(f)
        # index captions
        self.image_id_to_caption = {x['image_id']: x['caption'] for x in captions['annotations']}

        with open(instances_path) as f:
            instances = json.load(f)
        # index category names
        self.category_id_to_label = {x['id']: x['name'] for x in instances['categories']}

        # index image paths
        self.image_id_to_image_path = {x['id']: self.root_folder/ subset / x['file_name'] 
                                            for x in instances['images']}

        # index image annotations (segments)
        self.image_id_to_annotations = defaultdict(list)
        for annotation in instances['annotations']:
            if isinstance(annotation['segmentation'], list):
                image_id = annotation['image_id']
                annotation['label'] = self.category_id_to_label[annotation['category_id']]
                self.image_id_to_annotations[image_id].append(annotation)
        
        self.image_ids = sorted(list(self.image_id_to_image_path.keys()))
        
        if verbose:
            num_annotations = 0
            for image_id in self.image_id_to_annotations:
                num_annotations += len(self.image_id_to_annotations[image_id])
            print(f'Loaded MSCOCO subset: {subset}')
            print(f'Number of images: {len(self.image_ids)}')
            print(f'Total number of annotations: {num_annotations}')

    @staticmethod
    def draw_mask_from_annotation(image, annotation, use_convex_hull_masks=False):
        mask_contours = annotation['segmentation']
        mask_contours = [np.int32(x).reshape(-1,2) for x in mask_contours]
        if use_convex_hull_masks:
            mask_contours = [cv2.convexHull(np.concatenate(mask_contours))]
        canvas = np.zeros(image.size[:2][::-1], np.uint8)
        canvas = cv2.drawContours(canvas, mask_contours, -1, (255,255,255), -1)
        mask = Image.fromarray(canvas).convert('RGB')
        return mask

    def __getitem__(self, item):
        segment_idx = None
        if type(item) is int:
            image_idx = item
        elif type(item) is tuple:
            if len(item) == 2:
                image_idx, segment_idx = item
            else:
                raise ValueError(f'incorrect argument {item} provided to MSCOCO __getitem__')
        else:
            raise ValueError(f'incorrect argument {item} provided to MSCOCO __getitem__')
            

        image_id = self.image_ids[image_idx]
        image_path = self.image_id_to_image_path[image_id]
        image_name = f'{image_id}'
        image = Image.open(image_path).convert('RGB')
        caption = self.image_id_to_caption[image_id]
        annotations = self.image_id_to_annotations[image_id]

        if segment_idx is not None:
            if segment_idx < 0 or segment_idx >= len(annotations):
                raise IndexError(
                    f'segment_idx={segment_idx} is out of possible range [0, {len(annotations)-1}]')
            annotation = annotations[segment_idx]
            mask = MSCOCO.draw_mask_from_annotation(
                image, annotation, self.use_convex_hull_masks)
            label = annotation['label']
            image_name = f'{image_name}_{segment_idx}'
            return image_name, image, mask, label
    
        return image_name, image, annotations, caption

    def __len__(self):
        return len(self.image_ids)


class MSCOCOSubset:
    def __init__(self, subset_csv_path: Path, mscoco_ds: MSCOCO, verbose=False):
        self.mscoco_ds = mscoco_ds
        self.mscoco_subset_df = pd.read_csv(subset_csv_path, header=0)
        if verbose:
            print(f'Number of loaded samples from csv: {len(self)}')

    def __getitem__(self, idx):
        sample = self.mscoco_subset_df.iloc[idx]
        image_idx, segment_idx = sample['image_idx'], sample['segment_idx']
        image_name, image, mask, prompt = self.mscoco_ds[image_idx, segment_idx]
        return image_name, image, mask, prompt

    def __len__(self):
        return len(self.mscoco_subset_df)


if __name__ == '__main__':
    ds = MSCOCO(
        root_folder = './mscoco_data',
        subset='val2017',
        use_convex_hull_masks=True,
        verbose=True
    )
    image_idx = 3478
    image, annotations, caption = ds[image_idx]
    print(f'Number of annotations for idx {image_idx}: ', len(annotations))

    image, mask, prompt = ds[image_idx, 0]
    print('Image size:', image.size)
    print('Mask size:', mask.size)
    print('Prompt:', prompt)
    image.save('sample_image.png')
    mask.save('sample_mask.png')
