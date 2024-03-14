import argparse
import json
import logging
import os
import sys
from csv import DictWriter
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Subset
from tqdm import tqdm

import accuracy
import aesthetic
import clipscore
import pickscore
from mscoco import MSCOCO, MSCOCOSubset
from src import models
from src.methods import rasg, sd
from src.utils import IImage, resize

logging.disable(logging.INFO)

root_path = Path(__file__).resolve().parent.parent
negative_prompt = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
positive_prompt = "Full HD, 4K, high quality, high resolution"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=str, default='ds8_inp',
                        help='One of [ds8_inp, sd2_inp, sd15_inp]', required=False)
    parser.add_argument('--mscoco-test-csv', type=Path,
                        default=f'{root_path}/__assets__/mscoco_testset_10000.csv',
                        help='Path to MSCOCO validation subset.', required=False)
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Path to save evaluation outputs.')
    parser.add_argument('--mscoco-dir', type=Path, default=f'{root_path}/mscoco_data',
                        help='Path to MSCOCO dataset directory.', required=False)
    parser.add_argument('--method', type=str, default='painta+rasg',
                        help='One of [baseline, painta, rasg, painta+rasg]', required=False)
    parser.add_argument('--rasg-eta', type=float, default=0.25,
                        help='RASG eta value.', required=False)
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed to use for generation.', required=False)
    parser.add_argument('--neg-pos-prompts', action='store_true',
                        help='Use negative and positive prompts for generation.')
    parser.add_argument('--continue-eval', action='store_true',
                        help='Continue evaluation.')

    # please use --save-png to save base images for further pickscore calculation
    parser.add_argument('--pickscore-base-results-dir', type=Path, default=None,
                        help='Pregenerated base results dir for pickscore calculation for the curent method.')
    parser.add_argument('--save-png', action='store_true',
                        help='Save also png images?.')

    parser.add_argument('--use-results-dir', type=Path, default=None,
                        help='Use results from results dir.')

    # inference with partitions
    parser.add_argument('--num-partitions', type=int, default=None,
                        help='Num of partitions.')
    parser.add_argument('--partition-number', type=int, default=None,
                        help='Partition number.')

    return parser.parse_args()


class Metrics:
    def __init__(self, init_metrics_path=None):
        if init_metrics_path is not None:
            df = pd.read_csv(init_metrics_path)
            self.clipscore_sum = np.sum(df['clipscore'])
            self.accuracy_sum = np.sum(df['accuracy'])
            self.aesthetic_sum = np.sum(df['aesthetic'])
            self.pickscore_sum = np.sum(df['pickscore'])
            self.num_samples = len(df)
        else:
            self.clipscore_sum = 0.0
            self.accuracy_sum = 0.0
            self.aesthetic_sum = 0.0
            self.pickscore_sum = 0.0
            self.num_samples = 0.0

    def update(self, base_image, mask, mscoco_class, prompt, inpainted_image):
        metrics = {}

        metrics['accuracy'] = accuracy.get_score(inpainted_image, mask, mscoco_class)
        metrics['aesthetic'] = aesthetic.get_score(inpainted_image)
        metrics['clipscore'] = clipscore.get_score(inpainted_image, prompt, mask)
        metrics['pickscore'] = pickscore.get_score(base_image, inpainted_image, prompt)

        self.accuracy_sum += metrics['accuracy']
        self.aesthetic_sum += metrics['aesthetic']
        self.clipscore_sum += metrics['clipscore']
        self.pickscore_sum += metrics['pickscore']

        self.num_samples += 1

        return metrics

    def get_scores(self):
        return {
            'accuracy': self.accuracy_sum / self.num_samples,
            'aesthetic': self.aesthetic_sum / self.num_samples,
            'clipscore': self.clipscore_sum / self.num_samples,
            'pickscore': self.pickscore_sum / self.num_samples
        }


def get_inpainting_function(
        model_id: str,
        method: str,
        negative_prompt: str = '',
        positive_prompt: str = '',
        num_steps: int = 50,
        eta: float = 0.25,
        guidance_scale: float = 7.5,
        seed: int = 1
):
    inp_model = models.load_inpainting_model(model_id, cache=True)

    if 'rasg' in method:
        runner = rasg
    else:
        runner = sd

    def run(image: Image, mask: Image, prompt: str) -> Image:
        inpainted_image = runner.run(
            ddim=inp_model,
            method=method,
            prompt=prompt,
            image=IImage(image),
            mask=IImage(mask),
            seed=seed,
            eta=eta,
            negative_prompt=negative_prompt,
            positive_prompt=positive_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale
        ).pil()
        w, h = image.size
        inpainted_image = Image.fromarray(np.array(inpainted_image)[:h, :w])
        return inpainted_image

    return run


def get_save_outputs_function(
        output_dir: Path,
        save_preprocessed_image_mask: bool = False,
        log_metrics: bool = True,
        continue_eval: bool = False,
        num_partitions=1,
        partition_number=1,
        save_png=False
):
    output_dir.mkdir(exist_ok=True, parents=True)
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    results_dir = output_dir / 'results'
    if save_png:
        results_png_dir = output_dir / 'results_png'

    if num_partitions is not None and num_partitions > 1:
        log_csv_path = output_dir / f'metrics_part_{partition_number}_of_{num_partitions}.csv'
    else:
        log_csv_path = output_dir / 'metrics.csv'
    if os.path.exists(log_csv_path) and not continue_eval:
        os.remove(log_csv_path)

    if save_preprocessed_image_mask:
        images_dir.mkdir(exist_ok=True, parents=True)
        masks_dir.mkdir(exist_ok=True, parents=True)
    if save_png:
        results_png_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    def save(
            idx: int, image_name: str, image: Image, mask: Image, prompt: str, inpainted_image: Image,
            scores: Optional[dict] = None
    ):
        inpainted_image.save(results_dir / f'{image_name}.jpg')
        if save_png:
            inpainted_image.save(results_png_dir / f'{image_name}.png')

        if save_preprocessed_image_mask:
            image.save(images_dir / f'{image_name}.jpg')
            mask.save(masks_dir / f'{image_name}.png')
        if log_metrics:
            row = {
                'image_name': image_name,
                'prompt': prompt,
                **scores
            }
            fieldnames = ['image_name', 'prompt'] + list(sorted(scores.keys()))
            with open(log_csv_path, 'a') as f:
                writer = DictWriter(f, fieldnames=fieldnames)
                if idx == 0:
                    writer.writeheader()
                writer.writerow(row)
    return save


def main():
    args = get_args()
    print(f'Running evaluation on {args.model_id}...')

    if args.output_dir is None:
        args.output_dir = root_path / 'outputs' / args.model_id / args.method
        args.output_dir.mkdir(exist_ok=True, parents=True)

    mscoco_ds = MSCOCO(
        root_folder=args.mscoco_dir, subset='val2017', use_convex_hull_masks=True, verbose=True)
    mscoco_test_ds = MSCOCOSubset(
        subset_csv_path=args.mscoco_test_csv, mscoco_ds=mscoco_ds, verbose=True)

    metrics = Metrics(init_metrics_path=args.output_dir / 'metrics.csv' if args.continue_eval else None)

    run_inpainting = get_inpainting_function(
        model_id=args.model_id,
        method=args.method,
        eta=args.rasg_eta,
        seed=args.seed,
        negative_prompt=negative_prompt if args.neg_pos_prompts else '',
        positive_prompt=positive_prompt if args.neg_pos_prompts else ''
    )
    save_outputs = get_save_outputs_function(
        output_dir=args.output_dir,
        continue_eval=args.continue_eval,
        num_partitions=args.num_partitions,
        partition_number=args.partition_number,
        save_png=args.save_png
    )

    if args.num_partitions is not None and args.num_partitions > 1:
        total_num_samples = len(mscoco_test_ds)
        samples_per_partition = total_num_samples // args.num_partitions
        assert args.partition_number > 0
        start_idx = (args.partition_number - 1) * samples_per_partition
        end_idx = args.partition_number * samples_per_partition
        eval_ds = Subset(mscoco_test_ds, range(start_idx, end_idx))
    else:
        eval_ds = mscoco_test_ds

    for idx, (image_name, image, mask, mscoco_class) in tqdm(list(enumerate(eval_ds))):
        if idx < metrics.num_samples:
            continue
        image = resize(image, 512)
        mask = resize(mask, 512)
        prompt = mscoco_class

        if args.pickscore_base_results_dir is not None:
            result_ext = '.png'
            base_image_path = args.pickscore_base_results_dir / f'{image_name}{result_ext}'
            base_image = Image.open(base_image_path)
        else:
            base_image = image

        if args.use_results_dir is not None:
            result_ext = '.png'
            inpainted_image_path = args.use_results_dir / f'{image_name}{result_ext}'
            inpainted_image = Image.open(inpainted_image_path)
        else:
            inpainted_image = run_inpainting(image, mask, prompt)

        scores = metrics.update(base_image, mask, mscoco_class, prompt, inpainted_image)
        save_outputs(idx, image_name, image, mask, prompt, inpainted_image, scores)

    averaged_scores = metrics.get_scores()
    print('Metrics: ', averaged_scores)

    if args.num_partitions is not None and args.num_partitions > 1:
        with open(args.output_dir / f'final_metrics_part_{args.partition_number}_of_{args.num_partitions}.json',
                  'w') as f:
            json.dump(averaged_scores, f)
    else:
        with open(args.output_dir / 'final_metrics.json', 'w') as f:
            json.dump(averaged_scores, f)


if __name__ == '__main__':
    main()