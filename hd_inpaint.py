import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src import models
from src.methods import rasg, sd, sr
from src.utils import IImage, resize

logging.disable(logging.INFO)

root_path = Path(__file__).resolve().parent.parent
negative_prompt = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
positive_prompt = "Full HD, 4K, high quality, high resolution"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-path', type=Path, help='Image path.', required=True)
    parser.add_argument('--mask-path', type=Path, help='Mask path.', required=True)
    parser.add_argument('--prompt', type=str, help='Text prompt.', required=True)
    parser.add_argument('--output-dir', type=Path, help='Output dir.', required=True)
    parser.add_argument('--num-samples', type=int, help='Num of samples', default=1)

    parser.add_argument('--model-id', type=str, default='ds8_inp',
        help='One of [ds8_inp, sd2_inp, sd15_inp]', required=False)
    parser.add_argument('--method', type=str, default='painta+rasg',
        help='One of [baseline, painta, rasg, painta+rasg]', required=False)
    parser.add_argument('--sr-method', type=str,
        help='Superresolution method. One of [baseline, inpainting_specialized]',
        default='inpainting_specialized')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
        help='Classifier-free guidance scale.', required=False)
    parser.add_argument('--rasg-eta', type=float, default=0.1,
        help='RASG eta value.', required=False)
    parser.add_argument('--num-steps', type=int, default=50,
        help='Num of DDIM steps.', required=False)
    parser.add_argument('--seed', type=int, default=1,
        help='Seed to use for generation.', required=False)
    return parser.parse_args()


def get_inpainting_function(
    model_id: str,
    method: str,
    negative_prompt: str = '',
    positive_prompt: str = '',
    num_steps: int = 50,
    eta: float = 0.25,
    guidance_scale: float = 7.5
):
    inp_model = models.load_inpainting_model(model_id, device='cuda:0', cache=True)
    
    if 'rasg' in method:
        runner = rasg
    else:
        runner = sd
    
    def run(image: Image, mask: Image, prompt: str, seed: int = 1) -> Image:
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


def get_inpainting_sr_function(
    positive_prompt='high resolution professional photo',
    negative_prompt='',
    noise_level=20,
    use_sam_mask=False,
    blend_trick=True,
    blend_output=True
):
    sr_model = models.sd2_sr.load_model(device='cuda:0')
    sam_predictor = None
    if use_sam_mask:
        sam_predictor = models.sam.load_model(device='cuda:0')

    def run(inpainted_image: Image, image: Image, mask: Image, prompt: str, seed: int = 1) -> Image:
        return sr.run(
            sr_model,
            sam_predictor,
            inpainted_image,
            image,
            mask,
            prompt=f'{prompt}, {positive_prompt}',
            noise_level=noise_level,
            blend_trick=blend_trick,
            blend_output=blend_output,
            negative_prompt=negative_prompt, 
            seed=seed,
            use_sam_mask=use_sam_mask
        )
    return run


def main():
    args = get_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    run_inpainting = get_inpainting_function(
        model_id=args.model_id,
        method=args.method,
        eta=args.rasg_eta,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        negative_prompt=negative_prompt,
        positive_prompt=positive_prompt
    )

    run_inpainting_sr = get_inpainting_sr_function(
        blend_trick=args.sr_method == 'inpainting_specialized'
    )

    image = Image.open(args.image_path).convert('RGB')
    mask = Image.open(args.mask_path).convert('RGB')
    prompt = args.prompt

    resized_image = resize(image, 512)
    resized_mask = resize(mask, 512)

    for idx in tqdm(range(1, args.num_samples+1)):
        seed = args.seed + (idx-1) * 1000
        inpainted_image = run_inpainting(resized_image, resized_mask, prompt, seed=seed)
        inpainted_hd_image = run_inpainting_sr(inpainted_image, image, mask, prompt, seed=seed)

        if args.num_samples > 1:
            output_dir = args.output_dir / args.image_path.stem
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f'{idx}.jpg'
        else:
            output_path = args.output_dir / f'{args.image_path.stem}.jpg'

        inpainted_hd_image.save(output_path)


if __name__ == '__main__':
    main()