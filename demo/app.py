import os
import sys
from pathlib import Path
from collections import OrderedDict

import gradio as gr
import shutil
import uuid
import torch
from PIL import Image

demo_path = Path(__file__).resolve().parent
root_path = demo_path.parent
sys.path.append(str(root_path))
from src import models
from src.methods import rasg, sd, sr
from src.utils import IImage, poisson_blend, image_from_url_text


TMP_DIR = root_path / 'gradio_tmp'
if TMP_DIR.exists():
    shutil.rmtree(str(TMP_DIR))
TMP_DIR.mkdir(exist_ok=True, parents=True)

os.environ['GRADIO_TEMP_DIR'] = str(TMP_DIR)

negative_prompt_str = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
positive_prompt_str = "Full HD, 4K, high quality, high resolution"

examples_path = root_path / '__assets__/demo/examples'
example_inputs = [
    [f'{examples_path}/images_1024/a40.jpg', f'{examples_path}/images_2048/a40.jpg', 'medieval castle'],
    [f'{examples_path}/images_1024/a4.jpg', f'{examples_path}/images_2048/a4.jpg', 'parrot'],
    [f'{examples_path}/images_1024/a65.jpg', f'{examples_path}/images_2048/a65.jpg', 'hoodie'],
    [f'{examples_path}/images_1024/a54.jpg', f'{examples_path}/images_2048/a54.jpg', 'salad'],
    [f'{examples_path}/images_1024/a51.jpg', f'{examples_path}/images_2048/a51.jpg', 'space helmet'],
    [f'{examples_path}/images_1024/a46.jpg', f'{examples_path}/images_2048/a46.jpg', 'stack of books'],
    [f'{examples_path}/images_1024/a19.jpg', f'{examples_path}/images_2048/a19.jpg', 'antique greek vase'],
    [f'{examples_path}/images_1024/a2.jpg', f'{examples_path}/images_2048/a2.jpg', 'sunglasses'],
]

thumbnails = [
    'https://lh3.googleusercontent.com/pw/ABLVV87bkFc_SRKrbXuk5BTp18dETNm18MLbjoJo6JvwbIkYtjZXrjU_H1dCJIP799OJjHTZmo19mYVyMCC1RLmwqzoZrgwQzfB-SCtxLa83IbXBQ23xzmKoZgsRlPztxNJD6gmXzFyatdLRzDxHIusBQLUz=w3580-h1150-s-no-gm',
    'https://lh3.googleusercontent.com/pw/ABLVV85RWtrpTf1tMp2p3q37eg5DlFp5znifALK_JTjvxJua8UYMjytVoEy2GUW2cLXgBvQyYKg7GvrWXQ5hkdAsyih5Rf4rFnDq-JoiQYhVZHStCZLKxmeAlQna5ZwMPVTKG1TK63DH_OdK58gvSjWtF2ww=w3580-h1152-s-no-gm',
    'https://lh3.googleusercontent.com/pw/ABLVV84dkaU6SQs9fyDjajpk1X9JkYp_zQBEnPVL67oi11_05U6-Ys5ydQpuny8GBQCMyVbFKxJ5unn9w__gmP9K0cKQ4_IVoT7Hvfmya71klDqSI7vu9Iy_5P2Il5-0giJFpumtffBA3kryn1xtJdR4vSA0=w2924-h1858-s-no-gm',
    'https://lh3.googleusercontent.com/pw/ABLVV853ZyjvS4LvcPpVMY9BWz-232omt3-hgRiGcky_3ojE6WLKgtsrftsg1jSrUm2ccT_UOa279CulZy6fdnH_Xg1SunyRBxaRjOK0uxAkUFwb60rR1S4hI2MmhLV7KCi3tw1A-oiGi0f9JINyade-322A=w2622-h1858-s-no-gm',
    'https://lh3.googleusercontent.com/pw/ABLVV86AJGUVGjb0i6CPg8zlJlWObNY0xdOzM1x5Bq9gKhP-ZWre5aaexRJDxQUO2gmJtRIyohD88FJDG_aVX2G5M0QOyGRWlZmx7tOVXLh-Kbesobxo9MfD-wqk9Ts9O8NUGtIwkWzo9SEs2opKdu83gB9F=w2528-h1858-s-no-gm',
    'https://lh3.googleusercontent.com/pw/ABLVV87MplTciS7z-4i-eY3B3L0YhaK8UEQ3pTQD6W6uYVGR4hPD9u1WGEGyfg5ddqU-Bx2BrKskDhwxzF746cRhgFU5aPtbYA_-O7KfqXe9IsMxYCgUKxEHBm2ncqy64V-w-N8XOFgUMkAQqcuuNZ8Xapqp=w3580-h1186-s-no-gm',
    'https://lh3.googleusercontent.com/pw/ABLVV877Esi6l2Kuw3akH5QBlmDAbWydZDZEEJqlZ_N-X7g33NQZU8nv_UKdAVETS7q23byTuldIAhW-q99zCycFB8Yfc-5e_WPNIM9icU0p3gd6DUVZR233ZNUtLca384MYGIhMGud9Y_Xed1I3PpiMhrpG=w2846-h1858-s-no-gm',
    'https://lh3.googleusercontent.com/pw/ABLVV85hMQbSB6fCokdyut4ke7xTUqjERhuYygnj7T8IIA1k48e9GkaowDywPZzi5QJzZfj7wU3bgBHzjxop19qK1zOi5XDrjfXkn5bwj4MxicHa3TG-Rc-V-c1uyZVUyviyUlkGZ62FxuVROw2x0aGJIcr0=w3580-h1382-s-no-gm'
]

example_previews = [
    [thumbnails[0], 'Prompt: medieval castle'],
    [thumbnails[1], 'Prompt: parrot'],
    [thumbnails[2], 'Prompt: hoodie'],
    [thumbnails[3], 'Prompt: salad'],
    [thumbnails[4], 'Prompt: space helmet'],
    [thumbnails[5], 'Prompt: stack of books'],
    [thumbnails[6], 'Prompt: antique greek vase'],
    [thumbnails[7], 'Prompt: sunglasses'],
]

# Load models
# models.pre_download_inpainting_models()
inpainting_models = OrderedDict([
    ("Dreamshaper Inpainting V8", 'ds8_inp'),
    ("Stable-Inpainting 2.0", 'sd2_inp'),
    ("Stable-Inpainting 1.5", 'sd15_inp')
])
sr_model = models.sd2_sr.load_model(device='cuda:0')
sam_predictor = models.sam.load_model(device='cuda:0')

inp_model_name = list(inpainting_models.keys())[0]
inp_model = models.load_inpainting_model(
    inpainting_models[inp_model_name], device='cuda:0', cache=False)


def set_model_from_name(new_inp_model_name):
    global inp_model
    global inp_model_name
    if new_inp_model_name != inp_model_name:
        print (f"Activating Inpaintng Model: {new_inp_model_name}")
        inp_model = models.load_inpainting_model(
            inpainting_models[new_inp_model_name], device='cuda:0', cache=False)
        inp_model_name = new_inp_model_name


def save_user_session(hr_image, hr_mask, lr_results, prompt, session_id=None):
    if session_id == '':
        session_id = str(uuid.uuid4())
    
    session_dir = TMP_DIR / session_id
    session_dir.mkdir(exist_ok=True, parents=True)
    
    hr_image.save(session_dir / 'hr_image.png')
    hr_mask.save(session_dir / 'hr_mask.png')

    lr_results_dir = session_dir / 'lr_results'
    if lr_results_dir.exists():
        shutil.rmtree(lr_results_dir)
    lr_results_dir.mkdir(parents=True)
    for i, lr_result in enumerate(lr_results):
        lr_result.save(lr_results_dir / f'{i}.png')

    with open(session_dir / 'prompt.txt', 'w') as f:
        f.write(prompt)
    
    return session_id


def recover_user_session(session_id):
    if session_id == '':
        return None, None, [], ''
    
    session_dir = TMP_DIR / session_id
    lr_results_dir = session_dir / 'lr_results'

    hr_image = Image.open(session_dir / 'hr_image.png')
    hr_mask = Image.open(session_dir / 'hr_mask.png')
  
    lr_result_paths = list(lr_results_dir.glob('*.png'))
    gallery = []
    for lr_result_path in sorted(lr_result_paths):
        gallery.append(Image.open(lr_result_path))

    with open(session_dir / 'prompt.txt', "r") as f:
        prompt = f.read()

    return hr_image, hr_mask, gallery, prompt


def inpainting_run(model_name, use_rasg, use_painta, prompt, imageMask,
    hr_image, seed, eta, negative_prompt, positive_prompt, ddim_steps,
    guidance_scale=7.5, batch_size=1, session_id=''
):
    torch.cuda.empty_cache()
    set_model_from_name(model_name)

    method = ['default']
    if use_painta: method.append('painta')
    if use_rasg: method.append('rasg')
    method = '-'.join(method)

    if use_rasg:
        inpainting_f = rasg.run
    else:
        inpainting_f = sd.run

    seed = int(seed)
    batch_size = max(1, min(int(batch_size), 4))

    image = IImage(hr_image).resize(512)
    mask = IImage(imageMask['mask']).rgb().resize(512)

    method = ['default']
    if use_painta: method.append('painta')
    method = '-'.join(method)

    inpainted_images = []
    blended_images = []
    for i in range(batch_size):
        seed = seed + i * 1000

        inpainted_image = inpainting_f(
            ddim=inp_model,
            method=method,
            prompt=prompt,
            image=image,
            mask=mask,
            seed=seed,
            eta=eta,
            negative_prompt=negative_prompt,
            positive_prompt=positive_prompt,
            num_steps=ddim_steps,
            guidance_scale=guidance_scale
        ).crop(image.size)

        blended_image = poisson_blend(
            orig_img=image.data[0],
            fake_img=inpainted_image.data[0],
            mask=mask.data[0],
            dilation=12
        )
        blended_images.append(blended_image)
        inpainted_images.append(inpainted_image.pil())

    session_id = save_user_session(
        hr_image, imageMask['mask'], inpainted_images, prompt, session_id=session_id)
    
    return blended_images, session_id


def upscale_run(
    ddim_steps, seed, use_sam_mask, session_id, img_index,
    negative_prompt='', positive_prompt='high resolution professional photo'
):
    hr_image, hr_mask, gallery, prompt = recover_user_session(session_id)

    if len(gallery) == 0:
        return Image.open(root_path / '__assets__/demo/sr_info.png')

    torch.cuda.empty_cache()

    seed = int(seed)
    img_index = int(img_index)

    img_index = 0 if img_index < 0 else img_index
    img_index = len(gallery) - 1 if img_index >= len(gallery) else img_index
    inpainted_image = gallery[img_index if img_index >= 0 else 0]

    output_image = sr.run(
        sr_model,
        sam_predictor,
        inpainted_image,
        hr_image,
        hr_mask,
        prompt=f'{prompt}, {positive_prompt}',
        noise_level=20,
        blend_trick=True,
        blend_output=True,
        negative_prompt=negative_prompt, 
        seed=seed,
        use_sam_mask=use_sam_mask
    )

    return output_image


with gr.Blocks(css=demo_path / 'style.css') as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin-bottom: 0.5rem">
            🧑‍🎨 HD-Painter Demo
        </h1>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        Hayk Manukyan<sup>1*</sup>, Andranik Sargsyan<sup>1*</sup>, Barsegh Atanyan<sup>1</sup>, Zhangyang Wang<sup>1,2</sup>, Shant Navasardyan<sup>1</sup>
        and <a href="https://www.humphreyshi.com/home">Humphrey Shi</a><sup>1,3</sup>
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        <sup>1</sup>Picsart AI Resarch (PAIR), <sup>2</sup>UT Austin, <sup>3</sup>Georgia Tech
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
        [<a href="https://arxiv.org/abs/2312.14091" style="color:blue;">arXiv</a>] 
        [<a href="https://github.com/Picsart-AI-Research/HD-Painter" style="color:blue;">GitHub</a>]
        </h2>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0.7rem auto; max-width: 1000px">
        <b>HD-Painter</b> enables prompt-faithfull and high resolution (up to 2k) image inpainting upon any diffusion-based image inpainting method.
        </h2>
        </div>
        """)

    with open(demo_path / 'script.js', 'r') as f:
        js_str = f.read()

    demo.load(_js=js_str)

    with gr.Row():
        with gr.Column():
            model_picker = gr.Dropdown(
                list(inpainting_models.keys()),
                value=list(inpainting_models.keys())[0],
                label = "Please select a model!",
            )
        with gr.Column():
            use_painta = gr.Checkbox(value = True, label = "Use PAIntA")
            use_rasg = gr.Checkbox(value = True, label = "Use RASG")

    prompt = gr.Textbox(label = "Inpainting Prompt")
    with gr.Row():
        with gr.Column():
            imageMask = gr.ImageMask(label = "Input Image", brush_color='#ff0000', elem_id="inputmask", type="pil")
            hr_image = gr.Image(visible=False, type="pil")
            hr_image.change(fn=None, _js="function() {setTimeout(imageMaskResize, 200);}", inputs=[], outputs=[])
            imageMask.upload(
                fn=None,
                _js="async function (a) {hr_img = await resize_b64_img(a['image'], 2048); dp_img = await resize_b64_img(hr_img, 1024); return [hr_img, {image: dp_img, mask: null}]}",
                inputs=[imageMask],
                outputs=[hr_image, imageMask],
            )
            with gr.Row():
                inpaint_btn = gr.Button("Inpaint", scale = 0)
   
            with gr.Accordion('Advanced options', open=False):
                guidance_scale = gr.Slider(minimum = 0, maximum = 30, value = 7.5, label = "Guidance Scale")
                eta = gr.Slider(minimum = 0, maximum = 1, value = 0.1, label = "eta")
                ddim_steps = gr.Slider(minimum = 10, maximum = 100, value = 50, step =  1, label = 'Number of diffusion steps')
                with gr.Row():
                    seed = gr.Number(value = 49123, label = "Seed")
                    batch_size = gr.Number(value = 1, label = "Batch size", minimum=1, maximum=4) 
                negative_prompt = gr.Textbox(value=negative_prompt_str, label = "Negative prompt", lines=3)
                positive_prompt = gr.Textbox(value=positive_prompt_str, label = "Positive prompt", lines=1)

        with gr.Column():
            with gr.Row():
                output_gallery = gr.Gallery(
                    [],
                    columns = 4,
                    preview = True,
                    allow_preview = True,
                    object_fit='scale-down',
                    elem_id='outputgallery'
                )
            with gr.Row():
                upscale_btn = gr.Button("Send to Inpainting-Specialized Super-Resolution (x4)", scale = 1)
            with gr.Row():
                use_sam_mask = gr.Checkbox(value = False, label = "Use SAM mask for background preservation (for SR only, experimental feature)")
            with gr.Row():
                hires_image = gr.Image(label = "Hi-res Image")
    
    label = gr.Markdown("## High-Resolution Generation Samples (2048px large side)")
    
    with gr.Column():
        example_container = gr.Gallery(
            example_previews,
            columns = 4,
            preview = True,
            allow_preview = True,
            object_fit='scale-down'
        )

        gr.Examples(
            [example_inputs[i] + [[example_previews[i]]]
                for i in range(len(example_previews))],
            [imageMask, hr_image, prompt, example_container],
            elem_id='examples'
        )

    session_id = gr.Textbox(value='', visible=False)
    html_info = gr.HTML(elem_id=f'html_info', elem_classes="infotext")

    inpaint_btn.click(
        fn=inpainting_run, 
        inputs=[
            model_picker,
            use_rasg,
            use_painta,
            prompt,
            imageMask,
            hr_image,
            seed,
            eta,
            negative_prompt,
            positive_prompt,
            ddim_steps,
            guidance_scale,
            batch_size,
            session_id
        ], 
        outputs=[output_gallery, session_id], 
        api_name="inpaint"
    )
    upscale_btn.click(
        fn=upscale_run, 
        inputs=[
            ddim_steps,
            seed,
            use_sam_mask,
            session_id,
            html_info
        ],
        outputs=[hires_image], 
        api_name="upscale",
        _js="function(a, b, c, d, e){ return [a, b, c, d, selected_gallery_index()] }",
    )

demo.queue(max_size=20)
demo.launch(share=True, allowed_paths=[str(TMP_DIR)])