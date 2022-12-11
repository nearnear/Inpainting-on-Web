import torch
import gradio as gr
from huggingface_hub import login
from diffusers import StableDiffusionInpaintPipeline

device = torch.device("cuda")


def load_model():
    personal_token = HF_PERSONAL_TOKEN

    login(token=personal_token,
          add_to_git_credential=True)

    model_path = "runwayml/stable-diffusion-inpainting"

    return StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True
    ).to(device)


def in_paint_image_with_prompt(inputs, prompt):
    image = inputs['image'].convert("RGB").resize((512, 512))
    mask_image = inputs['mask'].convert("RGB").resize((512, 512))

    guidance_scale = 7.5
    num_samples = 3
    generator = torch.Generator().manual_seed(2022)
    pipe = load_model()

    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    return pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]


# get gradio demo
gr.Interface(
    in_paint_image_with_prompt,
    title='Stable Diffusion In-Painting',
    inputs=[
        gr.Image(source='upload', tool='sketch', type='pil'),
        gr.Textbox(label='prompt')
    ],
    outputs=[
        gr.Image()
        ]
).launch(share=True) # For debugging, set `.launch(debug=True)`
