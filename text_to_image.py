from diffusers import DiffusionPipeline, AutoencoderKL
from PIL import Image
import torch

# VAE and diffusion pipeline for initial generation
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

# prompt for initial generation
prompt = "Create a movie poster for a horro film titled 'The man next door'. "

# Generate images based on the prompt
image = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt=4)

# Function to create a grid of images
def image_grid(imgs, rows, cols, resize=256):
    grid_w = cols * resize
    grid_h = rows * resize
    grid = Image.new("RGB", size=(grid_w, grid_h))

    for i, img in enumerate(imgs):
        img = img.resize((resize, resize))
        x = i % cols * resize
        y = i // cols * resize
        grid.paste(img, (x, y))

    return grid

# Display initial grid of images
grid_image = image_grid(image.images, 2, 2)
grid_image.show()

# Define refiner pipeline for refining images
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define refinement parameters
n_steps = 40
high_noise_frac = 0.7

# Generate latent representation of the image for refinement
latent_image = pipe(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

# Refine the image using the refiner pipeline
refined_image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=latent_image,
).images[0]

# Display the refined image
refined_image.show()
