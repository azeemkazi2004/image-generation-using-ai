!pip install -q diffusers accelerate gradio torch matplotlib
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

# ---------------------------
# Load model
# ---------------------------
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Optional but recommended (reduces memory usage)
pipe.enable_attention_slicing()

# ---------------------------
# Image generation function
# ---------------------------
def generate_image(prompt, steps, guidance):
    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    
    return image

# ---------------------------
# Gradio UI
# ---------------------------
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="A futuristic spaceship in deep space"
        ),
        gr.Slider(
            minimum=10, maximum=50, value=25, step=1,
            label="Inference Steps"
        ),
        gr.Slider(
            minimum=1, maximum=15, value=7.5, step=0.5,
            label="Guidance Scale"
        )
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion Text-to-Image",
    description="Generate images using Stable Diffusion v1.5"
)

# ---------------------------
# Launch app
# ---------------------------
demo.launch()
