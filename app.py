import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    layout="centered"
)

st.title("🎨 Stable Diffusion Image Generator")
st.write("Generate images using **Stable Diffusion v1.5**")

# ---------------------------
# Device setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    return pipe

pipeline = load_pipeline()

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Settings")

num_steps = st.sidebar.slider(
    "Inference Steps",
    min_value=10,
    max_value=50,
    value=25
)

guidance_scale = st.sidebar.slider(
    "Guidance Scale (CFG)",
    min_value=1.0,
    max_value=15.0,
    value=7.5,
    step=0.5
)

# ---------------------------
# Prompt input
# ---------------------------
prompt = st.text_input(
    "Enter your prompt",
    value="a spaceship in space"
)

generate = st.button("🚀 Generate Image")

# ---------------------------
# Image generation
# ---------------------------
if generate:
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            image = pipeline(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            ).images[0]

        st.image(image, caption=f"Prompt: {prompt}", use_container_width=True)
        st.success("Image generation complete!")
