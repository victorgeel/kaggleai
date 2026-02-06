import os
import torch
import gc
import gradio as gr
import random
import tempfile
from PIL import Image
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig

# local aoti.py ကို လှမ်းခေါ်ခြင်း
import aoti 

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Memory Optimization
def flush():
    gc.collect()
    torch.cuda.empty_cache()

print("Loading 4-bit Quantized Model for Kaggle T4...")

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Transformer
transformer = WanTransformer3DModel.from_pretrained(
    "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers", # T4 အတွက် အဆင်ပြေဆုံး variant
    subfolder="transformer",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Load Pipeline
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.float16
)

# Crucial for T4 16GB
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Bypass AoT (Kaggle compatibility)
aoti.aoti_blocks_load(pipe.transformer, 'zerogpu-aoti/Wan2', variant='fp8da')

def generate(image, prompt, steps, duration, cfg, seed):
    flush()
    current_seed = random.randint(0, 2**32-1) if seed == -1 else int(seed)
    
    # Standard resize to 480p for T4 safety
    w, h = image.size
    image = image.resize((832, 480), Image.LANCZOS)
    
    output = pipe(
        image=image,
        prompt=prompt,
        num_frames=int(duration * 16),
        num_inference_steps=int(steps),
        guidance_scale=cfg,
        generator=torch.Generator("cuda").manual_seed(current_seed)
    ).frames[0]
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        export_to_video(output, tmp.name, fps=16)
        return tmp.name

# Simple Gradio UI
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Prompt"),
        gr.Slider(1, 10, value=6, label="Steps"),
        gr.Slider(1.0, 4.0, value=2.0, label="Duration"),
        gr.Slider(1.0, 7.0, value=1.0, label="CFG"),
        gr.Number(value=-1, label="Seed (-1 for random)")
    ],
    outputs=gr.Video(),
    title="Wan 2.2 I2V - Kaggle T4 Edition"
)

demo.launch(share=True)
