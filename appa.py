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

# Import dummy aoti to bypass incompatibility
import aoti 

# --- Model IDs from your list ---
BASE_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TRANSFORMER_ID = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers"
LORA_REPO = "Kijai/WanVideo_comfy"
LORA_FILE = "Wan22-Lightning/wan2.1_t2v_14b_lightning_f8.safetensors" # Lightning LoRA path

# Memory Optimization
def flush():
    gc.collect()
    torch.cuda.empty_cache()

print("Initializing Wan 2.2 for Kaggle T4...")

# 1. 4-bit Quantization Config (Crucial for 14B Model on 16GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. Load the specific 'cbensimon' Transformer in 4-bit
print(f"Loading Transformer from {TRANSFORMER_ID}...")
transformer = WanTransformer3DModel.from_pretrained(
    TRANSFORMER_ID,
    subfolder="transformer",     # Usually 'transformer' or root depending on repo structure
    quantization_config=bnb_config,
    torch_dtype=torch.float16    # Force float16 for T4 compatibility
)

# 3. Load the Base Pipeline (VAE, Text Encoder, Scheduler)
print(f"Loading Base Pipeline from {BASE_MODEL_ID}...")
pipe = WanImageToVideoPipeline.from_pretrained(
    BASE_MODEL_ID,
    transformer=transformer,     # Inject our 4-bit transformer
    torch_dtype=torch.float16
)

# Enable Offloading
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# 4. Bypass AoT (Because zerogpu-aoti crashes on T4)
# We pretend to load it but actually do nothing to prevent errors
aoti.aoti_blocks_load(pipe.transformer, 'zerogpu-aoti/Wan2', variant='fp8da')

# 5. Load Kijai Lightning LoRA
print(f"Loading LoRA from {LORA_REPO}...")
try:
    pipe.load_lora_weights(
        LORA_REPO, 
        weight_name=LORA_FILE,
        adapter_name="lightning"
    )
    pipe.set_adapters(["lightning"], adapter_weights=[1.0])
    print("✓ Lightning LoRA Loaded Successfully")
except Exception as e:
    print(f"! Warning: LoRA load failed ({e}). Check filename/path in Kijai repo.")

# --- Video Generation Function ---
def generate_video(
    image, prompt, steps, duration, guidance_scale, seed, randomize_seed
):
    flush()
    
    if image is None:
        raise gr.Error("Please upload an image")
        
    current_seed = random.randint(0, 2**32-1) if randomize_seed else int(seed)
    print(f"Generating: Seed {current_seed}, Steps {steps}")

    # Resize logic for T4 (Keep it under 480p equivalent to avoid OOM)
    # Standard Wan requirement: (W*H) <= 832*480 approximately for this setup
    w, h = image.size
    aspect = w / h
    # Force 480p based dimensions
    if aspect > 1:
        target_h = 480
        target_w = int(target_h * aspect)
    else:
        target_w = 480
        target_h = int(target_w / aspect)
        
    # Ensure divisible by 16
    target_w = round(target_w / 16) * 16
    target_h = round(target_h / 16) * 16
    
    resized_img = image.resize((target_w, target_h), Image.LANCZOS)
    
    # Calculate frames (Wan usually uses 16fps)
    num_frames = int(duration * 16) 

    try:
        output = pipe(
            image=resized_img,
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(current_seed),
            width=target_w,
            height=target_h
        ).frames[0]
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            export_to_video(output, tmp.name, fps=16)
            return tmp.name, current_seed
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# --- Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# Wan 2.2 (14B) on Kaggle T4")
    gr.Markdown(f"Using weights: `{TRANSFORMER_ID}` & `{LORA_REPO}`")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Input Image")
            prompt_in = gr.Textbox(label="Prompt", value="cinematic motion, smooth animation", lines=3)
            
            with gr.Group():
                steps = gr.Slider(1, 10, value=4, step=1, label="Steps (Lightning uses 4-8)")
                duration = gr.Slider(1.0, 4.0, value=2.0, label="Duration (sec)")
                cfg = gr.Slider(1.0, 5.0, value=1.0, label="Guidance Scale")
                seed = gr.Number(label="Seed (-1 for Random)", value=-1)
                rand_seed = gr.Checkbox(label="Randomize Seed", value=True)
                
            btn = gr.Button("Generate Video", variant="primary")
            
        with gr.Column():
            video_out = gr.Video(label="Result")
            seed_out = gr.Text(label="Seed Used")

    btn.click(
        generate_video,
        inputs=[img_in, prompt_in, steps, duration, cfg, seed, rand_seed],
        outputs=[video_out, seed_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
