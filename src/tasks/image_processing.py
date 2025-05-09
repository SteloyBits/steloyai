from typing import Optional, Dict, Any, List, Tuple
import torch
from PIL import Image
import os
import aiofiles
import asyncio
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
import numpy as np
from datetime import datetime
import io

async def load_model_async(model_id: str, pipeline_class=StableDiffusionPipeline) -> Any:
    """Asynchronously load a model from HuggingFace Hub.
    
    Args:
        model_id: The model ID to load
        pipeline_class: The pipeline class to use
        
    Returns:
        Loaded pipeline
    """
    # Run model loading in a thread pool since it's CPU/GPU bound
    loop = asyncio.get_event_loop()
    pipe = await loop.run_in_executor(
        None,
        lambda: pipeline_class.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

async def save_image_async(image: Image.Image, output_path: str) -> None:
    """Asynchronously save an image to disk.
    
    Args:
        image: PIL Image to save
        output_path: Path to save the image to
    """
    async with aiofiles.open(output_path, 'wb') as f:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        await f.write(img_byte_arr)

async def generate_image(
    prompt: str,
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    negative_prompt: Optional[str] = None
) -> str:
    """Generate an image from a text prompt using Stable Diffusion.
    
    Args:
        prompt: The text prompt to generate the image from
        model: The model ID to use (default: SDXL)
        width: Width of the generated image
        height: Height of the generated image
        guidance_scale: How closely to follow the prompt
        num_inference_steps: Number of denoising steps
        negative_prompt: Optional negative prompt
        
    Returns:
        Path to the generated image
    """
    # Initialize the pipeline asynchronously
    pipe = await load_model_async(model)
    
    # Generate the image in a thread pool since it's GPU bound
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        image = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]
        )
    
    # Create output directory in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, "outputs", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"generated_{timestamp}.png")
    
    # Save the image asynchronously
    await save_image_async(image, output_path)
    return output_path

async def caption_image(
    image_path: str,
    model: str = "Salesforce/blip-image-captioning-large",
    max_length: int = 30,
    num_beams: int = 5
) -> str:
    """Generate a caption for an image using BLIP.
    
    Args:
        image_path: Path to the input image
        model: The model ID to use
        max_length: Maximum length of the caption
        num_beams: Number of beams for beam search
        
    Returns:
        Generated caption
    """
    # Load the image asynchronously
    async with aiofiles.open(image_path, 'rb') as f:
        image_data = await f.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Initialize the model and processor asynchronously
    loop = asyncio.get_event_loop()
    processor = await loop.run_in_executor(None, lambda: AutoProcessor.from_pretrained(model))
    model = await loop.run_in_executor(
        None,
        lambda: AutoModelForVision2Seq.from_pretrained(
            model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    )
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate caption in a thread pool since it's GPU bound
    with torch.no_grad():
        output = await loop.run_in_executor(
            None,
            lambda: model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=1.0,
                early_stopping=True
            )
        )
    
    # Decode the caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

async def edit_image(
    image_path: str,
    prompt: str,
    model: str = "runwayml/stable-diffusion-v1-5",
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50
) -> str:
    """Edit an existing image using img2img.
    
    Args:
        image_path: Path to the input image
        prompt: The text prompt for editing
        model: The model ID to use
        strength: How much to transform the image (0-1)
        guidance_scale: How closely to follow the prompt
        num_inference_steps: Number of denoising steps
        
    Returns:
        Path to the edited image
    """
    # Load the image asynchronously
    async with aiofiles.open(image_path, 'rb') as f:
        image_data = await f.read()
        init_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Initialize the pipeline asynchronously
    pipe = await load_model_async(model, StableDiffusionImg2ImgPipeline)
    
    # Generate the edited image in a thread pool since it's GPU bound
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        image = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]
        )
    
    # Save the image
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = torch.randint(0, 10000, (1,)).item()
    output_path = os.path.join(output_dir, f"edited_{timestamp}.png")
    
    # Save the image asynchronously
    await save_image_async(image, output_path)
    return output_path 