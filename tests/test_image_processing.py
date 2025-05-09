import pytest
import torch
from unittest.mock import Mock, patch
from PIL import Image
import os
from src.tasks.image_processing import generate_image, caption_image, edit_image

@pytest.fixture
def mock_image():
    # Create a small test image
    image = Image.new('RGB', (100, 100), color='red')
    return image

@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.return_value.images = [Image.new('RGB', (100, 100), color='red')]
    return pipeline

def test_generate_image(mock_pipeline):
    with patch('diffusers.StableDiffusionPipeline.from_pretrained', return_value=mock_pipeline):
        # Test image generation
        output_path = generate_image(
            prompt="test prompt",
            width=512,
            height=512
        )
        
        assert os.path.exists(output_path)
        assert output_path.endswith('.png')
        mock_pipeline.assert_called_once()

def test_caption_image(mock_image):
    with patch('PIL.Image.open', return_value=mock_image), \
         patch('transformers.AutoProcessor.from_pretrained') as mock_processor, \
         patch('transformers.AutoModelForVision2Seq.from_pretrained') as mock_model:
        
        # Setup mocks
        mock_processor.return_value = Mock()
        mock_model.return_value = Mock()
        mock_processor.return_value.decode.return_value = "A test caption"
        
        # Test caption generation
        caption = caption_image("test_image.jpg")
        
        assert isinstance(caption, str)
        assert caption == "A test caption"
        mock_processor.assert_called_once()
        mock_model.assert_called_once()

def test_edit_image(mock_image, mock_pipeline):
    with patch('PIL.Image.open', return_value=mock_image), \
         patch('diffusers.StableDiffusionImg2ImgPipeline.from_pretrained', return_value=mock_pipeline):
        
        # Test image editing
        output_path = edit_image(
            image_path="test_image.jpg",
            prompt="test prompt",
            strength=0.75
        )
        
        assert os.path.exists(output_path)
        assert output_path.endswith('.png')
        mock_pipeline.assert_called_once()

def test_device_handling():
    # Test CPU fallback when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False), \
         patch('diffusers.StableDiffusionPipeline.from_pretrained') as mock_pipeline:
        
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value.images = [Image.new('RGB', (100, 100), color='red')]
        
        output_path = generate_image("test prompt")
        assert os.path.exists(output_path)

def test_error_handling():
    # Test with invalid model
    with pytest.raises(Exception):
        generate_image("test", model="invalid-model")
    
    with pytest.raises(Exception):
        caption_image("test.jpg", model="invalid-model")
    
    with pytest.raises(Exception):
        edit_image("test.jpg", "test", model="invalid-model")

def test_output_directory_creation():
    # Test that output directory is created if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "images")
    if os.path.exists(output_dir):
        os.rmdir(output_dir)
    
    with patch('diffusers.StableDiffusionPipeline.from_pretrained') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value.images = [Image.new('RGB', (100, 100), color='red')]
        
        generate_image("test prompt")
        assert os.path.exists(output_dir) 