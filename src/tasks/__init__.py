"""Task implementations for the AI Unified Platform.

This module contains implementations for various AI tasks like text generation,
image generation, classification, etc.
"""

from .text_generation import (
    TextGenerator,
    summarize_text,
    analyze_sentiment,
    extract_entities
)
from .text_classification import TextClassifier
from .image_processing import (
    generate_image,
    caption_image,
    edit_image
)

__all__ = [
    'TextGenerator',
    'summarize_text',
    'analyze_sentiment',
    'extract_entities',
    'TextClassifier',
    'generate_image',
    'caption_image',
    'edit_image'
]
