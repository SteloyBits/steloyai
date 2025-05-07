from typing import List, Dict, Any
from huggingface_hub import HfApi
import logging

logger = logging.getLogger(__name__)

def verify_models_availability(models: List[str]) -> List[str]:
    """Verify which models are available on HuggingFace Hub.
    
    Args:
        models: List of model IDs to verify
        
    Returns:
        List of verified available model IDs
    """
    api = HfApi()
    available_models = []
    
    for model_id in models:
        try:
            # Check if model exists and is accessible
            model_info = api.model_info(model_id)
            if model_info:
                available_models.append(model_id)
                logger.info(f"Model {model_id} is available")
            else:
                logger.warning(f"Model {model_id} not found")
        except Exception as e:
            logger.warning(f"Error checking model {model_id}: {str(e)}")
    
    return available_models

def get_verified_config() -> Dict[str, Any]:
    """Get configuration with verified available models.
    
    Returns:
        Dictionary containing verified model configurations
    """
    # Define model lists for each task
    model_lists = {
        "text_generation": [
            "gpt2",  # Smaller, widely available model
            "facebook/opt-125m",  # Small OPT model
            "EleutherAI/pythia-70m"  # Small Pythia model
        ],
        "text_summarization": [
            "facebook/bart-large-cnn",
            "t5-small"  # Smaller T5 model
        ],
        "sentiment_analysis": [
            "distilbert-base-uncased-finetuned-sst-2-english",  # Smaller, widely available
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ],
        "ner": [
            "dbmdz/bert-base-german-cased",  # More widely available
            "Jean-Baptiste/camembert-ner"  # French NER model
        ],
        "image_generation": [
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4"
        ],
        "image_captioning": [
            "Salesforce/blip-image-captioning-base",
            "nlpconnect/vit-gpt2-image-captioning"
        ],
        "object_detection": [
            "facebook/detr-resnet-50",
            "hustvl/yolos-small"  # Smaller YOLOS model
        ]
    }
    
    # Verify each model list
    verified_config = {}
    for task, models in model_lists.items():
        verified_models = verify_models_availability(models)
        if verified_models:
            verified_config[task] = {"available_models": verified_models}
        else:
            logger.error(f"No available models found for task: {task}")
            verified_config[task] = {"available_models": []}
    
    return verified_config 