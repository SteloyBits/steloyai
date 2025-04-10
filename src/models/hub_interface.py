import os
from typing import Dict, List, Optional, Union
import yaml
from huggingface_hub import HfApi, ModelFilter, snapshot_download
from transformers import AutoConfig, AutoModel, AutoProcessor

class HuggingFaceHubInterface:
    def __init__(self, cache_dir: str = "./.model_cache"):
        """Initialize the HuggingFace Hub interface.
        
        Args:
            cache_dir: Directory to store downloaded models
        """
        self.cache_dir = cache_dir
        self.api = HfApi()
        self.task_to_best_models = self._load_best_models()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_best_models(self) -> Dict[str, List[str]]:
        """Load the best models for each task from config."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  "config", "default_models.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configurations if file doesn't exist
            return {
                "text-generation": ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2"],
                "image-generation": ["stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5"],
                "text-classification": ["facebook/bart-large-mnli", "roberta-large-mnli"],
                "translation": ["facebook/mbart-large-50-many-to-many-mmt", "t5-base"],
                "summarization": ["facebook/bart-large-cnn", "t5-base"],
                "question-answering": ["deepset/roberta-base-squad2", "distilbert-base-cased-distilled-squad"],
                "image-classification": ["google/vit-base-patch16-224", "microsoft/resnet-50"],
                "speech-recognition": ["facebook/wav2vec2-base-960h", "openai/whisper-small"],
            }
    
    def get_best_model_for_task(self, task: str) -> str:
        """Get the best model for a specific task.
        
        Args:
            task: The task to get the best model for
            
        Returns:
            The model ID for the best model for the task
        """
        if task in self.task_to_best_models:
            return self.task_to_best_models[task][0]  # Return the first (best) model
        else:
            raise ValueError(f"No models found for task: {task}")
    
    def search_models(self, task: str, limit: int = 10) -> List[Dict]:
        """Search for models suitable for a specific task.
        
        Args:
            task: The task to search models for
            limit: Maximum number of models to return
            
        Returns:
            List of model information dictionaries
        """
        models = self.api.list_models(
            filter=ModelFilter(task=task),
            limit=limit,
            sort="downloads",
            direction=-1
        )
        return [{"id": model.id, "downloads": model.downloads} for model in models]
    
    def download_model(self, model_id: str) -> str:
        """Download a model from HuggingFace Hub.
        
        Args:
            model_id: The model ID to download
            
        Returns:
            Path to the downloaded model
        """
        return snapshot_download(
            repo_id=model_id,
            cache_dir=self.cache_dir,
            local_files_only=False
        )
    
    def load_model_for_task(self, task: str, model_id: Optional[str] = None) -> Dict:
        """Load a model for a specific task.
        
        Args:
            task: The task to load a model for
            model_id: Specific model ID to load (if None, uses the best model)
            
        Returns:
            Dictionary with model, processor, and config
        """
        if model_id is None:
            model_id = self.get_best_model_for_task(task)
        
        model_path = self.download_model(model_id)
        
        # Get appropriate model and processor based on task
        if task == "image-generation":
            # For diffusion models, we'll need special handling
            from diffusers import StableDiffusionPipeline
            import torch
            
            model = StableDiffusionPipeline.from_pretrained(
                model_path, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                model = model.to("cuda")
            processor = None
        else:
            # For transformer models
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            processor = AutoProcessor.from_pretrained(model_path, config=config)
        
        return {
            "model": model,
            "processor": processor,
            "config": config if "config" in locals() else None,
            "model_id": model_id
        }
