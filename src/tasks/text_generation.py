from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextGenerator:
    def __init__(self, hub_interface):
        """Initialize the text generator.
        
        Args:
            hub_interface: The HuggingFace Hub interface
        """
        self.hub = hub_interface
        self.loaded_models = {}  # Cache for loaded models
        
    def _load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a text generation model.
        
        Args:
            model_id: The model ID to load
            
        Returns:
            Dictionary with model and tokenizer
        """
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        model_path = self.hub.download_model(model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Cache the model
        self.loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer
        }
        
        return self.loaded_models[model_id]
    
    def generate(self, 
                prompt: str, 
                model_id: Optional[str] = None, 
                max_length: int = 256, 
                temperature: float = 0.7) -> str:
        """Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model_id: The model ID to use (if None, uses the best model)
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            
        Returns:
            Generated text
        """
        if model_id is None:
            model_id = self.hub.get_best_model_for_task("text-generation")
        
        model_dict = self._load_model(model_id)
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        
        # Prepare the input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1
            )
        
        # Decode and return
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Some models include the prompt in the output, so we remove it if necessary
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
        return generated_text
