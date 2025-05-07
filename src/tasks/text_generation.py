from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.prompt_manager import PromptManager

class TextGenerator:
    def __init__(self, hub_interface, prompt_manager: Optional[PromptManager] = None):
        """Initialize the text generator.
        
        Args:
            hub_interface: The HuggingFace Hub interface
            prompt_manager: Optional PromptManager instance for handling prompts and history
        """
        self.hub = hub_interface
        self.loaded_models = {}  # Cache for loaded models
        self.prompt_manager = prompt_manager or PromptManager()
        
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
                temperature: float = 0.7,
                style: str = 'default') -> str:
        """Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model_id: The model ID to use (if None, uses the best model)
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            style: The style of system prompt to use
            
        Returns:
            Generated text
        """
        if model_id is None:
            model_id = self.hub.get_best_model_for_task("text-generation")
        
        # Add user message to history
        self.prompt_manager.add_message('user', prompt)
        
        # Get conversation context
        context = self.prompt_manager.get_conversation_context(style)
        
        model_dict = self._load_model(model_id)
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        
        # Prepare the input with conversation context
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in context
        ])
        
        # Add a clear instruction for the model
        conversation_text += "\nASSISTANT: I will now provide a direct answer to the question."
        
        inputs = tokenizer(conversation_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate with more focused parameters
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.92,  # Slightly reduced for more focused responses
                top_k=40,    # Reduced for more focused responses
                num_return_sequences=1,
                repetition_penalty=1.2,  # Added to reduce repetition
                length_penalty=1.0,      # Added to encourage complete answers
                no_repeat_ngram_size=3   # Added to reduce repetition
            )
        
        # Decode and return
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean up the response
        if generated_text.startswith(conversation_text):
            generated_text = generated_text[len(conversation_text):].strip()
        
        # Remove any remaining role prefixes
        for prefix in ["ASSISTANT:", "Answer:", "Question:"]:
            if generated_text.startswith(prefix):
                generated_text = generated_text[len(prefix):].strip()
        
        # Add assistant's response to history
        self.prompt_manager.add_message('assistant', generated_text)
            
        return generated_text
