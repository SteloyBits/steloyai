from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
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
        
        # Set device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        try:
            # First try with device_map
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto"
            )
        except RuntimeError:
            # Fallback to manual device placement
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype
            ).to(device)
        
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

def summarize_text(text: str, model: str, max_length: int = 100) -> str:
    """Summarize the given text using a specified model.
    
    Args:
        text: The text to summarize
        model: The model ID to use for summarization
        max_length: Maximum length of the summary
        
    Returns:
        Generated summary
    """
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode and return
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def analyze_sentiment(text: str, model: str) -> Dict[str, Any]:
    """Analyze the sentiment of the given text.
    
    Args:
        text: The text to analyze
        model: The model ID to use for sentiment analysis
        
    Returns:
        Dictionary containing sentiment analysis results
    """
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get sentiment labels
    id2label = model.config.id2label
    sentiment = id2label[probs.argmax().item()]
    confidence = probs.max().item()
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "details": {
            label: float(prob)
            for label, prob in zip(id2label.values(), probs[0])
        }
    }

def extract_entities(text: str, model: str) -> List[Dict[str, Any]]:
    """Extract named entities from the given text.
    
    Args:
        text: The text to analyze
        model: The model ID to use for entity extraction
        
    Returns:
        List of dictionaries containing entity information
    """
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForTokenClassification.from_pretrained(model)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get entity labels
    id2label = model.config.id2label
    
    # Process predictions
    entities = []
    current_entity = None
    
    for i, (token, prob) in enumerate(zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), probs[0])):
        if token.startswith("##"):
            if current_entity:
                current_entity["text"] += token[2:]
        else:
            label_id = prob.argmax().item()
            label = id2label[label_id]
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "type": label[2:],
                    "confidence": float(prob[label_id])
                }
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += " " + token
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities
