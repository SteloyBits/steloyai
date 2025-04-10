from typing import Dict, List, Optional, Union, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextClassifier:
    def __init__(self, hub_interface):
        """Initialize the text classifier.
        
        Args:
            hub_interface: The HuggingFace Hub interface
        """
        self.hub = hub_interface
        self.loaded_models = {}  # Cache for loaded models
        
    def _load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a text classification model.
        
        Args:
            model_id: The model ID to load
            
        Returns:
            Dictionary with model and tokenizer
        """
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        model_path = self.hub.download_model(model_id)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Cache the model
        self.loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "id2label": model.config.id2label if hasattr(model.config, "id2label") else None
        }
        
        return self.loaded_models[model_id]
    
    def classify(self, 
                text: str, 
                model_id: Optional[str] = None,
                multi_label: bool = False) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Classify text into predefined categories.
        
        Args:
            text: The text to classify
            model_id: The model ID to use (if None, uses the best model)
            multi_label: Whether this is a multi-label classification task
            
        Returns:
            Dictionary of label:confidence pairs or list of dictionaries for multi-label
        """
        if model_id is None:
            model_id = self.hub.get_best_model_for_task("text-classification")
        
        model_dict = self._load_model(model_id)
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        id2label = model_dict["id2label"]
        
        # Prepare the input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Process results based on whether it's multi-label or not
        if multi_label:
            # For multi-label, apply sigmoid to each logit
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            results = []
            
            # Format the results
            for i, prob in enumerate(probs):
                label = id2label[i] if id2label else f"LABEL_{i}"
                results.append({"label": label, "score": float(prob)})
            
            # Sort by probability
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            return results
        else:
            # For single-label, apply softmax across all logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
            results = {}
            
            # Format the results
            for i, prob in enumerate(probs):
                label = id2label[i] if id2label else f"LABEL_{i}"
                results[label] = float(prob)
            
            # Sort by probability
            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
            return results
    
    def classify_batch(self,
                      texts: List[str],
                      model_id: Optional[str] = None,
                      multi_label: bool = False) -> List[Union[Dict[str, float], List[Dict[str, float]]]]:
        """Classify a batch of texts.
        
        Args:
            texts: List of texts to classify
            model_id: The model ID to use
            multi_label: Whether this is a multi-label classification task
            
        Returns:
            List of classification results
        """
        if model_id is None:
            model_id = self.hub.get_best_model_for_task("text-classification")
        
        model_dict = self._load_model(model_id)
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        id2label = model_dict["id2label"]
        
        # Prepare batch inputs
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        batch_results = []
        
        for i in range(len(texts)):
            if multi_label:
                # For multi-label, apply sigmoid to each logit
                probs = torch.sigmoid(logits[i]).cpu().numpy()
                results = []
                
                # Format the results
                for j, prob in enumerate(probs):
                    label = id2label[j] if id2label else f"LABEL_{j}"
                    results.append({"label": label, "score": float(prob)})
                
                # Sort by probability
                results = sorted(results, key=lambda x: x["score"], reverse=True)
                batch_results.append(results)
            else:
                # For single-label, apply softmax across all logits
                probs = torch.nn.functional.softmax(logits[i], dim=-1).cpu().numpy()
                results = {}
                
                # Format the results
                for j, prob in enumerate(probs):
                    label = id2label[j] if id2label else f"LABEL_{j}"
                    results[label] = float(prob)
                
                # Sort by probability
                results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
                batch_results.append(results)
                
        return batch_results
    
    def get_supported_labels(self, model_id: Optional[str] = None) -> List[str]:
        """Get the labels supported by the classification model.
        
        Args:
            model_id: The model ID to use
            
        Returns:
            List of label names supported by the model
        """
        if model_id is None:
            model_id = self.hub.get_best_model_for_task("text-classification")
            
        model_dict = self._load_model(model_id)
        id2label = model_dict["id2label"]
        
        if id2label:
            return list(id2label.values())
        else:
            return []
