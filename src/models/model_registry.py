import os
import yaml
import json
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
from huggingface_hub import ModelInfo

# TODO: Consider implementing a proper database backend instead of JSON files
# This would improve scalability and concurrent access handling
class ModelRegistry:
    """Registry for managing AI models and their metadata.
    
    This class provides a centralized registry for managing model metadata,
    task mappings, user preferences, and usage statistics. It uses a JSON file
    for persistence but could be extended to use a proper database.
    """
    
    def __init__(self, config_dir: str = "./config", registry_file: str = "model_registry.json"):
        """Initialize the model registry.
        
        Args:
            config_dir: Directory for configuration files
            registry_file: Name of the registry file
        """
        self.config_dir = config_dir
        self.registry_path = os.path.join(config_dir, registry_file)
        self.registry = self._load_registry()
        self.logger = logging.getLogger(__name__)
        
        # Ensure the config directory exists
        os.makedirs(config_dir, exist_ok=True)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry from disk.
        
        Returns:
            Dictionary containing model registry data
            
        TODO: Add proper error handling and recovery mechanisms
        TODO: Consider implementing file locking for concurrent access
        """
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._initialize_registry()
        else:
            return self._initialize_registry()
    
    def _initialize_registry(self) -> Dict[str, Any]:
        """Initialize an empty registry structure.
        
        Returns:
            Empty registry dictionary with default structure
            
        TODO: Consider moving default structure to a configuration file
        """
        registry = {
            "models": {},
            "task_mappings": {},
            "user_preferences": {},
            "usage_statistics": {},
            "last_updated": datetime.now().isoformat()
        }
        
        # Try to load default models from YAML if it exists
        default_models_path = os.path.join(self.config_dir, "default_models.yaml")
        if os.path.exists(default_models_path):
            try:
                with open(default_models_path, 'r') as f:
                    task_mappings = yaml.safe_load(f)
                    registry["task_mappings"] = {
                        task: [{"model_id": model, "priority": idx} 
                               for idx, model in enumerate(models)]
                        for task, models in task_mappings.items()
                    }
            except Exception as e:
                logging.warning(f"Failed to load default models: {e}")
        
        self._save_registry(registry)
        return registry
    
    def _save_registry(self, registry: Optional[Dict[str, Any]] = None) -> None:
        """Save the model registry to disk.
        
        Args:
            registry: Registry data to save, if None uses self.registry
            
        TODO: Implement atomic file writing to prevent corruption
        TODO: Add backup mechanism before saving
        """
        if registry is None:
            registry = self.registry
        
        registry["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, model_id: str, task: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a model in the registry.
        
        Args:
            model_id: HuggingFace model ID
            task: Task the model is used for
            metadata: Additional metadata about the model
            
        TODO: Add validation for model_id and task
        TODO: Consider adding versioning support
        """
        if metadata is None:
            metadata = {}
        
        # Add model to models section if not exists
        if model_id not in self.registry["models"]:
            self.registry["models"][model_id] = {
                "tasks": [task],
                "metadata": metadata,
                "first_registered": datetime.now().isoformat(),
                "last_used": None,
                "usage_count": 0
            }
        else:
            # Update existing model
            if task not in self.registry["models"][model_id]["tasks"]:
                self.registry["models"][model_id]["tasks"].append(task)
            
            # Update metadata with new info
            self.registry["models"][model_id]["metadata"].update(metadata)
        
        # Add model to task mappings if not exists
        if task not in self.registry["task_mappings"]:
            self.registry["task_mappings"][task] = []
        
        # Check if model already in task mappings
        model_exists = False
        for entry in self.registry["task_mappings"][task]:
            if entry["model_id"] == model_id:
                model_exists = True
                break
        
        if not model_exists:
            # Add model to end of task list
            next_priority = len(self.registry["task_mappings"][task])
            self.registry["task_mappings"][task].append({
                "model_id": model_id,
                "priority": next_priority
            })
        
        self._save_registry()
    
    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a registered model.
        
        Args:
            model_id: HuggingFace model ID
            metadata: New metadata to update
            
        TODO: Add validation for metadata structure
        TODO: Consider adding audit logging for metadata changes
        """
        if model_id in self.registry["models"]:
            self.registry["models"][model_id]["metadata"].update(metadata)
            self._save_registry()
        else:
            self.logger.warning(f"Attempted to update metadata for unregistered model: {model_id}")
    
    def register_model_from_info(self, model_info: ModelInfo, task: str) -> None:
        """Register a model using information from HuggingFace Hub.
        
        Args:
            model_info: ModelInfo object from HuggingFace Hub
            task: Task the model is used for
            
        TODO: Add more metadata fields from ModelInfo
        TODO: Consider adding validation for model compatibility with task
        """
        metadata = {
            "description": model_info.description,
            "tags": model_info.tags,
            "pipeline_tag": model_info.pipeline_tag,
            "downloads": model_info.downloads,
            "library_name": model_info.library_name,
            "model_url": f"https://huggingface.co/{model_info.id}"
        }
        
        self.register_model(model_info.id, task, metadata)
    
    def get_models_for_task(self, task: str) -> List[Dict[str, Union[str, int]]]:
        """Get all models registered for a specific task.
        
        Args:
            task: Task to get models for
            
        Returns:
            List of model entries sorted by priority
            
        TODO: Add caching for frequently accessed task lists
        TODO: Consider adding filtering options
        """
        if task in self.registry["task_mappings"]:
            # Sort by priority
            return sorted(self.registry["task_mappings"][task], key=lambda x: x["priority"])
        else:
            return []
    
    def get_best_model_for_task(self, task: str) -> Optional[str]:
        """Get the highest priority model for a task.
        
        Args:
            task: Task to get the best model for
            
        Returns:
            Model ID or None if no models are registered for the task
            
        TODO: Consider adding model performance metrics to influence selection
        TODO: Add support for user-specific model preferences
        """
        models = self.get_models_for_task(task)
        if models:
            return models[0]["model_id"]
        else:
            return None
    
    def set_model_priority(self, task: str, model_id: str, priority: int) -> None:
        """Set the priority of a model for a specific task.
        
        Args:
            task: Task to set priority for
            model_id: Model ID to set priority for
            priority: New priority value (0 = highest priority)
            
        TODO: Add validation for priority values
        TODO: Consider adding priority change history
        """
        if task not in self.registry["task_mappings"]:
            self.logger.warning(f"Task {task} not found in registry")
            return
        
        # Find model in task list
        model_found = False
        for entry in self.registry["task_mappings"][task]:
            if entry["model_id"] == model_id:
                entry["priority"] = priority
                model_found = True
                break
        
        if not model_found:
            self.logger.warning(f"Model {model_id} not found in task {task}")
            return
        
        # Re-sort all entries to ensure priorities are consecutive
        sorted_entries = sorted(self.registry["task_mappings"][task], key=lambda x: x["priority"])
        for i, entry in enumerate(sorted_entries):
            entry["priority"] = i
        
        self.registry["task_mappings"][task] = sorted_entries
        self._save_registry()
    
    def record_model_usage(self, model_id: str) -> None:
        """Record that a model was used.
        
        Args:
            model_id: ID of the model that was used
        """
        if model_id not in self.registry["models"]:
            self.logger.warning(f"Recording usage for unregistered model: {model_id}")
            return
        
        # Update model usage info
        self.registry["models"][model_id]["last_used"] = datetime.now().isoformat()
        self.registry["models"][model_id]["usage_count"] += 1
        
        # Add to usage statistics
        date_key = datetime.now().strftime("%Y-%m-%d")
        if date_key not in self.registry["usage_statistics"]:
            self.registry["usage_statistics"][date_key] = {}
        
        if model_id not in self.registry["usage_statistics"][date_key]:
            self.registry["usage_statistics"][date_key][model_id] = 1
        else:
            self.registry["usage_statistics"][date_key][model_id] += 1
        
        self._save_registry()
    
    def get_task_list(self) -> List[str]:
        """Get a list of all registered tasks.
        
        Returns:
            List of task names
        """
        return list(self.registry["task_mappings"].keys())
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model.
        
        Args:
            model_id: ID of the model to get metadata for
            
        Returns:
            Model metadata or None if model not found
        """
        if model_id in self.registry["models"]:
            return self.registry["models"][model_id]
        else:
            return None
    
    def set_user_preference(self, user_id: str, task: str, model_id: str) -> None:
        """Set a user's preferred model for a task.
        
        Args:
            user_id: ID of the user
            task: Task to set preference for
            model_id: ID of the preferred model
            
        TODO: Add validation for user_id and model_id
        TODO: Consider adding preference expiration
        """
        if user_id not in self.registry["user_preferences"]:
            self.registry["user_preferences"][user_id] = {}
        
        self.registry["user_preferences"][user_id][task] = model_id
        self._save_registry()
    
    def get_user_preference(self, user_id: str, task: str) -> Optional[str]:
        """Get a user's preferred model for a task.
        
        Args:
            user_id: ID of the user
            task: Task to get preference for
            
        Returns:
            Model ID or None if no preference is set
            
        TODO: Add caching for frequently accessed preferences
        TODO: Consider adding preference inheritance from groups
        """
        if (user_id in self.registry["user_preferences"] and 
            task in self.registry["user_preferences"][user_id]):
            return self.registry["user_preferences"][user_id][task]
        else:
            return None
    
    def get_usage_statistics(self, days: Optional[int] = None) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for models.
        
        Args:
            days: Number of days to get statistics for, None for all
            
        Returns:
            Dictionary with usage statistics by date
            
        TODO: Add aggregation functions for statistics
        TODO: Consider adding export functionality for analysis
        """
        if days is None:
            return self.registry["usage_statistics"]
        
        # Filter to last N days
        from datetime import datetime, timedelta
        
        result = {}
        today = datetime.now().date()
        
        for i in range(days):
            date_key = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            if date_key in self.registry["usage_statistics"]:
                result[date_key] = self.registry["usage_statistics"][date_key]
        
        return result
    
    def remove_model(self, model_id: str) -> None:
        """Remove a model from the registry.
        
        Args:
            model_id: ID of the model to remove
        """
        # Remove from models section
        if model_id in self.registry["models"]:
            del self.registry["models"][model_id]
        
        # Remove from task mappings
        for task in self.registry["task_mappings"]:
            self.registry["task_mappings"][task] = [
                entry for entry in self.registry["task_mappings"][task]
                if entry["model_id"] != model_id
            ]
        
        # Remove from user preferences
        for user_id in self.registry["user_preferences"]:
            user_prefs = self.registry["user_preferences"][user_id]
            for task in list(user_prefs.keys()):
                if user_prefs[task] == model_id:
                    del user_prefs[task]
        
        self._save_registry()
