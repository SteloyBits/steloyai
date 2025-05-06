import os
import yaml
from typing import Any, Dict, Optional

def load_config(config_path: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        default_config: Default configuration to use if file doesn't exist
        
    Returns:
        Dictionary containing the configuration
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif default_config is not None:
        # If file doesn't exist and default config is provided, save and return it
        save_config(config_path, default_config)
        return default_config
    else:
        raise FileNotFoundError(f"Config file not found at {config_path} and no default config provided")

def save_config(config_path: str, config: Dict[str, Any]) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config_path: Path where the configuration should be saved
        config: Configuration dictionary to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False) 