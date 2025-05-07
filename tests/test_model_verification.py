import pytest
from unittest.mock import Mock, patch
from src.utils.model_verification import verify_models_availability, get_verified_config

# Test data
SAMPLE_MODELS = [
    "gpt2",  # Known to exist
    "non-existent-model",  # Known to not exist
    "facebook/opt-125m"  # Known to exist
]

@pytest.fixture
def mock_hf_api():
    """Fixture to create a mock HfApi instance."""
    with patch('src.utils.model_verification.HfApi') as mock_api:
        # Create a mock instance
        mock_instance = Mock()
        mock_api.return_value = mock_instance
        yield mock_instance

def test_verify_models_availability_success(mock_hf_api):
    """Test successful model verification."""
    # Mock successful model info retrieval
    mock_hf_api.model_info.return_value = Mock()
    
    # Test with a single known model
    result = verify_models_availability(["gpt2"])
    assert len(result) == 1
    assert "gpt2" in result
    mock_hf_api.model_info.assert_called_once_with("gpt2")

def test_verify_models_availability_not_found(mock_hf_api):
    """Test model verification when model is not found."""
    # Mock model not found
    mock_hf_api.model_info.return_value = None
    
    result = verify_models_availability(["non-existent-model"])
    assert len(result) == 0
    mock_hf_api.model_info.assert_called_once_with("non-existent-model")

def test_verify_models_availability_error(mock_hf_api):
    """Test model verification when an error occurs."""
    # Mock an exception
    mock_hf_api.model_info.side_effect = Exception("API Error")
    
    result = verify_models_availability(["error-model"])
    assert len(result) == 0
    mock_hf_api.model_info.assert_called_once_with("error-model")

def test_verify_models_availability_mixed(mock_hf_api):
    """Test model verification with a mix of existing and non-existing models."""
    # Mock different responses for different models
    def mock_model_info(model_id):
        if model_id == "gpt2":
            return Mock()
        elif model_id == "non-existent-model":
            return None
        else:
            raise Exception("API Error")
    
    mock_hf_api.model_info.side_effect = mock_model_info
    
    result = verify_models_availability(SAMPLE_MODELS)
    assert len(result) == 1
    assert "gpt2" in result
    assert len(mock_hf_api.model_info.call_args_list) == 3

def test_get_verified_config(mock_hf_api):
    """Test the get_verified_config function."""
    # Mock successful model info retrieval for all models
    mock_hf_api.model_info.return_value = Mock()
    
    config = get_verified_config()
    
    # Verify the structure of the config
    assert isinstance(config, dict)
    assert "text_generation" in config
    assert "text_summarization" in config
    assert "sentiment_analysis" in config
    assert "ner" in config
    assert "image_generation" in config
    assert "image_captioning" in config
    assert "object_detection" in config
    
    # Verify each task has available_models key
    for task in config:
        assert "available_models" in config[task]
        assert isinstance(config[task]["available_models"], list)

def test_get_verified_config_no_models(mock_hf_api):
    """Test get_verified_config when no models are available."""
    # Mock all models as unavailable
    mock_hf_api.model_info.return_value = None
    
    config = get_verified_config()
    
    # Verify all tasks have empty model lists
    for task in config:
        assert config[task]["available_models"] == []

def test_get_verified_config_partial_availability(mock_hf_api):
    """Test get_verified_config with partial model availability."""
    # Mock different responses for different models
    def mock_model_info(model_id):
        if "gpt2" in model_id or "t5-small" in model_id:
            return Mock()
        return None
    
    mock_hf_api.model_info.side_effect = mock_model_info
    
    config = get_verified_config()
    
    # Verify that only some models are available
    assert len(config["text_generation"]["available_models"]) > 0
    assert len(config["text_summarization"]["available_models"]) > 0
    assert len(config["sentiment_analysis"]["available_models"]) == 0 