import pytest
import torch
from unittest.mock import Mock, patch
from src.tasks.text_generation import summarize_text, analyze_sentiment, extract_entities

@pytest.fixture
def mock_model():
    model = Mock()
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.convert_ids_to_tokens.return_value = ["This", "is", "a", "test"]
    return tokenizer

def test_summarize_text():
    # Mock the model and tokenizer
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('transformers.AutoModelForSeq2SeqLM.from_pretrained') as mock_model:
        
        # Setup mock returns
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # Mock the generate method
        mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value.decode.return_value = "This is a summary"
        
        # Test the function
        result = summarize_text("This is a long text that needs to be summarized", "test-model")
        
        assert isinstance(result, str)
        assert result == "This is a summary"
        mock_tokenizer.assert_called_once_with("test-model")
        mock_model.assert_called_once_with("test-model")

def test_analyze_sentiment(mock_model):
    # Mock the model and tokenizer
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model):
        
        # Setup mock returns
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # Mock the model output
        mock_model.return_value.return_value.logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        
        # Test the function
        result = analyze_sentiment("This is a positive text", "test-model")
        
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert "details" in result
        assert result["sentiment"] in ["NEGATIVE", "POSITIVE"]

def test_extract_entities(mock_model, mock_tokenizer):
    # Mock the model and tokenizer
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('transformers.AutoModelForTokenClassification.from_pretrained', return_value=mock_model):
        
        # Setup mock returns
        mock_model.return_value = Mock()
        mock_model.return_value.config.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        
        # Mock the model output
        mock_model.return_value.return_value.logits = torch.tensor([[[0.1, 0.8, 0.1]]])
        
        # Test the function
        result = extract_entities("John Smith is a person", "test-model")
        
        assert isinstance(result, list)
        if result:  # If any entities were found
            assert isinstance(result[0], dict)
            assert "text" in result[0]
            assert "type" in result[0]
            assert "confidence" in result[0]

def test_error_handling():
    # Test with invalid model
    with pytest.raises(Exception):
        summarize_text("test", "invalid-model")
    
    with pytest.raises(Exception):
        analyze_sentiment("test", "invalid-model")
    
    with pytest.raises(Exception):
        extract_entities("test", "invalid-model")

def test_device_handling():
    # Test CPU fallback when CUDA is not available
    with patch('torch.cuda.is_available', return_value=False):
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModelForSeq2SeqLM.from_pretrained') as mock_model:
            
            mock_tokenizer.return_value = Mock()
            mock_model.return_value = Mock()
            mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3]])
            mock_tokenizer.return_value.decode.return_value = "Test summary"
            
            result = summarize_text("test", "test-model")
            assert isinstance(result, str) 