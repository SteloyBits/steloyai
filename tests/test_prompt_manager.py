import pytest
from datetime import datetime
import json
import os
from src.utils.prompt_manager import PromptManager, Message

@pytest.fixture
def temp_history_file(tmp_path):
    """Create a temporary history file."""
    return str(tmp_path / "test_history.json")

@pytest.fixture
def prompt_manager(temp_history_file):
    """Create a PromptManager instance."""
    return PromptManager(max_history=5, history_file=temp_history_file)

def test_message_creation():
    """Test Message class creation and conversion."""
    msg = Message(role="user", content="test message")
    assert msg.role == "user"
    assert msg.content == "test message"
    assert isinstance(msg.timestamp, datetime)

def test_message_serialization():
    """Test Message serialization to and from dictionary."""
    original = Message(role="user", content="test message")
    data = original.to_dict()
    restored = Message.from_dict(data)
    
    assert restored.role == original.role
    assert restored.content == original.content
    assert isinstance(restored.timestamp, datetime)

def test_system_prompts(prompt_manager):
    """Test system prompt retrieval."""
    # Test default prompt
    default_prompt = prompt_manager.get_system_prompt()
    assert isinstance(default_prompt, str)
    assert len(default_prompt) > 0
    
    # Test Einstein prompt
    einstein_prompt = prompt_manager.get_system_prompt("einstein")
    assert isinstance(einstein_prompt, str)
    assert "Einstein" in einstein_prompt
    
    # Test invalid style
    invalid_prompt = prompt_manager.get_system_prompt("invalid_style")
    assert invalid_prompt == default_prompt

def test_message_history(prompt_manager):
    """Test message history management."""
    # Add messages
    prompt_manager.add_message("user", "Hello")
    prompt_manager.add_message("assistant", "Hi there")
    
    # Check history
    assert len(prompt_manager.conversation_history) == 2
    assert prompt_manager.conversation_history[0].content == "Hello"
    assert prompt_manager.conversation_history[1].content == "Hi there"

def test_history_limit(prompt_manager):
    """Test history size limit."""
    # Add more messages than the limit
    for i in range(10):
        prompt_manager.add_message("user", f"Message {i}")
    
    # Check that only the most recent messages are kept
    assert len(prompt_manager.conversation_history) == 5
    assert prompt_manager.conversation_history[0].content == "Message 5"

def test_conversation_context(prompt_manager):
    """Test conversation context generation."""
    # Add some messages
    prompt_manager.add_message("user", "Hello")
    prompt_manager.add_message("assistant", "Hi there")
    
    # Get context
    context = prompt_manager.get_conversation_context("einstein")
    
    # Check context structure
    assert len(context) == 3  # System prompt + 2 messages
    assert context[0]["role"] == "system"
    assert "Einstein" in context[0]["content"]
    assert context[1]["role"] == "user"
    assert context[2]["role"] == "assistant"

def test_history_persistence(temp_history_file, prompt_manager):
    """Test history persistence to file."""
    # Add messages
    prompt_manager.add_message("user", "Hello")
    prompt_manager.add_message("assistant", "Hi there")
    
    # Create new manager with same file
    new_manager = PromptManager(history_file=temp_history_file)
    
    # Check that history was loaded
    assert len(new_manager.conversation_history) == 2
    assert new_manager.conversation_history[0].content == "Hello"
    assert new_manager.conversation_history[1].content == "Hi there"

def test_clear_history(prompt_manager):
    """Test history clearing."""
    # Add messages
    prompt_manager.add_message("user", "Hello")
    prompt_manager.add_message("assistant", "Hi there")
    
    # Clear history
    prompt_manager.clear_history()
    
    # Check that history is empty
    assert len(prompt_manager.conversation_history) == 0 