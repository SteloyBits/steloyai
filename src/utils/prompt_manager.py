from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'system', 'user', or 'assistant'
    content: str
    timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create message from dictionary format."""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class PromptManager:
    """Manages system prompts and conversation history."""
    
    def __init__(self, max_history: int = 10, history_file: Optional[str] = None):
        """Initialize the prompt manager.
        
        Args:
            max_history: Maximum number of messages to keep in history
            history_file: Optional file path to persist conversation history
        """
        self.max_history = max_history
        self.history_file = history_file
        self.conversation_history: List[Message] = []
        self.system_prompts = {
            'einstein': """You are Albert Einstein, the renowned physicist and scholar. 
            You possess deep knowledge of physics, mathematics, and philosophy. 
            You communicate with clarity and wisdom, often using thought experiments 
            and analogies to explain complex concepts. You maintain a calm, 
            thoughtful demeanor and are known for your humility despite your 
            extraordinary intellect. When responding, you:
            1. Draw from your vast knowledge of science and philosophy
            2. Use clear, accessible language
            3. Provide thoughtful, well-reasoned responses
            4. Share relevant insights from your own experiences
            5. Maintain a warm, engaging tone""",
            
            'scholar': """You are a distinguished scholar with expertise across multiple 
            disciplines. You possess deep knowledge of literature, science, history, 
            and philosophy. You communicate with eloquence and precision, drawing 
            from a vast repository of knowledge. When responding, you:
            1. Provide well-researched, accurate information
            2. Use clear, academic language
            3. Support arguments with evidence
            4. Consider multiple perspectives
            5. Maintain intellectual rigor""",
            
            'default': """You are a helpful AI assistant with expertise in various domains. 
            You provide clear, accurate, and relevant responses while maintaining 
            a professional and engaging tone."""
        }
        
        if history_file and os.path.exists(history_file):
            self.load_history()

    def get_system_prompt(self, style: str = 'default') -> str:
        """Get the system prompt for the specified style.
        
        Args:
            style: The style of system prompt to use
            
        Returns:
            The system prompt string
        """
        return self.system_prompts.get(style.lower(), self.system_prompts['default'])

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender
            content: The message content
        """
        message = Message(role=role, content=content)
        self.conversation_history.append(message)
        
        # Maintain history size limit
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Save history if file is specified
        if self.history_file:
            self.save_history()

    def get_conversation_context(self, style: str = 'default') -> List[Dict]:
        """Get the formatted conversation context for the model.
        
        Args:
            style: The style of system prompt to use
            
        Returns:
            List of message dictionaries in the format expected by the model
        """
        context = [{'role': 'system', 'content': self.get_system_prompt(style)}]
        context.extend([msg.to_dict() for msg in self.conversation_history])
        return context

    def save_history(self) -> None:
        """Save conversation history to file."""
        if not self.history_file:
            return
            
        history_data = [msg.to_dict() for msg in self.conversation_history]
        with open(self.history_file, 'w') as f:
            json.dump(history_data, f, indent=2)

    def load_history(self) -> None:
        """Load conversation history from file."""
        if not self.history_file or not os.path.exists(self.history_file):
            return
            
        with open(self.history_file, 'r') as f:
            history_data = json.load(f)
            self.conversation_history = [Message.from_dict(msg) for msg in history_data]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        if self.history_file:
            self.save_history() 