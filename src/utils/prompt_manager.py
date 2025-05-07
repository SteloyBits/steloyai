from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
import uuid

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

@dataclass
class ChatSession:
    """Represents a chat session with its history and metadata."""
    id: str
    title: str
    messages: List[Message]
    created_at: datetime = datetime.now()
    last_updated: datetime = datetime.now()

    def to_dict(self) -> Dict:
        """Convert chat session to dictionary format."""
        return {
            'id': self.id,
            'title': self.title,
            'messages': [msg.to_dict() for msg in self.messages],
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        """Create chat session from dictionary format."""
        return cls(
            id=data['id'],
            title=data['title'],
            messages=[Message.from_dict(msg) for msg in data['messages']],
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )

class PromptManager:
    """Manages system prompts and conversation history."""
    
    def __init__(self, max_history: int = 10, history_dir: Optional[str] = None):
        """Initialize the prompt manager.
        
        Args:
            max_history: Maximum number of messages to keep in history
            history_dir: Optional directory to store chat histories
        """
        self.max_history = max_history
        self.history_dir = history_dir
        self.current_session: Optional[ChatSession] = None
        self.sessions: Dict[str, ChatSession] = {}
        
        # Create history directory if it doesn't exist
        if history_dir and not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        # Load existing sessions if directory exists
        if history_dir and os.path.exists(history_dir):
            self.load_sessions()
            
        self.system_prompts = {
            'einstein': """You are Albert Einstein, the renowned physicist and scholar. 
            Your task is to provide direct, insightful answers to questions while maintaining 
            your unique perspective and expertise. When responding:

            1. ALWAYS provide a direct answer to the question first
            2. Then, if relevant, add your unique Einstein perspective
            3. Use clear, accessible language
            4. Draw from your knowledge of physics, mathematics, and philosophy
            5. Use thought experiments or analogies when helpful
            6. Maintain a warm, engaging tone

            Remember: Your primary goal is to answer the question directly and completely, 
            not just to restate or rephrase it. If you're unsure about something, 
            acknowledge it while still providing the best answer you can.""",
            
            'scholar': """You are a distinguished scholar with expertise across multiple 
            disciplines. Your task is to provide comprehensive, well-reasoned answers 
            to questions. When responding:

            1. ALWAYS provide a direct answer to the question first
            2. Support your answer with relevant evidence or examples
            3. Use clear, academic language
            4. Consider multiple perspectives
            5. Maintain intellectual rigor
            6. Cite relevant sources or concepts when appropriate

            Remember: Your primary goal is to answer the question directly and completely, 
            not just to restate or rephrase it. If you're unsure about something, 
            acknowledge it while still providing the best answer you can.""",
            
            'default': """You are a helpful AI assistant with expertise in various domains. 
            Your task is to provide clear, accurate, and relevant answers to questions. 
            When responding:

            1. ALWAYS provide a direct answer to the question first
            2. Be concise but comprehensive
            3. Use clear, accessible language
            4. Provide relevant examples or explanations when helpful
            5. Maintain a professional and engaging tone

            Remember: Your primary goal is to answer the question directly and completely, 
            not just to restate or rephrase it. If you're unsure about something, 
            acknowledge it while still providing the best answer you can."""
        }

    def create_new_session(self, title: Optional[str] = None) -> str:
        """Create a new chat session.
        
        Args:
            title: Optional title for the session
            
        Returns:
            The ID of the new session
        """
        session_id = str(uuid.uuid4())
        title = title or f"Chat {len(self.sessions) + 1}"
        
        self.current_session = ChatSession(
            id=session_id,
            title=title,
            messages=[]
        )
        self.sessions[session_id] = self.current_session
        
        if self.history_dir:
            self.save_sessions()
            
        return session_id

    def switch_session(self, session_id: str) -> bool:
        """Switch to an existing chat session.
        
        Args:
            session_id: The ID of the session to switch to
            
        Returns:
            True if successful, False if session not found
        """
        if session_id in self.sessions:
            self.current_session = self.sessions[session_id]
            return True
        return False

    def get_system_prompt(self, style: str = 'default') -> str:
        """Get the system prompt for the specified style."""
        return self.system_prompts.get(style.lower(), self.system_prompts['default'])

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the current session."""
        if not self.current_session:
            self.create_new_session()
            
        message = Message(role=role, content=content)
        self.current_session.messages.append(message)
        self.current_session.last_updated = datetime.now()
        
        # Maintain history size limit
        if len(self.current_session.messages) > self.max_history:
            self.current_session.messages = self.current_session.messages[-self.max_history:]
        
        if self.history_dir:
            self.save_sessions()

    def get_conversation_context(self, style: str = 'default') -> List[Dict]:
        """Get the formatted conversation context for the model."""
        if not self.current_session:
            return [{'role': 'system', 'content': self.get_system_prompt(style)}]
            
        context = [{'role': 'system', 'content': self.get_system_prompt(style)}]
        
        for msg in self.current_session.messages:
            if msg.role == 'assistant':
                content = f"Answer: {msg.content}"
            else:
                content = f"Question: {msg.content}"
            context.append({'role': msg.role, 'content': content})
        
        return context

    def get_session_list(self) -> List[Dict]:
        """Get a list of all chat sessions."""
        return [
            {
                'id': session.id,
                'title': session.title,
                'created_at': session.created_at,
                'last_updated': session.last_updated,
                'message_count': len(session.messages)
            }
            for session in self.sessions.values()
        ]

    def save_sessions(self) -> None:
        """Save all chat sessions to disk."""
        if not self.history_dir:
            return
            
        sessions_data = {
            session_id: session.to_dict()
            for session_id, session in self.sessions.items()
        }
        
        with open(os.path.join(self.history_dir, 'chat_sessions.json'), 'w') as f:
            json.dump(sessions_data, f, indent=2)

    def load_sessions(self) -> None:
        """Load chat sessions from disk."""
        if not self.history_dir:
            return
            
        sessions_file = os.path.join(self.history_dir, 'chat_sessions.json')
        if not os.path.exists(sessions_file):
            return
            
        with open(sessions_file, 'r') as f:
            sessions_data = json.load(f)
            self.sessions = {
                session_id: ChatSession.from_dict(session_data)
                for session_id, session_data in sessions_data.items()
            }
            
        # Set current session to most recent if available
        if self.sessions:
            most_recent = max(
                self.sessions.values(),
                key=lambda s: s.last_updated
            )
            self.current_session = most_recent

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session.
        
        Args:
            session_id: The ID of the session to delete
            
        Returns:
            True if successful, False if session not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session and self.current_session.id == session_id:
                self.current_session = None
            if self.history_dir:
                self.save_sessions()
            return True
        return False 