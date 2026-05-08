"""
base_model.py
-------------
Abstract base class for all model backends.
Enforces a consistent interface across vLLM and HuggingFace backends.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract model interface. All backends must implement generate().

    Attributes:
        model_id : short identifier used in result filenames and tables
                   e.g. 'qwen2.5-7b', 'mistral-7b', 'llama3.1-8b'
    """

    def __init__(self, model_id: str, max_new_tokens: int = 1024):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        Returns the raw string output from the model.
        """
        ...

    @abstractmethod
    def load(self):
        """Load the model into memory (called once before evaluation)."""
        ...

    @abstractmethod
    def unload(self):
        """Release model from memory."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"
