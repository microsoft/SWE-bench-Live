"""
Common action parsing utilities for agent interactions.
"""
import re
from typing import Optional, Any
from abc import ABC, abstractmethod


class ActionParser(ABC):
    """Base class for parsing LLM responses into structured actions."""
    
    @abstractmethod
    def parse(self, response: str) -> Optional[Any]:
        """Parse response string into action object."""
        pass
    
    @staticmethod
    def extract_tag_content(response: str, tag: str) -> Optional[str]:
        """Extract content between XML-style tags."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1) if match else None
    
    @staticmethod
    def clean_response(response: str) -> str:
        """Remove reasoning tags from response if present."""
        if "<think>" in response:
            return response.split("</think>")[1]
        return response