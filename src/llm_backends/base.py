from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any


class LLMBackend(ABC):
    """
    Abstract base class that every LLM provider implementation must follow.
    This ensures consistent interface & metrics format across providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_chain_of_thought: bool = False,
        **kwargs: Any
    ) -> Tuple[str, Dict]:
        """
        Every backend must implement this method.

        Returns:
            (response_text: str, metrics: Dict)
        """
        pass


    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of the provider (ollama, openai, mock, bedrock, ...)"""
        pass