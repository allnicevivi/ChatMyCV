from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Union

class LLM(ABC):

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        provider: Optional[str] = None,
        **kwargs,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = provider
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def _create_client(self):
        pass

    @abstractmethod
    async def chat(
        self,
        prompt: str = "",
        system_prompt: str = "",
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str = "",
        system_prompt: str = "",
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Iterator[str]:
        pass

    @abstractmethod
    async def embed(
        self,
        input_texts: Union[List[str], str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for input texts."""
        pass