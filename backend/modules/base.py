from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseLLMProvider(ABC):
    """Base class for LLM and embedding providers."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
    
    @abstractmethod
    def get_chat_completion_with_usage(
        self,
        system_prompt: str = "You are a helpful assistant",
        prompt: str = "",
        messages: List[Dict[str, str]] = [],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Get a chat completion from the LLM with usage.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Returns:
            The generated response text
        """
        pass

    def get_chat_completion(
        self,
        system_prompt: str = "You are a helpful assistant",
        prompt: str = "",
        messages: List[Dict[str, str]] = [],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Get a chat completion from the LLM.
        """
        response = self.get_chat_completion_with_usage(system_prompt, prompt, messages, model, temperature, max_tokens, **kwargs)
        return response["content"]

    @abstractmethod
    def stream_chat_completion(
        self,
        model: str,
        system_prompt: str = "You are a helpful assistant",
        prompt: str = "",
        messages: List[Dict[str, str]] = [],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Stream chat completion from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Yields:
            Chunks of the generated response
        """
        pass

    @abstractmethod
    def get_embeddings_with_usage(self, texts: List[str]|str, model: Optional[str] = None, **kwargs) -> dict[str, Any]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            model: Optional model name to use
            
        Returns:
            List of embedding vectors (list of floats)
        """
        pass
    
    def get_embeddings(
        self, 
        texts: List[str]|str, 
        model: Optional[str] = None, 
        **kwargs
    ) -> List[List[float]]:
        """
        Generate a single embedding for a text.
        
        Args:
            text: Text to embed
            model: Optional model name to use
            
        Returns:
            List of embedding vectors (list of floats)
        """
        response = self.get_embeddings_with_usage([texts] if isinstance(texts, str) else texts, model, **kwargs)

        return response["embeddings"]
        

    def get_embedding(
        self, 
        text: str, 
        model: Optional[str] = None, 
        **kwargs
    ) -> List[List[float]]:
        """
        Generate a single embedding for a text.
        
        Args:
            text: Text to embed
            model: Optional model name to use
            
        Returns:
            List of embedding vectors (list of floats)
        """
        response = self.get_embeddings_with_usage([text], model, **kwargs)

        return response["embeddings"][0]
