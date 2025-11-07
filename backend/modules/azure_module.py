#!/usr/bin/env/python
# -*- coding:utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import os
from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import List, Dict, Optional, Union, Any
from backend.modules.base import BaseLLMProvider
from backend.utils.app_logger import LoggerSetup

logger = LoggerSetup(module_name="AzureModule").logger


class AzureModule(BaseLLMProvider):
    """Azure OpenAI LLM provider implementation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "2024-02-15-preview"
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        base_url = base_url or os.getenv("AZURE_OPENAI_API_BASE")
        super().__init__(api_key, base_url)
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.base_url,
            api_version=self.api_version
        )
    
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
        Get a chat completion from Azure OpenAI with usage.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model deployment name
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Azure OpenAI API parameters
            
        Returns:
            The generated response text
        """
        if not messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

        model = model or os.getenv("AZURE_OPENAI_LLM_ENGINE")
        print(model)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return {
                "content": response.choices[0].message.content.strip(),
                "usage": response.usage.model_dump()
            }
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}", exc_info=True)
            raise
    
    def stream_chat_completion(
        self,
        system_prompt: str = "You are a helpful assistant",
        prompt: str = "",
        messages: List[Dict[str, str]] = [],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Stream chat completion from Azure OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model deployment name
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Azure OpenAI API parameters
            
        Yields:
            Chunks of the generated response
        """
        if not messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}", exc_info=True)
            raise

    def get_embeddings_with_usage(
        self, 
        texts: List[str]|str, 
        model: Optional[str] = None, 
        **kwargs
    ) -> dict[str, Any]:
        """
        Generate embeddings for multiple texts using Azure OpenAI.
        
        Args:
            texts: List of texts to embed
            model: Model deployment name (default: 'text-embedding-3-small')
            
        Returns:
            List of embedding vectors
        """
        model = model or os.getenv("AZURE_OPENAI_EMBED_ENGINE")
        dim = int(kwargs.get("dim") or os.getenv("EMBED_DIM") or 768)
        timeout = int(kwargs.get("timeout") or os.getenv("EMBED_TIMEOUT") or 3)
        try:
            response = self.client.embeddings.create(
                input=[texts] if isinstance(texts, str) else texts,
                model=model,
                dimensions=dim,
                timeout=timeout,
            )
            return {
                "embeddings": [item.embedding for item in response.data],
                "usage": {"embedding_tokens": response.usage.total_tokens}
            }
        except Exception as e:
            logger.error(f"Azure OpenAI embeddings API error: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    azure_client = AzureModule()
    
    # embed = azure_client.get_embedding("hi")
    # print(len(embed))
    # print(embed)

    res = azure_client.get_chat_completion(
        system_prompt="You are a helpful assistant",
        prompt="What is the capital of France?",
    )

    print(res)