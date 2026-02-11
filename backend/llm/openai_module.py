import os
import time
from typing import Optional, List, Union

from openai import AsyncOpenAI
from dotenv import load_dotenv

from .base import LLM

load_dotenv()


class OpenAILLM(LLM):
    """OpenAI LLM implementation."""

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(temperature, max_tokens, timeout, provider, **kwargs)
        self._client = self._create_client()

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def chat(
        self,
        prompt: str = "",
        system_prompt: str = "",
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        model: str = "",
        **kwargs
    ) -> Optional[str]:
        if not messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=model or os.getenv("OPENAI_LLM_MODEL", "gpt-4o"),
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
        )

        return {"content": response.choices[0].message.content, "usage": response.usage}

    async def stream(
        self,
        prompt: str = "",
        system_prompt: str = "",
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        model: str = "",
        **kwargs
    ):
        if not messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=model or os.getenv("OPENAI_LLM_MODEL", "gpt-4o"),
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content

    async def embed(
        self,
        input_texts: Union[List[str], str],
        model: str = "",
        dimensions: int = 1536,
        **kwargs
    ) -> List[List[float]]:
        t = time.time()
        embeddings = await self._client.embeddings.create(
            input=input_texts,
            model=model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            dimensions=dimensions,
        )
        print(f"\n(openai embedding spent {time.time()-t:.3f} sec)")

        return [ele.embedding for ele in embeddings.data]
