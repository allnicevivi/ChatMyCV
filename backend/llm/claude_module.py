import os
from typing import Optional, List, Union

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from .base import LLM

load_dotenv()


class ClaudeLLM(LLM):
    """Claude (Anthropic) LLM implementation."""

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

    def _create_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def chat(
        self,
        prompt: str = "",
        system_prompt: str = "",
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        model: str = "",
        **kwargs
    ) -> Optional[str]:
        # Claude uses a different message format - system is separate
        system = ""
        claude_messages = []

        if messages:
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    claude_messages.append(msg)
        else:
            system = system_prompt
            if prompt:
                claude_messages.append({"role": "user", "content": prompt})

        response = await self._client.messages.create(
            model=model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=self.max_tokens or 4096,
            system=system,
            messages=claude_messages,
            temperature=temperature or self.temperature,
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    async def stream(
        self,
        prompt: str = "",
        system_prompt: str = "",
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        model: str = "",
        **kwargs
    ):
        # Claude uses a different message format - system is separate
        system = ""
        claude_messages = []

        if messages:
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    claude_messages.append(msg)
        else:
            system = system_prompt
            if prompt:
                claude_messages.append({"role": "user", "content": prompt})

        async with self._client.messages.stream(
            model=model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=self.max_tokens or 4096,
            system=system,
            messages=claude_messages,
            temperature=temperature or self.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(
        self,
        input_texts: Union[List[str], str],
        **kwargs
    ) -> List[List[float]]:
        """Claude does not support embedding API."""
        raise NotImplementedError(
            "Claude (Anthropic) does not provide an embedding API. "
            "Please use a different provider for embeddings (azure, openai, or gemini)."
        )
