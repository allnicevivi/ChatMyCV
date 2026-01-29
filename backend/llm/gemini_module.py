import os
import time
from typing import Optional, List, Union

import google.generativeai as genai
from dotenv import load_dotenv

from .base import LLM

load_dotenv()


class GeminiLLM(LLM):
    """Google Gemini LLM implementation."""

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

    def _create_client(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel(
            os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")
        )

    def _convert_messages_to_gemini_format(
        self,
        messages: List[dict],
        system_prompt: str = ""
    ) -> tuple:
        """Convert OpenAI-style messages to Gemini format."""
        history = []
        system_instruction = system_prompt

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        return history, system_instruction

    async def chat(
        self,
        prompt: str = "",
        system_prompt: str = "",
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        model: str = "",
        **kwargs
    ) -> Optional[str]:
        if model:
            client = genai.GenerativeModel(model)
        else:
            client = self._client

        if messages:
            history, system_instruction = self._convert_messages_to_gemini_format(
                messages, system_prompt
            )
        else:
            history = []
            system_instruction = system_prompt
            if prompt:
                history.append({"role": "user", "parts": [prompt]})

        # Create chat with system instruction
        if system_instruction:
            client = genai.GenerativeModel(
                model or os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash"),
                system_instruction=system_instruction
            )

        chat = client.start_chat(history=history[:-1] if history else [])

        generation_config = genai.types.GenerationConfig(
            temperature=temperature or self.temperature,
            max_output_tokens=self.max_tokens,
        )

        # Get the last user message
        last_message = history[-1]["parts"][0] if history else prompt

        response = await chat.send_message_async(
            last_message,
            generation_config=generation_config
        )

        return {
            "content": response.text,
            "usage": {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
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
        if messages:
            history, system_instruction = self._convert_messages_to_gemini_format(
                messages, system_prompt
            )
        else:
            history = []
            system_instruction = system_prompt
            if prompt:
                history.append({"role": "user", "parts": [prompt]})

        # Create model with system instruction
        client = genai.GenerativeModel(
            model or os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash"),
            system_instruction=system_instruction if system_instruction else None
        )

        chat = client.start_chat(history=history[:-1] if history else [])

        generation_config = genai.types.GenerationConfig(
            temperature=temperature or self.temperature,
            max_output_tokens=self.max_tokens,
        )

        # Get the last user message
        last_message = history[-1]["parts"][0] if history else prompt

        response = await chat.send_message_async(
            last_message,
            generation_config=generation_config,
            stream=True
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def embed(
        self,
        input_texts: Union[List[str], str],
        model: str = "",
        **kwargs
    ) -> List[List[float]]:
        t = time.time()

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        embed_model = model or os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

        result = await genai.embed_content_async(
            model=f"models/{embed_model}",
            content=input_texts,
            task_type="retrieval_document"
        )

        print(f"\n(gemini embedding spent {time.time()-t:.3f} sec)")

        # Handle both single and batch embeddings
        if isinstance(result["embedding"][0], list):
            return result["embedding"]
        else:
            return [result["embedding"]]
