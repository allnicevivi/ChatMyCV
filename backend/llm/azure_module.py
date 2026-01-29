import sys
sys.path.append("./")

from .base import LLM

from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Optional
import httpx
import asyncio

# nest_asyncio is needed for running async code in environments with existing event loops
# but fails on Streamlit Cloud's event loop type - we must skip importing it entirely
# because even importing nest_asyncio modifies asyncio.run() behavior
import os as _os
_is_streamlit_cloud = _os.path.exists("/mount/src")

_nest_asyncio_applied = False
if not _is_streamlit_cloud:
    try:
        import nest_asyncio
        nest_asyncio.apply()
        _nest_asyncio_applied = True
    except (ValueError, RuntimeError):
        pass

import os
import time
from dotenv import load_dotenv
load_dotenv()

class AzureOpenaiLLM(LLM):
    def __init__(self, temperature: float = 0.7, max_tokens: Optional[int] = None, timeout: Optional[int] = None, provider: Optional[str] = None, **kwargs):
        super().__init__(temperature, max_tokens, timeout, provider, **kwargs)
        self._client = self._create_client()
        self._embed_client = self._create_client()
        # Handle warmup in environments with or without existing event loops
        self._run_warmup()

    def _run_warmup(self):
        """Run warmup safely across different async environments."""
        # Skip warmup on Streamlit Cloud - it causes event loop conflicts
        # because the client binds to one loop during warmup but uses another at runtime
        if _is_streamlit_cloud:
            print("[AzureOpenaiLLM] Warmup skipped on Streamlit Cloud")
            return

        try:
            loop = asyncio.get_running_loop()
            # Already in an async context, schedule the warmup as a task
            loop.create_task(self._warmup_embed_and_chat())
        except RuntimeError:
            # No running event loop
            if _nest_asyncio_applied:
                # nest_asyncio is working, safe to use asyncio.run()
                asyncio.run(self._warmup_embed_and_chat())
            else:
                # Skip warmup if no safe way to run it
                print(f"[AzureOpenaiLLM] Warmup skipped: no event loop available")

    def _create_client(self) -> AsyncAzureOpenAI:
        
        client = AsyncAzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        return client

    async def _warmup_embed_and_chat(self):
        try:
            # Test embedding
            emb_result = await self.embed(["hi"])
            # Test chat
            chat_result = await self.chat(prompt="hi")
            print(f"[AzureOpenaiLLM] Warm-up successfully.")
        except Exception as e:
            print(f"[AzureOpenaiLLM] Warm-up failed: {e}")
    
    async def chat(self, prompt: str = "", system_prompt: str = "", messages: Optional[list[dict[str, str]]] = None, temperature: Optional[float] = None, engine: str="", max_tokens: Optional[int] = None) -> Optional[str]:
        
        if not messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})
        
        response = await self._client.chat.completions.create(
            model=engine or os.getenv("AZURE_OPENAI_LLM_ENGINE"),
            messages=messages,
            temperature=temperature or self.temperature,  # 值越低则输出文本随机性越低
            max_tokens=self.max_tokens,
        )

        return {"content": response.choices[0].message.content, "usage": response.usage}

    async def stream(self, prompt: str = "", system_prompt: str = "", messages: Optional[list[dict[str, str]]] = None, temperature: Optional[float] = None, engine: str="", max_tokens: Optional[int] = None):

        if not messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=engine or os.getenv("AZURE_OPENAI_LLM_ENGINE"),
            messages=messages,
            temperature=temperature or self.temperature,  # 值越低则输出文本随机性越低
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content

    async def embed(self, input_texts: list[str] | str, engine: str = "", dimensions: int = 1536, **kwargs) -> list[list[float]]:
        """
        Generate embeddings for a list of input texts.

        Args:
            input_texts (list[str] or str): List of strings to embed.
            engine (str, optional): Embedding engine/model to use. If not provided, uses OPENAI_EMBEDDING_ENGINE env var.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        t = time.time()
        embeddings = await self._embed_client.embeddings.create(
            input=input_texts,
            model=os.getenv("AZURE_OPENAI_EMBED_ENGINE") or engine,
            dimensions=dimensions,
            timeout=kwargs.get("timeout", None)
        )
        print(f"\n(azure openai embedding spent {time.time()-t:.3f} sec)")

        return [ele.embedding for ele in embeddings.data]



if __name__ == "__main__":
    azure_client = AzureOpenaiLLM(provider="azure")

    # # ==================================================
    # # simple non-streaming test
    # # ==================================================
    # resp = asyncio.run(azure_client.chat(prompt="hi"))
    # print(resp)

    # # ==================================================
    # # streaming test
    # # ==================================================
    # async def main():
    #     async for chunk in azure_client.stream(prompt="tell me a bed story"):
    #         print(chunk, end="", flush=True)
    # asyncio.run(main())

    # ==================================================
    # embedding test
    # ==================================================
    t = time.time()
    queries = [
        'Where is the body wash',
        'Where is body wash',
    ]
    for query in queries:
        embedding = asyncio.run(azure_client.embed(input_texts=query))
    # print(f"\n(Dense embedding 耗時: {time.time()-t:.3f} sec)")
    # print(embedding)