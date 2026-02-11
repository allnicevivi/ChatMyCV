import os
from typing import Optional

from dotenv import load_dotenv

from .base import LLM
from .azure_module import AzureOpenaiLLM
from .openai_module import OpenAILLM
from .claude_module import ClaudeLLM
# from .gemini_module import GeminiLLM

load_dotenv()

# Provider mapping
LLM_PROVIDERS = {
    "azure": AzureOpenaiLLM,
    "openai": OpenAILLM,
    "claude": ClaudeLLM,
    # "gemini": GeminiLLM,
}

# Providers that support embedding
EMBED_PROVIDERS = {
    "azure": AzureOpenaiLLM,
    "openai": OpenAILLM,
    # "gemini": GeminiLLM,
}


def create_llm_client(provider: Optional[str] = None, **kwargs) -> LLM:
    """
    Create an LLM client based on the specified provider.

    Args:
        provider: LLM provider name. If not specified, uses LLM_PROVIDER env var.
                  Options: azure, openai, claude, gemini

    Returns:
        LLM instance
    """
    provider = provider or os.getenv("LLM_PROVIDER", "azure")
    provider = provider.lower()

    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Available providers: {list(LLM_PROVIDERS.keys())}"
        )

    return LLM_PROVIDERS[provider](provider=provider, **kwargs)


def create_embed_client(provider: Optional[str] = None, **kwargs) -> LLM:
    """
    Create an embedding client based on the specified provider.

    Args:
        provider: Embedding provider name. If not specified, uses EMBED_PROVIDER env var.
                  Options: azure, openai, gemini (claude not supported)

    Returns:
        LLM instance with embedding support
    """
    provider = provider or os.getenv("EMBED_PROVIDER", "azure")
    provider = provider.lower()

    if provider not in EMBED_PROVIDERS:
        raise ValueError(
            f"Unknown or unsupported embedding provider: {provider}. "
            f"Available providers: {list(EMBED_PROVIDERS.keys())}. "
            "Note: Claude does not support embeddings."
        )

    return EMBED_PROVIDERS[provider](provider=provider, **kwargs)


# Create default clients based on environment variables
llm_client = create_llm_client()
embed_client = create_embed_client()

# Legacy alias for backward compatibility
azure_client = llm_client
