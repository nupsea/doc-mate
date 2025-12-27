"""
LLM provider abstraction package.

This package provides a unified interface for multiple LLM providers
(OpenAI, Local/Ollama, and future Anthropic support) with automatic
fallback and privacy mode enforcement.

Usage:
    >>> from src.llm.providers import ModelRouter
    >>> router = ModelRouter()
    >>> provider = router.get_provider()  # Uses default from config
    >>> response = await provider.chat_completion_async(messages=[...])

Privacy Mode:
    >>> # Force local-only (no external API calls)
    >>> os.environ["PRIVACY_MODE"] = "true"
    >>> router = ModelRouter()
    >>> provider = router.get_provider()  # Always returns LocalProvider
"""

from .base import LLMProvider, ChatCompletionResponse
from .openai_provider import OpenAIProvider
from .local_provider import LocalProvider
from .model_router import ModelRouter

__all__ = [
    "LLMProvider",
    "ChatCompletionResponse",
    "OpenAIProvider",
    "LocalProvider",
    "ModelRouter",
]
