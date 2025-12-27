"""
Base provider interface for LLM providers.

This module defines the abstract base class and data structures for implementing
LLM provider adapters that support OpenAI, Anthropic, local models (Ollama), etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ChatCompletionResponse:
    """
    Unified response format across all LLM providers.

    This standardizes responses from different providers (OpenAI, Anthropic, Ollama)
    into a single consistent format for the application to consume.
    """
    content: str
    model: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str
    tool_calls: Optional[List[Dict]] = None
    raw_response: Optional[Any] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations (OpenAI, Anthropic, Local/Ollama) must implement
    this interface to ensure consistent behavior across the application.

    Design principles:
    - Both sync and async methods for flexibility
    - Unified response format via ChatCompletionResponse
    - Health check capability to support automatic fallback
    - Cost estimation for tracking expenses
    """

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Synchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Override default model for this provider
            tools: OpenAI-format tool definitions for function calling
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific additional arguments

        Returns:
            ChatCompletionResponse with unified format

        Raises:
            RuntimeError: If provider is unavailable or API call fails
        """
        pass

    @abstractmethod
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Asynchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Override default model for this provider
            tools: OpenAI-format tool definitions for function calling
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific additional arguments

        Returns:
            ChatCompletionResponse with unified format

        Raises:
            RuntimeError: If provider is unavailable or API call fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Health check - determine if provider is available/ready to use.

        Returns:
            True if provider can handle requests, False otherwise

        Examples:
            - OpenAI: Check if API key exists
            - Local/Ollama: Ping the local server
            - Anthropic: Check if API key exists
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Unique identifier for this provider.

        Returns:
            Provider name (e.g., "openai", "local", "anthropic")
        """
        pass

    @property
    def cost_per_1k_tokens(self) -> float:
        """
        Cost estimation in AUD per 1000 tokens.

        Returns:
            Cost per 1k tokens (0.0 for local models)

        Note:
            This is a rough estimate for cost tracking. Actual costs may vary
            based on model, usage patterns, and pricing changes.
        """
        return 0.0
