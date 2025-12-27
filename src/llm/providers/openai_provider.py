"""
OpenAI provider implementation.

Wraps the OpenAI Python SDK to implement the LLMProvider interface,
maintaining backward compatibility while enabling provider abstraction.
"""

import os
import logging
from typing import Optional
from openai import OpenAI, AsyncOpenAI

from .base import LLMProvider, ChatCompletionResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation.

    Wraps the official OpenAI Python SDK to provide both synchronous and
    asynchronous chat completions with function calling support.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Default model to use (default: gpt-4o-mini)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Initialize both sync and async clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def chat_completion(
        self,
        messages,
        model=None,
        tools=None,
        temperature=0.7,
        max_tokens=None,
        **kwargs
    ) -> ChatCompletionResponse:
        """Synchronous chat completion via OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return self._convert_response(response)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI provider failed: {e}") from e

    async def chat_completion_async(
        self,
        messages,
        model=None,
        tools=None,
        temperature=0.7,
        max_tokens=None,
        **kwargs
    ) -> ChatCompletionResponse:
        """Asynchronous chat completion via OpenAI API."""
        try:
            response = await self.async_client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return self._convert_response(response)

        except Exception as e:
            logger.error(f"OpenAI API error (async): {e}")
            raise RuntimeError(f"OpenAI provider failed: {e}") from e

    def is_available(self) -> bool:
        """
        Check if OpenAI provider is available.

        Returns:
            True if OPENAI_API_KEY is set
        """
        return bool(self.api_key)

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "openai"

    @property
    def cost_per_1k_tokens(self) -> float:
        """
        Cost estimation in AUD per 1000 tokens (rough approximation).

        Note: Assumes GPT-4o-mini pricing. Actual costs may vary.
        Exchange rate: ~1 USD = 1.5 AUD (approximate)
        """
        # Pricing in USD, converted to AUD
        pricing_usd = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-4o-2024-08-06": 0.005,
            "gpt-4-turbo": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0015,
        }

        usd_cost = pricing_usd.get(self.model, 0.001)
        aud_cost = usd_cost * 1.5  # Approximate USD to AUD conversion

        return aud_cost

    def _convert_response(self, response) -> ChatCompletionResponse:
        """
        Convert OpenAI response to unified ChatCompletionResponse format.

        Args:
            response: OpenAI ChatCompletion response object

        Returns:
            ChatCompletionResponse with standardized fields
        """
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls if present
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return ChatCompletionResponse(
            content=message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            raw_response=response,
        )
