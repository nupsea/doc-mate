"""
Local LLM provider implementation for Ollama.

Supports dual-model mode:
- llama3.2:3b (default, fast)
- llama3.1:8b (for comparisons only)
"""

import os
import logging
from typing import Optional, List, Dict
from openai import OpenAI, AsyncOpenAI
import requests
import asyncio

from .base import LLMProvider, ChatCompletionResponse
from .query_classifier import needs_complex_model

logger = logging.getLogger(__name__)

# Semaphore to limit concurrent requests (prevents system overload)
_request_semaphore: Optional[asyncio.Semaphore] = None


class LocalProvider(LLMProvider):
    """
    Local LLM provider using Ollama's OpenAI-compatible API.

    Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1,
    allowing us to reuse the OpenAI Python SDK with a custom base_url.

    Key features:
    - Supports Llama 3.3 70B and other Ollama models
    - Function calling compatible with OpenAI format
    - JSON response format workaround for compatibility
    - Health checks via Ollama's native API
    - Zero cost (local inference)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama3.2:3b",  # Default to fast model
        timeout: int = 120,
        max_concurrent_requests: int = 2,
    ):
        """
        Initialize Local (Ollama) provider.

        Default: llama3.2:3b (fast)
        Auto-switches to llama3.1:8b for comparisons
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = model
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests

        # Dual model config
        self.fast_model = "llama3.2:3b"
        self.complex_model = "llama3.1:8b"

        # Concurrency control
        global _request_semaphore
        if _request_semaphore is None:
            _request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._semaphore = _request_semaphore

        # OpenAI SDK clients
        self.client = OpenAI(base_url=self.base_url, api_key="ollama", timeout=timeout)
        self.async_client = AsyncOpenAI(base_url=self.base_url, api_key="ollama", timeout=timeout)

        logger.info(f"[LOCAL] Default: {self.fast_model}, Complex: {self.complex_model}")

    def chat_completion(
        self,
        messages,
        model=None,
        tools=None,
        temperature=0.7,
        max_tokens=None,
        **kwargs
    ) -> ChatCompletionResponse:
        """Synchronous chat completion via Ollama."""
        try:
            # Select model: check if query needs complex model
            if not model:
                user_query = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                if needs_complex_model(user_query):
                    model = self.complex_model
                    logger.info(f"[ROUTING] Using {model} for comparison query")
                else:
                    model = self.fast_model

            # Handle JSON response format
            if kwargs.get("response_format", {}).get("type") == "json_object":
                kwargs.pop("response_format")
                messages = self._inject_json_instruction(messages)

            # Set max_tokens
            if max_tokens is None:
                max_tokens = 1024 if model == self.fast_model else 2048

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return self._convert_response(response)

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Local provider failed: {e}") from e

    async def chat_completion_async(
        self,
        messages,
        model=None,
        tools=None,
        temperature=0.7,
        max_tokens=None,
        **kwargs
    ) -> ChatCompletionResponse:
        """Asynchronous chat completion via Ollama with concurrency control."""
        async with self._semaphore:
            try:
                # Select model: check if query needs complex model
                if not model:
                    user_query = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                    if needs_complex_model(user_query):
                        model = self.complex_model
                        logger.info(f"[ROUTING] Using {model} for comparison query")
                    else:
                        model = self.fast_model

                # Handle JSON response format
                if kwargs.get("response_format", {}).get("type") == "json_object":
                    kwargs.pop("response_format")
                    messages = self._inject_json_instruction(messages)

                # Set max_tokens
                if max_tokens is None:
                    max_tokens = 1024 if model == self.fast_model else 2048

                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return self._convert_response(response)

            except Exception as e:
                logger.error(f"Ollama API error (async): {e}")
                raise RuntimeError(f"Local provider failed: {e}") from e

    def is_available(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if Ollama server responds to health check
        """
        try:
            # Ping Ollama's native API endpoint
            ollama_base = self.base_url.replace("/v1", "")
            response = requests.get(f"{ollama_base}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "local"

    @property
    def cost_per_1k_tokens(self) -> float:
        """
        Cost estimation for local models.

        Returns:
            0.0 (local models are free)
        """
        return 0.0

    def _inject_json_instruction(self, messages: List[Dict]) -> List[Dict]:
        """
        Inject JSON formatting instruction into messages.

        Ollama doesn't support OpenAI's response_format parameter,
        so we inject JSON instructions into the system message.

        Args:
            messages: Original message list

        Returns:
            Modified messages with JSON instruction
        """
        json_instruction = (
            "\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object. "
            "Start your response with { and end with }. "
            "Do not include any explanatory text before or after the JSON. "
            "Ensure all strings are properly quoted and escaped."
        )

        # Copy messages to avoid mutating original
        messages = list(messages)

        # Find system message and append instruction
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + json_instruction
                return messages

        # No system message found, prepend one
        messages.insert(
            0, {"role": "system", "content": f"You are a helpful assistant.{json_instruction}"}
        )

        return messages

    def _convert_response(self, response) -> ChatCompletionResponse:
        """
        Convert Ollama response to unified ChatCompletionResponse format.

        Ollama's OpenAI-compatible API returns the same format as OpenAI,
        so we can use the same conversion logic.

        Args:
            response: Ollama ChatCompletion response object

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
