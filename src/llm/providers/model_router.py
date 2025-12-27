"""
Model router for intelligent LLM provider selection with automatic fallback.

This module provides smart routing to LLM providers (OpenAI, Local/Ollama, Anthropic)
with health checks and automatic fallback logic.
"""

from typing import Optional, Dict
import logging

from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .local_provider import LocalProvider
from ..config import LLMConfig

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Route queries to optimal LLM provider with automatic fallback.

    This router follows the same pattern as ParserRouter (for document types),
    providing intelligent provider selection based on configuration, health checks,
    and privacy requirements.

    Design principles:
    - Privacy mode enforcement (local-only, no fallback)
    - Health-based provider selection
    - Automatic fallback to OpenAI when primary unavailable
    - Provider instance caching for performance
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize model router.

        Args:
            config: LLMConfig instance (defaults to loading from environment)
        """
        self.config = config or LLMConfig.from_env()
        self._providers: Dict[str, LLMProvider] = {}

    def get_provider(
        self, provider_name: Optional[str] = None, fallback: Optional[bool] = None
    ) -> LLMProvider:
        """
        Get LLM provider with automatic fallback.

        Args:
            provider_name: Specific provider to use (overrides config default)
            fallback: Enable fallback to OpenAI if primary fails
                     (overrides config, except in privacy mode)

        Returns:
            LLMProvider instance ready to handle requests

        Raises:
            RuntimeError: If no available provider found

        Examples:
            >>> router = ModelRouter()
            >>> provider = router.get_provider()  # Uses default from config
            >>> provider = router.get_provider("local")  # Force local
        """
        # Privacy mode: force local, disable fallback
        if self.config.privacy_mode:
            logger.info("Privacy mode enabled - forcing local provider")
            provider_name = "local"
            fallback = False  # No fallback in privacy mode

        # Use specified provider or default from config
        provider_name = provider_name or self.config.default_provider
        enable_fallback = (
            fallback if fallback is not None else self.config.enable_fallback
        )

        # Try primary provider with health check
        try:
            provider = self._get_provider_instance(provider_name)
            if provider.is_available():
                logger.info(f"Using {provider_name} provider")
                return provider
            else:
                logger.warning(
                    f"{provider_name} provider unavailable (health check failed)"
                )
        except Exception as e:
            logger.error(f"Failed to initialize {provider_name} provider: {e}")

        # Attempt fallback to OpenAI
        if enable_fallback and provider_name != "openai":
            logger.info(f"Falling back to OpenAI provider (primary: {provider_name})")
            try:
                openai_provider = self._get_provider_instance("openai")
                if openai_provider.is_available():
                    return openai_provider
                else:
                    logger.error("OpenAI fallback also unavailable")
            except Exception as e:
                logger.error(f"OpenAI fallback failed: {e}")

        # No available provider
        raise RuntimeError(
            f"No available LLM provider. Tried: {provider_name}"
            + (", openai (fallback)" if enable_fallback else "")
        )

    def _get_provider_instance(self, provider_name: str) -> LLMProvider:
        """
        Get or create provider instance (cached for performance).

        Args:
            provider_name: Provider identifier (openai, local, anthropic)

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider_name is unknown
        """
        # Return cached instance if exists
        if provider_name in self._providers:
            return self._providers[provider_name]

        # Create new provider instance
        if provider_name == "openai":
            provider = OpenAIProvider(model=self.config.openai_model)

        elif provider_name == "local":
            provider = LocalProvider(
                base_url=self.config.ollama_base_url, model=self.config.ollama_model
            )

        elif provider_name == "anthropic":
            # Anthropic provider not yet implemented
            # Will be added in future iteration
            raise ValueError(
                "Anthropic provider not yet implemented. "
                "Use 'openai' or 'local' instead."
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Valid options: openai, local"
            )

        # Cache and return
        self._providers[provider_name] = provider
        return provider

    def clear_cache(self):
        """
        Clear provider instance cache.

        Useful for testing or when provider configuration changes.
        """
        self._providers.clear()
        logger.debug("Provider cache cleared")
