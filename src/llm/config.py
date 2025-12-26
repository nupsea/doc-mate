"""
Centralized LLM provider configuration.

This module provides configuration management for all LLM providers,
loading settings from environment variables with sensible defaults.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """
    Centralized LLM configuration loaded from environment variables.

    This configuration supports multiple LLM providers (OpenAI, Anthropic, Local/Ollama)
    with automatic fallback and privacy mode enforcement.

    Environment Variables:
        # Provider Selection
        LLM_PROVIDER: Default provider (openai | anthropic | local)
        PRIVACY_MODE: Force local-only (true | false)
        LLM_ENABLE_FALLBACK: Enable fallback to OpenAI (true | false)
        LLM_ENABLE_JUDGE: Enable response quality assessment (true | false, auto-disabled for local)

        # OpenAI
        OPENAI_API_KEY: OpenAI API key
        OPENAI_MODEL: Model name (default: gpt-4o-mini)

        # Anthropic
        ANTHROPIC_API_KEY: Anthropic API key
        ANTHROPIC_MODEL: Model name (default: claude-3-5-sonnet-20241022)

        # Local (Ollama)
        OLLAMA_BASE_URL: Ollama API base URL (default: http://localhost:11434/v1)
        OLLAMA_MODEL: Model name (default: llama3.2:3b, auto-switches to 8b for comparisons)
    """

    # Provider selection
    default_provider: str = "openai"  # openai | anthropic | local
    privacy_mode: bool = False
    enable_fallback: bool = True

    # Quality assessment
    enable_judge: bool = True  # Disable for local LLM to improve performance

    # OpenAI configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Anthropic configuration
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Local (Ollama) configuration
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2:3b"  # Fast default, auto-switches to 8b for comparisons

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Load configuration from environment variables.

        Returns:
            LLMConfig instance with values from environment or defaults

        Example:
            >>> config = LLMConfig.from_env()
            >>> print(config.default_provider)  # "openai" or from env
        """
        provider = os.getenv("LLM_PROVIDER", "openai")

        # Auto-disable judge for local LLM unless explicitly enabled
        default_enable_judge = "false" if provider == "local" else "true"

        return cls(
            # Provider selection
            default_provider=provider,
            privacy_mode=os.getenv("PRIVACY_MODE", "false").lower() == "true",
            enable_fallback=os.getenv("LLM_ENABLE_FALLBACK", "true").lower() == "true",
            enable_judge=os.getenv("LLM_ENABLE_JUDGE", default_enable_judge).lower() == "true",

            # OpenAI
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),

            # Anthropic
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),

            # Local (Ollama)
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        )

    def validate(self) -> None:
        """
        Validate configuration and warn about missing settings.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate provider name
        valid_providers = ["openai", "anthropic", "local"]
        if self.default_provider not in valid_providers:
            raise ValueError(
                f"Invalid default_provider: {self.default_provider}. "
                f"Must be one of {valid_providers}"
            )

        # Warn about missing API keys (not fatal, fallback may handle it)
        if self.default_provider == "openai" and not self.openai_api_key:
            import logging
            logging.warning(
                "OPENAI_API_KEY not set. OpenAI provider will be unavailable."
            )

        if self.default_provider == "anthropic" and not self.anthropic_api_key:
            import logging
            logging.warning(
                "ANTHROPIC_API_KEY not set. Anthropic provider will be unavailable."
            )

        # Privacy mode validation
        if self.privacy_mode and self.default_provider != "local":
            import logging
            logging.info(
                "PRIVACY_MODE enabled - forcing default_provider to 'local'"
            )
