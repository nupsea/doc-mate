"""
Phoenix tracing initialization for LLM observability.

Phoenix automatically captures all OpenAI API calls (prompts, responses, tokens, latency)
and visualizes them in a web UI (default: http://localhost:6006).

Usage:
    Call init_phoenix_tracing() once at application startup, before any OpenAI calls.
    All subsequent OpenAI interactions will be automatically traced.

    For ephemeral mode (no tracing), set DISABLE_TRACING=true before initialization.

Environment variables:
    PHOENIX_COLLECTOR_ENDPOINT: Collector URL (default: http://localhost:4317)
    PHOENIX_UI_URL: UI URL for links in the application (default: http://localhost:6006)
    PHOENIX_PROJECT_NAME: Project name in UI (default: book-mate)
    DISABLE_TRACING: Set to 'true' to completely disable tracing
"""

import os
import socket
from urllib.parse import urlparse
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

_phoenix_initialized = False
_instrumentor = None


def _is_collector_reachable(endpoint: str) -> bool:
    """Check if the Phoenix collector is reachable."""
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 4317
        
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except (socket.timeout, socket.error):
        return False


def init_phoenix_tracing():
    """Initialize Phoenix tracing. Call once at startup or when re-enabling."""
    global _phoenix_initialized, _instrumentor

    # Check if tracing is explicitly disabled via env var (global override)
    if os.getenv("DISABLE_TRACING", "false").lower() == "true":
        return

    if _phoenix_initialized:
        return

    try:
        collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
        
        # Check connectivity before initializing to avoid gRPC error spam
        if not _is_collector_reachable(collector_endpoint):
            print(f"[PHOENIX] Tracing skipped: Collector at {collector_endpoint} is not reachable")
            return

        project_name = os.getenv("PHOENIX_PROJECT_NAME", "book-mate")
        ui_url = os.getenv("PHOENIX_UI_URL", "http://localhost:6006")

        tracer_provider = register(
            project_name=project_name,
            endpoint=collector_endpoint,
        )

        _instrumentor = OpenAIInstrumentor()
        _instrumentor.instrument(tracer_provider=tracer_provider)

        _phoenix_initialized = True
        print(f"[PHOENIX] Initialized | Project: {project_name} | UI: {ui_url}")

    except Exception as e:
        print(f"[PHOENIX] Failed to initialize: {e}")


def disable_tracing():
    """Disable Phoenix tracing and uninstrument OpenAI calls."""
    global _phoenix_initialized, _instrumentor

    if _instrumentor is not None:
        try:
            _instrumentor.uninstrument()
            print("[PHOENIX] Tracing disabled and uninstrumented")
            _phoenix_initialized = False
            _instrumentor = None
        except Exception as e:
            print(f"[PHOENIX] Warning: Could not uninstrument: {e}")


def is_phoenix_enabled() -> bool:
    """Check if Phoenix tracing is enabled."""
    return _phoenix_initialized
