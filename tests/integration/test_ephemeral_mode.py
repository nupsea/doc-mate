"""
Test ephemeral and internal modes
"""
import pytest
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from src.mcp_client.agent import BookMateAgent


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_ephemeral_mode(anyio_backend):
    """Test that ephemeral mode doesn't save metrics or traces"""
    print("="*80)
    print("TEST: Ephemeral Mode")
    print("="*80)

    f_out = io.StringIO()
    f_err = io.StringIO()

    with redirect_stdout(f_out), redirect_stderr(f_err):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            agent = BookMateAgent(openai_api_key=api_key, ephemeral=True)
        else:
            agent = BookMateAgent(provider="local", ephemeral=True)

        await agent.connect_to_mcp_server()
        response, _, _, _ = await agent.chat("What is 2+2?")
        await agent.close()

    output = f_out.getvalue() + f_err.getvalue()

    has_metrics_msg = "[METRICS] Database persistence enabled" in output
    has_phoenix_msg = "Phoenix" in output or "OpenTelemetry" in output

    print(f"Metrics messages found: {has_metrics_msg}")
    print(f"Phoenix messages found: {has_phoenix_msg}")

    assert not has_metrics_msg, "Ephemeral mode leaked metrics"
    assert not has_phoenix_msg, "Ephemeral mode leaked tracing"
    print("✓ PASSED: Ephemeral mode working correctly")


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_internal_mode(anyio_backend):
    """Test that internal mode forces local LLM"""
    print("\n" + "="*80)
    print("TEST: Internal Mode")
    print("="*80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("no API key to test with")

    from src.llm.providers.local_provider import LocalProvider
    if not LocalProvider().is_available():
        pytest.skip("Local LLM (Ollama) not available")

    # Even with API key, internal mode should use local
    agent = BookMateAgent(openai_api_key=api_key, internal_mode=True)

    # Check that it's using local provider
    is_local = agent.llm_provider.provider_name == "local"

    await agent.connect_to_mcp_server()
    await agent.close()

    assert is_local, f"Internal mode using {agent.llm_provider.provider_name}"
    print("✓ PASSED: Internal mode forced local LLM")


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_ephemeral_internal_mode(anyio_backend):
    """Test combined ephemeral + internal mode"""
    print("\n" + "="*80)
    print("TEST: Ephemeral + Internal Mode")
    print("="*80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("no API key")

    from src.llm.providers.local_provider import LocalProvider
    if not LocalProvider().is_available():
        pytest.skip("Local LLM (Ollama) not available")

    f_out = io.StringIO()
    f_err = io.StringIO()

    with redirect_stdout(f_out), redirect_stderr(f_err):
        agent = BookMateAgent(openai_api_key=api_key, ephemeral=True, internal_mode=True)
        await agent.connect_to_mcp_server()
        response, _, _, _ = await agent.chat("Test query")
        await agent.close()

    output = f_out.getvalue() + f_err.getvalue()

    is_local = agent.llm_provider.provider_name == "local"
    has_metrics = "[METRICS]" in output
    has_phoenix = "Phoenix" in output

    assert is_local, "Not using local LLM in private mode"
    assert not has_metrics, "Metrics leaked in private mode"
    assert not has_phoenix, "Phoenix leaked in private mode"
    print("✓ PASSED: Private mode (ephemeral+internal) working")
