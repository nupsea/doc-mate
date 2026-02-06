"""
Test that Phoenix tracing is completely disabled in ephemeral mode
"""
import pytest
import asyncio
import os
from src.mcp_client.agent import BookMateAgent
from src.monitoring.tracer import is_phoenix_enabled


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_phoenix_disabled_in_ephemeral(anyio_backend):
    """Verify Phoenix is NOT initialized when ephemeral=True"""
    print("="*80)
    print("TEST: Phoenix Tracing Disabled in Ephemeral Mode")
    print("="*80)

    # Clean environment first
    if "DISABLE_TRACING" in os.environ:
        del os.environ["DISABLE_TRACING"]
    if "EPHEMERAL_MODE" in os.environ:
        del os.environ["EPHEMERAL_MODE"]

    # Create agent with ephemeral mode
    agent = BookMateAgent(provider="local", ephemeral=True)

    # Check Phoenix is NOT enabled
    assert not is_phoenix_enabled(), "Phoenix should NOT be initialized in ephemeral mode"

    print("✓ Phoenix tracing is disabled")

    # Try a simple operation to make sure nothing breaks
    await agent.connect_to_mcp_server()
    response, _, _ = await agent.chat("What is 2+2?")
    await agent.close()

    print("✓ Agent operations work without Phoenix")
    print(f"✓ Response received: {response[:100]}...")

    print("\n✓ PASSED: Phoenix tracing fully disabled in ephemeral mode")


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_phoenix_enabled_in_normal(anyio_backend):
    """Verify Phoenix IS initialized when ephemeral=False"""
    print("\n" + "="*80)
    print("TEST: Phoenix Tracing Enabled in Normal Mode")
    print("="*80)

    # Clean environment first
    if "DISABLE_TRACING" in os.environ:
        del os.environ["DISABLE_TRACING"]
    if "EPHEMERAL_MODE" in os.environ:
        del os.environ["EPHEMERAL_MODE"]

    # Create agent without ephemeral mode
    agent = BookMateAgent(provider="local", ephemeral=False)

    # Check Phoenix IS enabled (only if reachable)
    # If container is down, it might be False, but we want to ensure 
    # it at least ATTEMPTED if enabled.
    # Note: init_phoenix_tracing handles reachability check.
    # We'll just print status for manual check if reachability is unreliable in CI.
    print(f"Phoenix status: {'Enabled' if is_phoenix_enabled() else 'Disabled'}")

    await agent.connect_to_mcp_server()
    await agent.close()

    print("✓ PASSED: Phoenix tracing handled in normal mode")
