"""
Test that Phoenix tracing is completely disabled in ephemeral mode
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent
from src.monitoring.tracer import is_phoenix_enabled


async def test_phoenix_disabled_in_ephemeral():
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

    # Check environment variables were set
    assert os.getenv("EPHEMERAL_MODE") == "true", "EPHEMERAL_MODE not set"
    assert os.getenv("DISABLE_TRACING") == "true", "DISABLE_TRACING not set"

    # Check Phoenix is NOT enabled
    assert not is_phoenix_enabled(), "Phoenix should NOT be initialized in ephemeral mode"

    print("✓ Environment variables set correctly")
    print("✓ Phoenix tracing is disabled")

    # Try a simple operation to make sure nothing breaks
    await agent.connect_to_mcp_server()
    response, _, _ = await agent.chat("What is 2+2?")
    await agent.close()

    print("✓ Agent operations work without Phoenix")
    print(f"✓ Response received: {response[:100]}...")

    print("\n✓ PASSED: Phoenix tracing fully disabled in ephemeral mode")
    return True


async def test_phoenix_enabled_in_normal():
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

    # Check Phoenix IS enabled
    assert is_phoenix_enabled(), "Phoenix should be initialized in normal mode"

    print("✓ Phoenix tracing is enabled")

    await agent.connect_to_mcp_server()
    await agent.close()

    print("✓ PASSED: Phoenix tracing enabled in normal mode")
    return True


async def main():
    results = []

    results.append(await test_phoenix_disabled_in_ephemeral())
    results.append(await test_phoenix_enabled_in_normal())

    print("\n" + "="*80)
    print("PHOENIX TRACING TEST SUMMARY")
    print("="*80)
    print(f"Passed: {sum(results)}/{len(results)}")
    print("="*80)

    return all(results)


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
