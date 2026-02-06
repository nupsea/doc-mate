"""
Test switching between privacy modes
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent
from src.monitoring.tracer import is_phoenix_enabled


async def test_ephemeral_to_normal_switch():
    """Test switching from ephemeral to normal mode enables tracing"""
    print("="*80)
    print("TEST: Switch from Ephemeral to Normal Mode")
    print("="*80)

    # Clean environment first
    for key in ["DISABLE_TRACING", "EPHEMERAL_MODE"]:
        if key in os.environ:
            del os.environ[key]

    # Step 1: Start in ephemeral mode
    print("\n[1] Creating agent in EPHEMERAL mode...")
    agent1 = BookMateAgent(provider="local", ephemeral=True)

    env_check_1 = os.getenv("DISABLE_TRACING") == "true"
    phoenix_check_1 = is_phoenix_enabled()

    print(f"  - DISABLE_TRACING env var: {env_check_1}")
    print(f"  - Phoenix enabled: {phoenix_check_1}")

    assert env_check_1, "DISABLE_TRACING should be set"
    assert not phoenix_check_1, "Phoenix should NOT be enabled"
    print("  ✓ Ephemeral mode working")

    await agent1.connect_to_mcp_server()
    await agent1.close()

    # Step 2: Switch to normal mode (simulate UI behavior)
    print("\n[2] Creating NEW agent in NORMAL mode...")
    agent2 = BookMateAgent(provider="local", ephemeral=False)

    env_check_2 = os.getenv("DISABLE_TRACING")
    phoenix_check_2 = is_phoenix_enabled()

    print(f"  - DISABLE_TRACING env var: {env_check_2}")
    print(f"  - Phoenix enabled: {phoenix_check_2}")

    assert env_check_2 is None, "DISABLE_TRACING should be cleared"
    assert phoenix_check_2, "Phoenix SHOULD be enabled now"
    print("  ✓ Normal mode working")

    await agent2.connect_to_mcp_server()
    await agent2.close()

    print("\n✓ PASSED: Mode switching works correctly")
    return True


async def test_normal_to_ephemeral_switch():
    """Test switching from normal to ephemeral mode disables tracing"""
    print("\n" + "="*80)
    print("TEST: Switch from Normal to Ephemeral Mode")
    print("="*80)

    # Clean environment first
    for key in ["DISABLE_TRACING", "EPHEMERAL_MODE"]:
        if key in os.environ:
            del os.environ[key]

    # Step 1: Start in normal mode
    print("\n[1] Creating agent in NORMAL mode...")
    agent1 = BookMateAgent(provider="local", ephemeral=False)

    phoenix_check_1 = is_phoenix_enabled()
    print(f"  - Phoenix enabled: {phoenix_check_1}")
    assert phoenix_check_1, "Phoenix should be enabled"
    print("  ✓ Normal mode working")

    await agent1.connect_to_mcp_server()
    await agent1.close()

    # Step 2: Switch to ephemeral mode
    print("\n[2] Creating NEW agent in EPHEMERAL mode...")
    agent2 = BookMateAgent(provider="local", ephemeral=True)

    env_check_2 = os.getenv("DISABLE_TRACING") == "true"
    print(f"  - DISABLE_TRACING env var: {env_check_2}")

    # Note: Phoenix may still report as "enabled" because uninstrumentation
    # doesn't change the global flag, but DISABLE_TRACING prevents new traces
    assert env_check_2, "DISABLE_TRACING should be set"
    print("  ✓ Ephemeral mode working (new traces disabled)")

    await agent2.connect_to_mcp_server()
    await agent2.close()

    print("\n✓ PASSED: Mode switching works correctly")
    return True


async def main():
    results = []

    results.append(await test_ephemeral_to_normal_switch())
    results.append(await test_normal_to_ephemeral_switch())

    print("\n" + "="*80)
    print("MODE SWITCHING TEST SUMMARY")
    print("="*80)
    print(f"Passed: {sum(results)}/{len(results)}")
    print("="*80)

    return all(results)


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
