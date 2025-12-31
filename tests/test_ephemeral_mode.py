"""
Test ephemeral and internal modes
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent


async def test_ephemeral_mode():
    """Test that ephemeral mode doesn't save metrics or traces"""
    print("="*80)
    print("TEST: Ephemeral Mode")
    print("="*80)

    # Capture stdout to check for metrics messages
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    f_out = io.StringIO()
    f_err = io.StringIO()

    with redirect_stdout(f_out), redirect_stderr(f_err):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            agent = BookMateAgent(openai_api_key=api_key, ephemeral=True)
        else:
            agent = BookMateAgent(provider="local", ephemeral=True)

        await agent.connect_to_mcp_server()
        response, _, _ = await agent.chat("What is 2+2?")
        await agent.close()

    output = f_out.getvalue() + f_err.getvalue()

    has_metrics_msg = "[METRICS] Database persistence enabled" in output
    has_phoenix_msg = "Phoenix" in output or "OpenTelemetry" in output

    print(f"Metrics messages found: {has_metrics_msg}")
    print(f"Phoenix messages found: {has_phoenix_msg}")

    if has_metrics_msg or has_phoenix_msg:
        print("✗ FAILED: Ephemeral mode leaked metrics/tracing")
        return False
    else:
        print("✓ PASSED: Ephemeral mode working correctly")
        return True


async def test_internal_mode():
    """Test that internal mode forces local LLM"""
    print("\n" + "="*80)
    print("TEST: Internal Mode")
    print("="*80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Skipping (no API key to test with)")
        return True

    # Even with API key, internal mode should use local
    agent = BookMateAgent(openai_api_key=api_key, internal_mode=True)

    # Check that it's using local provider
    is_local = agent.llm_provider.provider_name == "local"

    await agent.connect_to_mcp_server()
    await agent.close()

    if is_local:
        print("✓ PASSED: Internal mode forced local LLM")
        return True
    else:
        print(f"✗ FAILED: Internal mode using {agent.llm_provider.provider_name}")
        return False


async def test_ephemeral_internal_mode():
    """Test combined ephemeral + internal mode"""
    print("\n" + "="*80)
    print("TEST: Ephemeral + Internal Mode")
    print("="*80)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Skipping (no API key)")
        return True

    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    f_out = io.StringIO()
    f_err = io.StringIO()

    with redirect_stdout(f_out), redirect_stderr(f_err):
        agent = BookMateAgent(openai_api_key=api_key, ephemeral=True, internal_mode=True)
        await agent.connect_to_mcp_server()
        response, _, _ = await agent.chat("Test query")
        await agent.close()

    output = f_out.getvalue() + f_err.getvalue()

    is_local = agent.llm_provider.provider_name == "local"
    has_metrics = "[METRICS]" in output
    has_phoenix = "Phoenix" in output

    if is_local and not has_metrics and not has_phoenix:
        print("✓ PASSED: Private mode (ephemeral+internal) working")
        return True
    else:
        print(f"✗ FAILED: Local:{is_local}, Metrics:{has_metrics}, Phoenix:{has_phoenix}")
        return False


async def main():
    results = []

    results.append(await test_ephemeral_mode())
    results.append(await test_internal_mode())
    results.append(await test_ephemeral_internal_mode())

    print("\n" + "="*80)
    print("PRIVACY MODES TEST SUMMARY")
    print("="*80)
    print(f"Passed: {sum(results)}/{len(results)}")
    print("="*80)

    return all(results)


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
