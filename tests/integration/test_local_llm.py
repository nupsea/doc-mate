"""
Test suite for local LLM (Ollama) agent behavior.

Tests local LLM-specific concerns:
1. Function calling reliability with smaller models
2. Parameter normalization (list vs string)
3. Temperature and max_tokens settings
4. Simple vs full prompts
5. Edge cases specific to local models
"""
import asyncio
from src.mcp_client.agent import BookMateAgent


async def test_local_llm():
    agent = BookMateAgent(provider="local")

    try:
        await agent.connect_to_mcp_server()
        print("Connected to MCP server\n")
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")
        return

    test_results = {
        "total": 0,
        "passed": 0,
        "errors": 0,
        "failures": []
    }

    test_cases = [
        # Basic function calling
        {
            "name": "Single book query",
            "query": "What is The Iliad about?",
            "notes": "Test basic function calling with local LLM"
        },
        {
            "name": "Book identifier normalization",
            "query": "What does Mike highlight in the Sample Meeting?",
            "notes": "CRITICAL: Tests list vs string normalization for book_identifier"
        },
        {
            "name": "Comparative query",
            "query": "compare Marcus and Homer on persistence",
            "notes": "Tests search_multiple_books with local LLM"
        },
        {
            "name": "Simple search",
            "query": "Find passages about wisdom in The Odyssey",
            "notes": "Tests search_book function calling"
        },
        {
            "name": "Author with multiple books",
            "query": "summarize Homer's work",
            "notes": "Tests finding all books by author"
        },
        {
            "name": "Conversation document",
            "query": "What topics were discussed in the sample meeting?",
            "notes": "Tests conversation document search"
        },
        {
            "name": "Technical document",
            "query": "What does the design data intensive apps book say about replication?",
            "notes": "Tests tech doc search"
        },
        {
            "name": "Empty result handling",
            "query": "Tell me about Harry Potter",
            "notes": "Tests graceful handling of unavailable books"
        },
    ]

    print("=" * 80)
    print("LOCAL LLM (OLLAMA) - FUNCTION CALLING TEST SUITE")
    print("=" * 80)
    print(f"\nTotal tests: {len(test_cases)}")
    print(f"Model: {agent.llm_provider.model}")
    print(f"Temperature: {agent._get_temperature_for_provider()}")
    print(f"Max tokens: {agent._get_max_tokens_for_provider()}")
    print(f"Using simple prompt: {agent.llm_provider.provider_name == 'local'}")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        test_results["total"] += 1

        print(f"\n[{i}/{len(test_cases)}] {test['name']}")
        print(f"Query: '{test['query']}'")
        print(f"Notes: {test['notes']}")
        print("-" * 80)

        try:
            response_text, _, _ = await agent.chat(test['query'])

            print("Response received")
            print(f"First 200 chars: {response_text[:200]}...")
            print("-" * 80)

            test_results["passed"] += 1

        except Exception as e:
            print(f"Error: {e}")
            print("-" * 80)

            test_results["errors"] += 1
            test_results["failures"].append({
                "name": test['name'],
                "query": test['query'],
                "error": str(e)
            })

        await asyncio.sleep(1)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print("\nOVERALL RESULTS:")
    print(f"  Total Tests:    {test_results['total']}")
    print(f"  Passed:         {test_results['passed']}")
    print(f"  Errors:         {test_results['errors']}")

    if test_results["errors"] > 0:
        print("\nERROR DETAILS:")
        for failure in test_results["failures"]:
            print(f"  {failure['name']}")
            print(f"    Query: {failure['query']}")
            print(f"    Error: {failure['error']}")
            print()

    print("\n" + "=" * 80)
    print("LOCAL LLM VERIFICATION CHECKLIST")
    print("=" * 80)
    print("\nCRITICAL CHECKS:")
    print()
    print("1. Function calling reliability")
    print("   - Did local LLM correctly identify and call tools?")
    print("   - Check logs for tool call JSON parsing errors")
    print()
    print("2. Parameter normalization")
    print("   - Test 2: 'Sample Meeting' query")
    print("   - Check for 'list' object has no attribute 'lower' error")
    print("   - Should normalize ['stm'] -> 'stm'")
    print()
    print("3. Temperature settings")
    print("   - Local LLM uses temperature=0.0 for deterministic function calls")
    print("   - Check if tool calls are consistent across runs")
    print()
    print("4. Response quality")
    print("   - Are responses coherent despite smaller model?")
    print("   - Do responses include citations?")
    print()
    print("5. Prompt optimization")
    print("   - Local LLM uses simplified system prompt")
    print("   - Check if responses are still accurate")
    print("=" * 80)

    await agent.close()
    print("\nTests complete.")


if __name__ == "__main__":
    asyncio.run(test_local_llm())
