"""
Basic tool calling tests for Doc-Mate agent.

Tests:
- Single book queries
- Multi-book comparative queries
- Author expansion
- Summary queries
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent


async def test_basic_tool_calling():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return

    agent = BookMateAgent(openai_api_key=api_key)

    # Connect to MCP server
    try:
        await agent.connect_to_mcp_server()
        print("✓ Connected to MCP server\n")
    except Exception as e:
        print(f"✗ Failed to connect to MCP server: {e}")
        return

    test_cases = [
        {
            "name": "Single book query",
            "query": "What is The Iliad about?",
            "expected_tool": "get_book_summary or search_book",
            "expected_books": ["ili"],
            "notes": "Should call tool for book in library"
        },
        {
            "name": "Comparative query (2 books)",
            "query": "compare The Iliad with The Odyssey on heroism",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody"],
            "notes": "MUST use search_multiple_books, NOT two search_book calls"
        },
        {
            "name": "Comparative query (2 authors - CRITICAL TEST)",
            "query": "compare Marcus and Homer on bravery",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["mam", "ili", "ody"],
            "notes": "CRITICAL: Must include ALL 3 books (Marcus=mam, Homer=ili+ody) in ONE search_multiple_books call"
        },
        {
            "name": "Author with multiple books",
            "query": "summarize Homer's work",
            "expected_tool": "get_book_summary for each book",
            "expected_books": ["ili", "ody"],
            "notes": "Should find ALL books by Homer (both Iliad and Odyssey)"
        },
        {
            "name": "Book summary query",
            "query": "What are the main themes in Meditations?",
            "expected_tool": "get_book_summary",
            "expected_books": ["mam"],
            "notes": "Should call get_book_summary for overview questions"
        },
        {
            "name": "Specific content search",
            "query": "Find passages about wisdom in The Odyssey",
            "expected_tool": "search_book",
            "expected_books": ["ody"],
            "notes": "Should use search_book for finding specific content"
        }
    ]

    print("=" * 80)
    print("BASIC TOOL CALLING TESTS")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}\n")

    passed = 0
    errors = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test['name']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected tool: {test['expected_tool']}")
        print(f"Expected books: {test['expected_books']}")
        print(f"Notes: {test['notes']}")
        print("-" * 80)

        try:
            response_text, _, _ = await agent.chat(test['query'])
            print("✓ Response received")
            print(f"First 200 chars: {response_text[:200]}...")
            print("-" * 80)
            passed += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            print("-" * 80)
            errors += 1

        # Small delay between tests
        await asyncio.sleep(1)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed} ({passed * 100 // len(test_cases)}%)")
    print(f"Errors: {errors}")
    print("=" * 80)

    # Cleanup
    await agent.close()
    print("\n✓ Tests complete. Review logs for detailed tool call behavior.")


if __name__ == "__main__":
    asyncio.run(test_basic_tool_calling())
