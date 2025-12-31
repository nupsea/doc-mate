"""
Edge case tests for Doc-Mate agent.

Tests:
- Unavailable authors
- Author expansion
- Contextual queries
- Chapter-specific searches
- Ambiguous references
- Three-way comparisons
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent


async def test_edge_cases():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return

    agent = BookMateAgent(openai_api_key=api_key)

    try:
        await agent.connect_to_mcp_server()
        print("✓ Connected to MCP server\n")
    except Exception as e:
        print(f"✗ Failed to connect to MCP server: {e}")
        return

    test_cases = [
        {
            "name": "Unavailable author (no substitution)",
            "query": "compare Marcus Aurelius and Jordan Peterson on ethics",
            "expected_tool": "search_book (mam only)",
            "expected_books": ["mam"],
            "notes": "Should NOT search Hegel or other philosophy books as Peterson substitute"
        },
        {
            "name": "Comparative with author expansion",
            "query": "compare Homer with Lewis Carroll on character arcs",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody", "alice"],
            "notes": "Homer → both ili and ody, Carroll → alice, use search_multiple_books"
        },
        {
            "name": "Contextual query (after discussing Alice)",
            "query": "who is the Cheshire Cat?",
            "expected_tool": "search_book (alice)",
            "expected_books": ["alice"],
            "notes": "Should infer book from conversation context or character name"
        },
        {
            "name": "Chapter-specific query",
            "query": "What does Marcus say in Chapter 3 about anger?",
            "expected_tool": "search_book (mam)",
            "expected_books": ["mam"],
            "notes": "Should search broadly, then filter results by Chapter 3"
        },
        {
            "name": "Collection-wide query",
            "query": "what themes connect all books in my collection?",
            "expected_tool": "get_book_summary for multiple books",
            "expected_books": ["multiple"],
            "notes": "Should call tools for each/multiple books, not use general knowledge"
        },
        {
            "name": "Ambiguous character reference",
            "query": "Tell me about the Mad Hatter",
            "expected_tool": "search_book (alice)",
            "expected_books": ["alice"],
            "notes": "Should infer Alice from character name and search that book"
        },
        {
            "name": "Three-way comparison",
            "query": "Compare views on virtue in The Iliad, The Odyssey, and Meditations",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody", "mam"],
            "notes": "Should use search_multiple_books with all three book slugs"
        }
    ]

    print("=" * 80)
    print("EDGE CASE TESTS")
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

        await asyncio.sleep(1)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed} ({passed * 100 // len(test_cases)}%)")
    print(f"Errors: {errors}")
    print("=" * 80)

    await agent.close()
    print("\n✓ Tests complete. Review logs for detailed tool call behavior.")


if __name__ == "__main__":
    asyncio.run(test_edge_cases())
