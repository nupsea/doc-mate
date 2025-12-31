"""
Document type specific tests for Doc-Mate agent.

Tests:
- Conversation documents (with/without speaker labels)
- Technical documents
- Movie scripts
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent


async def test_document_types():
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
        # CONVERSATION DOCUMENTS
        {
            "category": "CONVERSATION",
            "name": "Speaker attribution WITH labels",
            "query": "What does Sarah Chen say in the sample meeting?",
            "expected_tool": "search_book (stm)",
            "expected_books": ["stm"],
            "notes": "Should return passages with [Speakers: Sarah Chen, ...] citations"
        },
        {
            "category": "CONVERSATION",
            "name": "Speaker attribution WITHOUT labels (CRITICAL)",
            "query": "What does Anup say in Dec Town Hall?",
            "expected_tool": "search_book (dth)",
            "expected_books": ["dth"],
            "notes": "CRITICAL: Must respond ONLY with 'This document lacks speaker labels.' NO topics, NO inference"
        },
        {
            "category": "CONVERSATION",
            "name": "Meeting topic search",
            "query": "What topics were discussed in the sample meeting about Q1?",
            "expected_tool": "search_book (stm)",
            "expected_books": ["stm"],
            "notes": "Should search for 'Q1 planning budget hiring' and return relevant passages"
        },
        {
            "category": "CONVERSATION",
            "name": "Compare meetings",
            "query": "compare the sample meeting and dec town hall on key decisions",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["stm", "dth"],
            "notes": "Should use search_multiple_books with both conversation slugs"
        },

        # TECHNICAL DOCUMENTS
        {
            "category": "TECH_DOC",
            "name": "Technical concept search",
            "query": "What does the design data intensive apps book say about replication?",
            "expected_tool": "search_book (ddia)",
            "expected_books": ["ddia"],
            "notes": "Should search for 'replication consistency distributed' in tech doc"
        },
        {
            "category": "TECH_DOC",
            "name": "Tech doc chapter summaries",
            "query": "Give me a chapter breakdown of design data intensive apps",
            "expected_tool": "get_chapter_summaries (ddia)",
            "expected_books": ["ddia"],
            "notes": "Should return chapter-by-chapter structure"
        },
        {
            "category": "TECH_DOC",
            "name": "Cross-genre comparison",
            "query": "compare how the Gita and design data intensive apps approach balance and trade-offs",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["gita", "ddia"],
            "notes": "Cross-genre comparison: philosophical vs technical text"
        },

        # MOVIE SCRIPTS
        {
            "category": "SCRIPT",
            "name": "Character dialogue search",
            "query": "What does Neo say about choice in The Matrix?",
            "expected_tool": "search_book (matrix)",
            "expected_books": ["matrix"],
            "notes": "Should search for 'Neo choice decision free will' in script"
        },
        {
            "category": "SCRIPT",
            "name": "Script scene analysis",
            "query": "Describe the red pill blue pill scene in The Matrix",
            "expected_tool": "search_book (matrix)",
            "expected_books": ["matrix"],
            "notes": "Should search for 'red pill blue pill Morpheus choice' and return scene text"
        }
    ]

    print("=" * 80)
    print("DOCUMENT TYPE TESTS")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}\n")

    # Group by category
    categories = {}
    for test in test_cases:
        cat = test['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(test)

    passed = 0
    errors = 0

    for category, tests in categories.items():
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {category} ({len(tests)} tests)")
        print("=" * 80)

        for i, test in enumerate(tests, 1):
            print(f"\n[{category}-{i}] {test['name']}")
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
    asyncio.run(test_document_types())
