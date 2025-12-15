"""
Comprehensive test suite for Book Mate agent tool calling behavior.

Tests cover:
1. Basic tool calling (single book, multi-book, summaries)
2. Edge cases (unavailable authors, contextual queries, chapter-specific)
3. Demo queries (showing Book Mate's advantages over foundation models)
"""
import asyncio
import json
import os
from src.mcp_client.agent import BookMateAgent


async def test_queries():
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

    # Track test results
    test_results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "by_category": {},
        "failures": []
    }

    test_cases = [
        # ===== BASIC TOOL CALLING TESTS =====
        {
            "category": "BASIC",
            "name": "Single book query",
            "query": "What is The Iliad about?",
            "expected_tool": "get_book_summary or search_book",
            "expected_books": ["ili"],
            "notes": "Should call tool for book in library"
        },
        {
            "category": "BASIC",
            "name": "Comparative query (2 books)",
            "query": "compare The Iliad with The Odyssey on heroism",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody"],
            "notes": "MUST use search_multiple_books, NOT two search_book calls"
        },
        {
            "category": "BASIC",
            "name": "Author with multiple books",
            "query": "summarize Homer's work",
            "expected_tool": "get_book_summary for each book",
            "expected_books": ["ili", "ody"],
            "notes": "Should find ALL books by Homer (both Iliad and Odyssey)"
        },
        {
            "category": "BASIC",
            "name": "Book summary query",
            "query": "What are the main themes in Meditations?",
            "expected_tool": "get_book_summary",
            "expected_books": ["mam"],
            "notes": "Should call get_book_summary for overview questions"
        },
        {
            "category": "BASIC",
            "name": "Specific content search",
            "query": "Find passages about wisdom in The Odyssey",
            "expected_tool": "search_book",
            "expected_books": ["ody"],
            "notes": "Should use search_book for finding specific content"
        },

        # ===== EDGE CASES WE FIXED =====
        {
            "category": "EDGE CASE",
            "name": "Unavailable author (no substitution)",
            "query": "compare Marcus Aurelius and Jordan Peterson on ethics",
            "expected_tool": "search_book (mam only)",
            "expected_books": ["mam"],
            "notes": "Should NOT search Hegel or other philosophy books as Peterson substitute"
        },
        {
            "category": "EDGE CASE",
            "name": "Comparative with author expansion",
            "query": "compare Homer with Lewis Carroll on character arcs",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody", "alice"],
            "notes": "Homer → both ili and ody, Carroll → alice, use search_multiple_books"
        },
        {
            "category": "EDGE CASE",
            "name": "Contextual query (after discussing Alice)",
            "query": "who is the Cheshire Cat?",
            "expected_tool": "search_book (alice)",
            "expected_books": ["alice"],
            "notes": "Should infer book from conversation context (test in multi-turn chat)",
            "context": [{"role": "user", "content": "Tell me about Alice in Wonderland"}]
        },
        {
            "category": "EDGE CASE",
            "name": "Chapter-specific query",
            "query": "What does Marcus say in Chapter 3 about anger?",
            "expected_tool": "search_book (mam)",
            "expected_books": ["mam"],
            "notes": "Should search broadly, then filter results by Chapter 3"
        },
        {
            "category": "EDGE CASE",
            "name": "Collection-wide query",
            "query": "what themes connect all books in my collection?",
            "expected_tool": "get_book_summary for multiple books",
            "expected_books": ["multiple"],
            "notes": "Should call tools for each/multiple books, not use general knowledge"
        },
        {
            "category": "EDGE CASE",
            "name": "Ambiguous character reference",
            "query": "Tell me about the Mad Hatter",
            "expected_tool": "search_book (alice)",
            "expected_books": ["alice"],
            "notes": "Should infer Alice from character name and search that book"
        },
        {
            "category": "EDGE CASE",
            "name": "Three-way comparison",
            "query": "Compare views on virtue in The Iliad, The Odyssey, and Meditations",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody", "mam"],
            "notes": "Should use search_multiple_books with all three book slugs"
        },

        # ===== DEMO QUERIES (Book Mate advantages) =====
        {
            "category": "DEMO",
            "name": "Specific passage search with citations",
            "query": "Find passages where Marcus Aurelius discusses death and acceptance",
            "expected_tool": "search_book (mam)",
            "expected_books": ["mam"],
            "notes": "Should return actual passages with [Chapter X, Source: chunk_id] citations"
        },
        {
            "category": "DEMO",
            "name": "Multi-book thematic comparison",
            "query": "How do The Iliad, The Odyssey, and Meditations portray the concept of fate?",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody", "mam"],
            "notes": "Should cite passages from ALL three books in comparison"
        },
        {
            "category": "DEMO",
            "name": "Precise author comparison",
            "query": "Compare how Homer and Marcus Aurelius view glory and honor",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody", "mam"],
            "notes": "Homer → both books, cited passages from actual texts"
        },
        {
            "category": "DEMO",
            "name": "Chapter-by-chapter analysis",
            "query": "Give me a chapter-by-chapter summary of Meditations",
            "expected_tool": "get_chapter_summaries (mam)",
            "expected_books": ["mam"],
            "notes": "Should return structured chapter summaries from actual content"
        },
        {
            "category": "DEMO",
            "name": "Verify quote authenticity",
            "query": "Does Marcus Aurelius actually say 'You have power over your mind - not outside events'?",
            "expected_tool": "search_book (mam)",
            "expected_books": ["mam"],
            "notes": "Should search and verify with exact passage citation if found"
        },
        {
            "category": "DEMO",
            "name": "Character analysis across books",
            "query": "Compare Achilles in The Iliad with Odysseus in The Odyssey",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody"],
            "notes": "Should cite specific passages showing character traits from each book"
        },
        {
            "category": "DEMO",
            "name": "Thematic deep dive",
            "query": "What does Alice in Wonderland reveal about logic, absurdity, and rules?",
            "expected_tool": "search_book (alice)",
            "expected_books": ["alice"],
            "notes": "Should return relevant passages demonstrating these themes with citations"
        },
        {
            "category": "DEMO",
            "name": "Cross-genre comparison",
            "query": "How do ancient epics (Homer) differ from philosophical texts (Marcus Aurelius) in their treatment of human suffering?",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "ody", "mam"],
            "notes": "Should search all three books and provide comparative analysis with citations"
        },
        {
            "category": "DEMO",
            "name": "Complex thematic search",
            "query": "How do The Iliad and Meditations address the relationship between duty and personal desire?",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["ili", "mam"],
            "notes": "Should craft query with concrete terms like 'duty obligation responsibility desire wants conflict' and cite both books"
        },
        {
            "category": "DEMO",
            "name": "Historical context query",
            "query": "What does The Odyssey tell us about ancient Greek hospitality customs?",
            "expected_tool": "search_book",
            "expected_books": ["ody"],
            "notes": "Should search for 'hospitality guests hosts customs traditions' with citations"
        }
    ]

    print("=" * 90)
    print("BOOK MATE AGENT - COMPREHENSIVE TOOL CALLING TEST SUITE")
    print("=" * 90)
    print(f"\nTotal tests: {len(test_cases)}")

    # Group by category
    categories = {}
    for test in test_cases:
        cat = test['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(test)

    print(f"Categories: {', '.join(categories.keys())}")
    print("=" * 90)

    for category, tests in categories.items():
        print(f"\n{'=' * 90}")
        print(f"CATEGORY: {category} ({len(tests)} tests)")
        print("=" * 90)

        # Initialize category stats
        if category not in test_results["by_category"]:
            test_results["by_category"][category] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            }

        for i, test in enumerate(tests, 1):
            test_results["total"] += 1
            test_results["by_category"][category]["total"] += 1

            print(f"\n[{category}-{i}] {test['name']}")
            print(f"Query: '{test['query']}'")
            print(f"Expected tool: {test['expected_tool']}")
            print(f"Expected books: {test['expected_books']}")
            print(f"Notes: {test['notes']}")
            print("-" * 90)

            try:
                # For contextual queries, we need multi-turn conversation
                # For now, just test with single query
                # TODO: Add multi-turn support when needed

                # Get response (tool calls will appear in logs)
                response_text, _, _ = await agent.chat(test['query'])

                print(f"✓ Response received")
                print(f"First 200 chars: {response_text[:200]}...")
                print("-" * 90)

                # Mark as passed (manual verification needed from logs)
                test_results["passed"] += 1
                test_results["by_category"][category]["passed"] += 1

            except Exception as e:
                print(f"✗ Error: {e}")
                print("-" * 90)

                # Mark as error
                test_results["errors"] += 1
                test_results["by_category"][category]["errors"] += 1
                test_results["failures"].append({
                    "category": category,
                    "name": test['name'],
                    "query": test['query'],
                    "error": str(e)
                })

            # Small delay between tests
            await asyncio.sleep(1)

    print("\n" + "=" * 90)
    print("TEST SUMMARY")
    print("=" * 90)

    # Overall statistics
    print(f"\nOVERALL RESULTS:")
    print(f"  Total Tests:    {test_results['total']}")
    print(f"  Passed:         {test_results['passed']} ({test_results['passed'] * 100 // test_results['total'] if test_results['total'] > 0 else 0}%)")
    print(f"  Errors:         {test_results['errors']}")

    # Category breakdown
    print(f"\nRESULTS BY CATEGORY:")
    for category, stats in test_results["by_category"].items():
        pass_rate = stats['passed'] * 100 // stats['total'] if stats['total'] > 0 else 0
        print(f"  {category:12} - {stats['passed']}/{stats['total']} passed ({pass_rate}%), {stats['errors']} errors")

    # Error details
    if test_results["errors"] > 0:
        print(f"\nERROR DETAILS:")
        for failure in test_results["failures"]:
            print(f"  [{failure['category']}] {failure['name']}")
            print(f"    Query: {failure['query']}")
            print(f"    Error: {failure['error']}")
            print()

    print("\n" + "=" * 90)
    print("MANUAL REVIEW CHECKLIST (Check agent logs above)")
    print("=" * 90)
    print("\nCRITICAL ISSUES TO VERIFY:")
    print("1. Comparative queries - Did agent use search_multiple_books (ONE call)?")
    print("   - NOT multiple search_book calls")
    print("   - Examples: [BASIC-2], [EDGE CASE-2], [DEMO-2], [DEMO-3], [DEMO-6], [DEMO-8]")
    print()
    print("2. Author expansion - Did agent find ALL books by author?")
    print("   - Homer → BOTH Iliad AND Odyssey")
    print("   - Examples: [BASIC-3], [EDGE CASE-2], [DEMO-3]")
    print()
    print("3. Book substitution - Did agent avoid searching unavailable books?")
    print("   - User asks about Peterson (not in library)")
    print("   - Should NOT search Hegel or other philosophy books")
    print("   - Should state \"not in library\" and search ONLY Marcus Aurelius")
    print("   - Example: [EDGE CASE-1]")
    print()
    print("4. Slug formatting - Did agent use slugs correctly?")
    print("   - Should use: 'mam' (just the slug)")
    print("   - Should NOT use: '[mam]' or 'Meditations'")
    print("   - Check all tool calls in logs")
    print()
    print("5. Citations - Did responses include proper citations?")
    print("   - Format: [Chapter X, Source: chunk_id]")
    print("   - For comparative queries: citations from ALL books searched")
    print()
    print("6. Query crafting - Did searches use concrete terms?")
    print("   - Should transform: 'character arcs' → 'character development growth transformation'")
    print("   - Should avoid meta-words: 'compare', 'similarities', 'differences'")
    print("=" * 90)

    # Cleanup - properly close the MCP session
    print("\n✓ Tests complete. Review logs above for detailed tool call behavior.")
    print("✓ Closing MCP session...")

    try:
        # Close in reverse order: session first, then stdio context
        if hasattr(agent, 'session') and agent.session:
            await agent.session.__aexit__(None, None, None)
            agent.session = None

        if hasattr(agent, 'stdio_context') and agent.stdio_context:
            await agent.stdio_context.__aexit__(None, None, None)
            agent.stdio_context = None

        print("✓ MCP session closed successfully.")
    except Exception as e:
        # Suppress expected cleanup exceptions
        if "cancel scope" not in str(e).lower():
            print(f"Note: Cleanup exception: {type(e).__name__}: {e}")

    print("✓ Done.")


if __name__ == "__main__":
    asyncio.run(test_queries())
