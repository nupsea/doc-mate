"""
Comprehensive test suite for Book Mate agent tool calling behavior.

Tests cover:
1. Basic tool calling (single book, multi-book, summaries)
2. Edge cases (unavailable authors, contextual queries, chapter-specific)
3. Demo queries (showing Book Mate's advantages over foundation models)
"""
import asyncio
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
            "name": "Comparative query (2 authors - critical test)",
            "query": "compare Marcus and Homer on bravery",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["mam", "ili", "ody"],
            "notes": "CRITICAL: Must include ALL 3 books (Marcus=mam, Homer=ili+ody) in ONE search_multiple_books call"
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
        },

        # ===== CONVERSATION DOCUMENTS =====
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
            "name": "Speaker attribution WITHOUT labels (negative test)",
            "query": "What does Anup say in Dec Town Hall?",
            "expected_tool": "search_book (dth)",
            "expected_books": ["dth"],
            "notes": "CRITICAL: Must respond ONLY with 'This document lacks speaker labels. I cannot determine who said what.' NO topics, NO inference"
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

        # ===== TECHNICAL DOCUMENTS =====
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
            "name": "Compare tech concepts across books",
            "query": "compare how the Gita and design data intensive apps approach balance and trade-offs",
            "expected_tool": "search_multiple_books (ONE call)",
            "expected_books": ["gita", "ddia"],
            "notes": "Cross-genre comparison: philosophical vs technical text"
        },

        # ===== MOVIE SCRIPTS =====
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
        },

        # ===== NEGATIVE TESTS =====
        {
            "category": "NEGATIVE",
            "name": "Author not in library",
            "query": "What does Stephen King say about fear?",
            "expected_tool": "None (should state unavailable)",
            "expected_books": [],
            "notes": "Should respond: 'Stephen King is not in the library.' DO NOT substitute with other authors"
        },
        {
            "category": "NEGATIVE",
            "name": "Book not in library",
            "query": "Tell me about Harry Potter",
            "expected_tool": "None (should state unavailable)",
            "expected_books": [],
            "notes": "Should respond that Harry Potter is not in library, NOT search similar books"
        },
        {
            "category": "NEGATIVE",
            "name": "Comparative with unavailable author",
            "query": "compare Marcus Aurelius and Viktor Frankl on suffering",
            "expected_tool": "search_book (mam only)",
            "expected_books": ["mam"],
            "notes": "Frankl not in library - should search ONLY Marcus, state Frankl unavailable"
        },
        {
            "category": "NEGATIVE",
            "name": "Ambiguous query without context",
            "query": "What happens in chapter 5?",
            "expected_tool": "None or clarification",
            "expected_books": [],
            "notes": "No book context - should ask user to clarify which book"
        },
        {
            "category": "NEGATIVE",
            "name": "Empty/nonsense query",
            "query": "xyz abc qwe",
            "expected_tool": "None",
            "expected_books": [],
            "notes": "Should handle gracefully, possibly ask for clarification"
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

                print("✓ Response received")
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
    print("\nOVERALL RESULTS:")
    print(f"  Total Tests:    {test_results['total']}")
    print(f"  Passed:         {test_results['passed']} ({test_results['passed'] * 100 // test_results['total'] if test_results['total'] > 0 else 0}%)")
    print(f"  Errors:         {test_results['errors']}")

    # Category breakdown
    print("\nRESULTS BY CATEGORY:")
    for category, stats in test_results["by_category"].items():
        pass_rate = stats['passed'] * 100 // stats['total'] if stats['total'] > 0 else 0
        print(f"  {category:12} - {stats['passed']}/{stats['total']} passed ({pass_rate}%), {stats['errors']} errors")

    # Error details
    if test_results["errors"] > 0:
        print("\nERROR DETAILS:")
        for failure in test_results["failures"]:
            print(f"  [{failure['category']}] {failure['name']}")
            print(f"    Query: {failure['query']}")
            print(f"    Error: {failure['error']}")
            print()

    print("\n" + "=" * 90)
    print("MANUAL REVIEW CHECKLIST (Check agent logs above)")
    print("=" * 90)
    print("\nCRITICAL ISSUES TO VERIFY:")
    print()
    print("1. [BASIC-3] Comparative query 2 authors - MOST CRITICAL TEST")
    print("   Query: 'compare Marcus and Homer on bravery'")
    print("   ✓ MUST call search_multiple_books(['mam', 'ili', 'ody']) in ONE call")
    print("   ✗ NOT: search_book('mam') + search_multiple_books(['ili', 'ody'])")
    print("   ✗ NOT: search_book('mam') + search_book('ili') + search_book('ody')")
    print()
    print("2. Comparative queries - Did agent use search_multiple_books (ONE call)?")
    print("   - NOT multiple search_book calls")
    print("   - Examples: [BASIC-2], [BASIC-3], [EDGE CASE-2], [CONVERSATION-4], [TECH_DOC-3]")
    print()
    print("3. Author expansion - Did agent find ALL books by author?")
    print("   - Homer → BOTH Iliad AND Odyssey")
    print("   - Examples: [BASIC-3], [BASIC-4], [EDGE CASE-2], [DEMO-3]")
    print()
    print("4. Speaker attribution WITHOUT labels - [CONVERSATION-2] CRITICAL")
    print("   Query: 'What does Anup say in Dec Town Hall?'")
    print("   ✓ Response: 'This document lacks speaker labels. I cannot determine who said what.'")
    print("   ✗ NOT: List topics or infer content")
    print()
    print("5. Speaker attribution WITH labels - [CONVERSATION-1]")
    print("   Query: 'What does Sarah Chen say in the sample meeting?'")
    print("   ✓ Citations should show: [Speakers: Sarah Chen, ...]")
    print()
    print("6. Book substitution - Did agent avoid searching unavailable books?")
    print("   - Examples: [EDGE CASE-1], [NEGATIVE-1], [NEGATIVE-2], [NEGATIVE-3]")
    print("   - Should state \"not in library\" and NOT substitute with other books")
    print()
    print("7. Slug formatting - Did agent use slugs correctly?")
    print("   - Should use: 'mam' (just the slug)")
    print("   - Should NOT use: '[mam]' or 'Meditations'")
    print("   - Check all tool calls in logs")
    print()
    print("8. Citations - Did responses include proper citations?")
    print("   - Format: [Chapter X, Source: chunk_id] or [Speakers: ..., Source: chunk_id]")
    print("   - For comparative queries: citations from ALL books searched")
    print()
    print("9. Cross-document type comparisons - [TECH_DOC-3]")
    print("   Query: 'compare Gita and DDIA on balance'")
    print("   ✓ Should work across philosophical and technical documents")
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
