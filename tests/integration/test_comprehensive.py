"""
Comprehensive integration test for Doc-Mate.

Tests all main functionality in a single test run:
1. MCP server connection
2. Document availability
3. Single document search (search_book)
4. Document summary (get_book_summary)
5. Chapter/section summaries (get_chapter_summaries)
6. Multi-document search (search_multiple_books)
7. Multi-author search with author expansion (CRITICAL)
8. Different document types (book, conversation, tech_doc)
9. Privacy mode (local LLM)

Usage:
    # Test with OpenAI (default)
    python -m tests.integration.test_comprehensive

    # Test with local LLM
    LLM_PROVIDER=local python -m tests.integration.test_comprehensive
"""
import asyncio
import os
import sys
from src.mcp_client.agent import BookMateAgent


async def run_comprehensive_test():
    """Run comprehensive integration test covering all main functionality."""

    # Determine which provider to use
    provider = os.getenv("LLM_PROVIDER", "openai")

    print("=" * 80)
    print("DOC-MATE COMPREHENSIVE INTEGRATION TEST")
    print("=" * 80)
    print(f"Provider: {provider}")
    print(f"Mode: {'Local LLM (Ollama)' if provider == 'local' else 'OpenAI API'}")
    print("=" * 80)
    print()

    # Initialize agent
    if provider == "local":
        print("[SETUP] Initializing agent with local LLM...")
        agent = BookMateAgent(provider="local")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set")
            print("Set it or run with: LLM_PROVIDER=local python -m tests.integration.test_comprehensive")
            return False
        print(f"[SETUP] Initializing agent with OpenAI API...")
        agent = BookMateAgent(openai_api_key=api_key)

    results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }

    # Test 1: MCP Server Connection
    print("\n" + "=" * 80)
    print("TEST 1: MCP Server Connection")
    print("=" * 80)
    try:
        await agent.connect_to_mcp_server()
        print("✓ PASS: Successfully connected to MCP server")
        print(f"  Tools available: {len(agent.tools_cache)}")
        results["passed"] += 1
    except Exception as e:
        print(f"✗ FAIL: Could not connect to MCP server: {e}")
        results["failed"] += 1
        results["errors"].append(f"MCP Connection: {e}")
        await agent.close()
        return False  # Can't continue without MCP connection

    # Test 2: Single Document Search
    print("\n" + "=" * 80)
    print("TEST 2: Single Document Search (search_book)")
    print("=" * 80)
    print("Query: 'What does The Odyssey say about wisdom?'")
    print("Expected: Should call search_book with slug 'ody'")
    try:
        response, _, _ = await agent.chat("What does The Odyssey say about wisdom?")
        if len(response) > 50 and "ody" in response.lower():
            print("✓ PASS: Single document search working")
            print(f"  Response length: {len(response)} chars")
            print(f"  Preview: {response[:150]}...")
            results["passed"] += 1
        else:
            print(f"✗ FAIL: Response too short or doesn't mention Odyssey")
            print(f"  Response: {response}")
            results["failed"] += 1
            results["errors"].append("Single document search: Invalid response")
    except Exception as e:
        print(f"✗ FAIL: Error in single document search: {e}")
        results["failed"] += 1
        results["errors"].append(f"Single document search: {e}")

    await asyncio.sleep(1)

    # Test 3: Document Summary
    print("\n" + "=" * 80)
    print("TEST 3: Document Summary (get_book_summary)")
    print("=" * 80)
    print("Query: 'What is Meditations about?'")
    print("Expected: Should call get_book_summary for slug 'mam'")
    try:
        response, _, _ = await agent.chat("What is Meditations about?")
        if len(response) > 100:
            print("✓ PASS: Document summary working")
            print(f"  Response length: {len(response)} chars")
            print(f"  Preview: {response[:150]}...")
            results["passed"] += 1
        else:
            print(f"✗ FAIL: Summary response too short")
            print(f"  Response: {response}")
            results["failed"] += 1
            results["errors"].append("Document summary: Response too short")
    except Exception as e:
        print(f"✗ FAIL: Error getting document summary: {e}")
        results["failed"] += 1
        results["errors"].append(f"Document summary: {e}")

    await asyncio.sleep(1)

    # Test 4: Chapter Summaries
    print("\n" + "=" * 80)
    print("TEST 4: Chapter/Section Summaries (get_chapter_summaries)")
    print("=" * 80)
    print("Query: 'Give me a chapter breakdown of The Iliad'")
    print("Expected: Should call get_chapter_summaries for slug 'ili'")
    try:
        response, _, _ = await agent.chat("Give me a chapter breakdown of The Iliad")
        if len(response) > 200 and ("chapter" in response.lower() or "book" in response.lower()):
            print("✓ PASS: Chapter summaries working")
            print(f"  Response length: {len(response)} chars")
            print(f"  Preview: {response[:150]}...")
            results["passed"] += 1
        else:
            print(f"✗ FAIL: Chapter summaries response invalid")
            print(f"  Response: {response}")
            results["failed"] += 1
            results["errors"].append("Chapter summaries: Invalid response")
    except Exception as e:
        print(f"✗ FAIL: Error getting chapter summaries: {e}")
        results["failed"] += 1
        results["errors"].append(f"Chapter summaries: {e}")

    await asyncio.sleep(1)

    # Test 5: Multi-Document Search (Two Specific Books)
    print("\n" + "=" * 80)
    print("TEST 5: Multi-Document Search (search_multiple_books)")
    print("=" * 80)
    print("Query: 'Compare The Iliad and The Odyssey on heroism'")
    print("Expected: Should call search_multiple_books with ['ili', 'ody'] in ONE call")
    try:
        response, _, _ = await agent.chat("Compare The Iliad and The Odyssey on heroism")
        # Check if response mentions both books
        has_iliad = "iliad" in response.lower()
        has_odyssey = "odyssey" in response.lower()
        if len(response) > 200 and has_iliad and has_odyssey:
            print("✓ PASS: Multi-document search working")
            print(f"  Response length: {len(response)} chars")
            print(f"  Mentions Iliad: {has_iliad}, Mentions Odyssey: {has_odyssey}")
            print(f"  Preview: {response[:150]}...")
            results["passed"] += 1
        else:
            print(f"✗ FAIL: Multi-document search response incomplete")
            print(f"  Has Iliad: {has_iliad}, Has Odyssey: {has_odyssey}")
            print(f"  Response: {response[:300]}...")
            results["failed"] += 1
            results["errors"].append("Multi-document search: Incomplete response")
    except Exception as e:
        print(f"✗ FAIL: Error in multi-document search: {e}")
        results["failed"] += 1
        results["errors"].append(f"Multi-document search: {e}")

    await asyncio.sleep(1)

    # Test 6: CRITICAL - Multi-Author Search with Author Expansion
    print("\n" + "=" * 80)
    print("TEST 6: Multi-Author Search with Author Expansion (CRITICAL)")
    print("=" * 80)
    print("Query: 'Compare Marcus Aurelius and Homer on bravery'")
    print("Expected: Should find Marcus=mam, Homer=ili+ody")
    print("Expected: Should call search_multiple_books(['mam', 'ili', 'ody']) in ONE call")
    print("THIS IS THE CRITICAL TEST - Must search ALL 3 books together")
    try:
        response, _, _ = await agent.chat("Compare Marcus Aurelius and Homer on bravery")
        # Check if response mentions all three works
        has_meditations = "meditation" in response.lower() or "marcus" in response.lower()
        has_iliad = "iliad" in response.lower()
        has_odyssey = "odyssey" in response.lower()

        if len(response) > 200 and has_meditations and (has_iliad or has_odyssey):
            print("✓ PASS: Multi-author search with expansion working")
            print(f"  Response length: {len(response)} chars")
            print(f"  Mentions Marcus/Meditations: {has_meditations}")
            print(f"  Mentions Iliad: {has_iliad}")
            print(f"  Mentions Odyssey: {has_odyssey}")
            print(f"  Preview: {response[:150]}...")
            results["passed"] += 1
        else:
            print(f"✗ FAIL: Multi-author search incomplete - did not search all books")
            print(f"  Has Marcus/Meditations: {has_meditations}")
            print(f"  Has Iliad: {has_iliad}")
            print(f"  Has Odyssey: {has_odyssey}")
            print(f"  Response: {response[:300]}...")
            results["failed"] += 1
            results["errors"].append("Multi-author search: Did not search all author's books")
    except Exception as e:
        print(f"✗ FAIL: Error in multi-author search: {e}")
        results["failed"] += 1
        results["errors"].append(f"Multi-author search: {e}")

    await asyncio.sleep(1)

    # Test 7: Different Document Type - Conversation
    print("\n" + "=" * 80)
    print("TEST 7: Conversation Document Type")
    print("=" * 80)
    print("Query: 'What was discussed in the sample meeting about Q1?'")
    print("Expected: Should search conversation document")
    try:
        response, _, _ = await agent.chat("What was discussed in the sample meeting about Q1?")
        if len(response) > 50:
            print("✓ PASS: Conversation document search working")
            print(f"  Response length: {len(response)} chars")
            print(f"  Preview: {response[:150]}...")
            results["passed"] += 1
        else:
            print(f"✗ FAIL: Conversation search response too short")
            print(f"  Response: {response}")
            results["failed"] += 1
            results["errors"].append("Conversation search: Response too short")
    except Exception as e:
        print(f"✗ FAIL: Error in conversation search: {e}")
        results["failed"] += 1
        results["errors"].append(f"Conversation search: {e}")

    await asyncio.sleep(1)

    # Test 8: Technical Documentation
    print("\n" + "=" * 80)
    print("TEST 8: Technical Documentation Type")
    print("=" * 80)
    print("Query: 'What does design data intensive apps say about replication?'")
    print("Expected: Should search technical documentation")
    try:
        response, _, _ = await agent.chat("What does design data intensive apps say about replication?")
        if len(response) > 50:
            print("✓ PASS: Technical documentation search working")
            print(f"  Response length: {len(response)} chars")
            print(f"  Preview: {response[:150]}...")
            results["passed"] += 1
        else:
            print(f"✗ FAIL: Tech doc search response too short")
            print(f"  Response: {response}")
            results["failed"] += 1
            results["errors"].append("Tech doc search: Response too short")
    except Exception as e:
        print(f"✗ FAIL: Error in tech doc search: {e}")
        results["failed"] += 1
        results["errors"].append(f"Tech doc search: {e}")

    # Cleanup
    await agent.close()

    # Print final results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']} ✓")
    print(f"Failed: {results['failed']} ✗")

    if results['failed'] > 0:
        print("\nFailed Tests:")
        for i, error in enumerate(results['errors'], 1):
            print(f"  {i}. {error}")

    print("=" * 80)

    # Return success if all tests passed
    success = results['failed'] == 0
    if success:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {results['failed']} TEST(S) FAILED")

    return success


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)
