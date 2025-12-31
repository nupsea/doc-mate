"""
Comprehensive edge case testing for BookMateAgent
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent


async def test_empty_query():
    """Test handling of empty queries"""
    print("="*80)
    print("TEST: Empty Query Handling")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    try:
        response, _, _ = await agent.chat("")
        await agent.close()

        # Should handle gracefully, not crash
        print(f"Response: {response[:200]}...")
        print("✓ PASSED: Empty query handled gracefully")
        return True
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception on empty query: {e}")
        return False


async def test_nonexistent_book():
    """Test searching for a book that doesn't exist"""
    print("\n" + "="*80)
    print("TEST: Non-existent Book Search")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    try:
        response, _, _ = await agent.chat("What does the book 'XYZ_NONEXISTENT_123' say about time?")
        await agent.close()

        # Should handle gracefully, possibly saying book not found
        print(f"Response: {response[:300]}...")
        print("✓ PASSED: Non-existent book handled gracefully")
        return True
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception on non-existent book: {e}")
        return False


async def test_very_long_query():
    """Test handling of very long queries"""
    print("\n" + "="*80)
    print("TEST: Very Long Query")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    # Create a very long query
    long_query = "What does Meditations say about " + " and ".join(
        ["virtue", "wisdom", "courage", "justice", "temperance"] * 20
    ) + "?"

    try:
        response, _, _ = await agent.chat(long_query)
        await agent.close()

        print(f"Query length: {len(long_query)} characters")
        print(f"Response length: {len(response)} characters")
        print("✓ PASSED: Long query handled")
        return True
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception on long query: {e}")
        return False


async def test_multi_turn_conversation():
    """Test multi-turn conversation with context"""
    print("\n" + "="*80)
    print("TEST: Multi-turn Conversation")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    try:
        # First turn
        response1, _, _ = await agent.chat("What is the Iliad about?")
        print(f"Turn 1 response length: {len(response1)}")

        # Second turn - follow-up question
        response2, _, _ = await agent.chat("Who are the main characters?")
        print(f"Turn 2 response length: {len(response2)}")

        # Third turn - another follow-up
        response3, _, _ = await agent.chat("What happens to Achilles?")
        print(f"Turn 3 response length: {len(response3)}")

        await agent.close()

        # All should have reasonable responses
        all_have_content = all(len(r) > 50 for r in [response1, response2, response3])

        if all_have_content:
            print("✓ PASSED: Multi-turn conversation handled")
            return True
        else:
            print("✗ FAILED: Some responses too short")
            return False
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception in multi-turn: {e}")
        return False


async def test_special_characters_query():
    """Test queries with special characters"""
    print("\n" + "="*80)
    print("TEST: Special Characters in Query")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    try:
        # Query with special characters
        response, _, _ = await agent.chat("What does the Gita say about 'dharma' & duty?")
        await agent.close()

        print(f"Response length: {len(response)}")
        print("✓ PASSED: Special characters handled")
        return True
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception with special characters: {e}")
        return False


async def test_concurrent_searches():
    """Test multiple searches in quick succession"""
    print("\n" + "="*80)
    print("TEST: Concurrent/Quick Searches")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    try:
        # Fire multiple searches quickly
        queries = [
            "What is virtue in Meditations?",
            "What is the main theme of the Iliad?",
            "What does the Odyssey teach about homecoming?"
        ]

        results = []
        for query in queries:
            response, _, _ = await agent.chat(query)
            results.append(response)
            print(f"Query '{query[:30]}...' -> Response length: {len(response)}")

        await agent.close()

        # All should have responses
        all_have_content = all(len(r) > 50 for r in results)

        if all_have_content:
            print("✓ PASSED: Concurrent searches handled")
            return True
        else:
            print("✗ FAILED: Some responses missing")
            return False
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception in concurrent searches: {e}")
        return False


async def test_ambiguous_book_reference():
    """Test queries with ambiguous book references"""
    print("\n" + "="*80)
    print("TEST: Ambiguous Book Reference")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    try:
        # Ambiguous - could be multiple books
        response, _, _ = await agent.chat("What do the ancient Greeks say about heroism?")
        await agent.close()

        print(f"Response length: {len(response)}")
        # Should either search multiple books or ask for clarification
        print("✓ PASSED: Ambiguous reference handled")
        return True
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception with ambiguous reference: {e}")
        return False


async def test_json_injection_attempt():
    """Test query that looks like JSON (security)"""
    print("\n" + "="*80)
    print("TEST: JSON Injection Attempt")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    try:
        # Query that looks like JSON
        response, _, _ = await agent.chat('{"query": "hack", "book": "all"}')
        await agent.close()

        # Should treat as regular text query, not crash
        print(f"Response length: {len(response)}")
        print("✓ PASSED: JSON-like query handled safely")
        return True
    except Exception as e:
        await agent.close()
        print(f"✗ FAILED: Exception with JSON-like query: {e}")
        return False


async def main():
    results = []

    results.append(await test_empty_query())
    results.append(await test_nonexistent_book())
    results.append(await test_very_long_query())
    results.append(await test_multi_turn_conversation())
    results.append(await test_special_characters_query())
    results.append(await test_concurrent_searches())
    results.append(await test_ambiguous_book_reference())
    results.append(await test_json_injection_attempt())

    print("\n" + "="*80)
    print("EDGE CASES TEST SUMMARY")
    print("="*80)
    print(f"Passed: {sum(results)}/{len(results)}")
    print("="*80)

    return all(results)


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
