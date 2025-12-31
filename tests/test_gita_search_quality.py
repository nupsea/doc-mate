"""
Test Gita search quality after re-ingestion
"""
import asyncio
import os
from src.mcp_client.agent import BookMateAgent


async def test_gita_search_arjuna():
    """Test that searching for Arjuna returns actual content, not front matter"""
    print("="*80)
    print("TEST: Gita Search Quality - Arjuna")
    print("="*80)

    # Use local LLM to avoid external API calls during testing
    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    # Search for Arjuna
    response, _, _ = await agent.chat("What does the Gita say about Arjuna?")

    await agent.close()

    # Check if response contains front matter keywords
    front_matter_keywords = ['preface', 'contents', 'table of contents',
                             'dedication', 'translator', 'introduction',
                             'gutenberg', 'produced by']

    has_front_matter = any(keyword in response.lower() for keyword in front_matter_keywords)

    # Check if response contains actual Gita content
    actual_content_keywords = ['krishna', 'battle', 'arjuna', 'dharma',
                               'warrior', 'duryodhana', 'pandava']

    has_actual_content = any(keyword in response.lower() for keyword in actual_content_keywords)

    print(f"\nResponse length: {len(response)} characters")
    print(f"Has front matter keywords: {has_front_matter}")
    print(f"Has actual content keywords: {has_actual_content}")
    print(f"\nFirst 500 chars of response:")
    print(response[:500])
    print("...")

    if not has_front_matter and has_actual_content:
        print("\n✓ PASSED: Search returns actual Gita content, not front matter")
        return True
    elif has_front_matter:
        print("\n✗ FAILED: Response still contains front matter")
        return False
    else:
        print("\n⚠️  WARNING: Response doesn't contain expected Gita content keywords")
        return False


async def test_gita_search_draupadi():
    """Test that searching for Draupadi (which doesn't exist) returns reasonable content"""
    print("\n" + "="*80)
    print("TEST: Gita Search Quality - Draupadi (not in Gita)")
    print("="*80)

    agent = BookMateAgent(provider="local", ephemeral=True)
    await agent.connect_to_mcp_server()

    # Search for Draupadi (doesn't exist in Gita)
    response, _, _ = await agent.chat("What is Draupadi's significance in the Gita?")

    await agent.close()

    # Should ideally say Draupadi is not mentioned, or return general content
    mentions_not_found = any(phrase in response.lower() for phrase in
                            ['not mentioned', 'not found', 'does not appear',
                             'not in the gita', 'not discussed'])

    # Should NOT return front matter
    front_matter_keywords = ['preface', 'contents', 'dedication', 'translator']
    has_front_matter = any(keyword in response.lower() for keyword in front_matter_keywords)

    print(f"\nResponse length: {len(response)} characters")
    print(f"Mentions not found: {mentions_not_found}")
    print(f"Has front matter: {has_front_matter}")
    print(f"\nFirst 500 chars of response:")
    print(response[:500])

    if not has_front_matter:
        print("\n✓ PASSED: No front matter in response")
        return True
    else:
        print("\n✗ FAILED: Response contains front matter")
        return False


async def main():
    results = []

    results.append(await test_gita_search_arjuna())
    results.append(await test_gita_search_draupadi())

    print("\n" + "="*80)
    print("GITA SEARCH QUALITY TEST SUMMARY")
    print("="*80)
    print(f"Passed: {sum(results)}/{len(results)}")
    print("="*80)

    return all(results)


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
