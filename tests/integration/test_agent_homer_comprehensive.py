"""
Comprehensive tests for Homer's Epics (Iliad & Odyssey).
Focuses on deep graph relationships, comparative analysis, and thematic exploration.
"""
import asyncio
import os
import sys

sys.path.append(os.getcwd())
from src.mcp_client.agent import BookMateAgent

async def test_homer_comprehensive():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    agent = BookMateAgent(openai_api_key=api_key)
    
    test_cases = [
        # 1. Deep Graph / Lineage
        {
            "name": "Telemachus Lineage",
            "query": "Who are the parents of Telemachus?",
            "expected_slugs": ["ody"],
            "keywords": ["Odysseus", "Penelope"]
        },
        {
            "name": "Gods' Involvement",
            "query": "Which goddess favors Odysseus in The Odyssey?",
            "expected_slugs": ["ody"],
            "keywords": ["Athena"]
        },
        
        # 2. Comparative Graph (Cross-Doc Entities)
        {
            "name": "Menelaus Cross-over",
            "query": "Compare the role of Menelaus in The Iliad vs The Odyssey",
            "expected_intent": "compare",
            "expected_slugs": ["ili", "ody"],
            "keywords": ["Sparta", "Helen", "Paris", "guest"]
        },
        {
            "name": "Achilles' Fate",
            "query": "Does Achilles appear in The Odyssey? If so, where?",
            "expected_slugs": ["ody", "ili"],
            "keywords": ["underworld", "hades", "dead"]
        },

        # 3. Thematic Comparison
        {
            "name": "Heroism Contrast",
            "query": "Contrast the heroism of Achilles with the heroism of Odysseus",
            "expected_strategy": "hybrid",
            "keywords": ["strength", "wit", "cunning", "guile", "force"]
        },
        
        # 4. Specific Event Retrieval
        {
            "name": "The Trojan Horse",
            "query": "Is the Trojan Horse described in The Iliad?",
            "expected_slugs": ["ili"],
            "keywords": ["no", "not", "odyssey", "briefly"]  # Trick question: It's mostly in Odyssey/Aeneid
        }
    ]

    print("\n" + "="*80)
    print("HOMER COMPREHENSIVE TESTS (GRAPH & COMPARISON)")
    print("="*80)

    passed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test['name']}")
        print(f"Query: '{test['query']}'")
        
        try:
            response, _, _, _ = await agent.chat(test['query'])
            print(f"Response (preview): {response[:200]}...")
            
            content_lower = response.lower()
            found = [k for k in test['keywords'] if k.lower() in content_lower]
            
            if len(found) > 0:
                print(f"✓ Keywords found: {found}")
                passed += 1
            else:
                print(f"✗ Missing keywords. Expected one of: {test['keywords']}")
                
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "="*80)
    print(f"Summary: {passed}/{len(test_cases)} Passed")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_homer_comprehensive())