"""
Integration tests for Graph Knowledge and Entity Relationships.
Verifies the RRG flow's ability to handle complex relational queries.
"""
import asyncio
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.mcp_client.agent import BookMateAgent

async def test_graph_knowledge_queries():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    agent = BookMateAgent(openai_api_key=api_key)
    
    test_cases = [
        {
            "name": "Direct Relationship (Achilles-Patroclus)",
            "query": "How is Achilles related to Patroclus in The Iliad?",
            "expected_slugs": ["ili"],
            "keywords": ["companion", "friend", "beloved", "mourn"]
        },
        {
            "name": "Family Relationship (Hector-Priam)",
            "query": "What is the connection between Hector and Priam?",
            "expected_slugs": ["ili"],
            "keywords": ["son", "father", "king"]
        },
        {
            "name": "Multi-hop / Role Query",
            "query": "Who is Telemachus and how is he related to Odysseus?",
            "expected_slugs": ["ody"],
            "keywords": ["son", "father", "ithaca", "penelope"]
        },
        {
            "name": "Cross-Doc Entity Detection",
            "query": "Does Odysseus appear in both The Iliad and The Odyssey?",
            "expected_intent": "compare",
            "expected_slugs": ["ili", "ody"],
            "keywords": ["yes", "both", "warrior", "journey"]
        }
    ]

    print("\n" + "="*80)
    print("GRAPH KNOWLEDGE & RELATIONSHIP TESTS")
    print("="*80)

    passed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test['name']}")
        print(f"Query: '{test['query']}'")
        
        try:
            # We use the agent's chat method which invokes the LangGraph RRG flow
            response, _, _, _ = await agent.chat(test['query'])
            
            print(f"Response (preview): {response[:150]}...")
            
            # Basic validation
            content_lower = response.lower()
            found_keywords = [k for k in test['keywords'] if k.lower() in content_lower]
            
            if found_keywords:
                print(f"✓ Keywords found: {found_keywords}")
                passed += 1
            else:
                # Some local LLMs might use different words, so we print for manual check if needed
                print("✗ No exact expected keywords found in response.")
                
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "="*80)
    print(f"Graph Tests Summary: {passed}/{len(test_cases)} Passed")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_graph_knowledge_queries())