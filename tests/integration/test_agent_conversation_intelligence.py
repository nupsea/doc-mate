"""
Integration tests for Conversation Intelligence.
Verifies context expansion, relationship tracking, and type-aware formatting.
"""
import asyncio
import os
import sys

sys.path.append(os.getcwd())
from src.mcp_client.agent import BookMateAgent

async def test_conversation_intelligence():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    agent = BookMateAgent(openai_api_key=api_key)
    
    test_cases = [
        # 1. Office Chat - Technical Troubleshooting
        {
            "name": "Incident Cause",
            "query": "What caused the login failures in EU-West-1 and who fixed it?",
            "expected_slugs": ["office_chat"],
            "keywords": ["security", "Jen", "port 6379", "Redis"]
        },
        {
            "name": "Decision Context Expansion",
            "query": "Sarah mentioned a billing schema change. What was the solution and whose idea was it?",
            "expected_slugs": ["office_chat"],
            "keywords": ["middleware", "Jen", "multi-currency"]
        },
        
        # 2. Personal Gossip - Complex Relationships
        {
            "name": "The Breakup Trace",
            "query": "Why did Liam and Sophie break up and how is Noah involved?",
            "expected_slugs": ["gossip_chat"],
            "keywords": ["texting", "Ava", "Noah"]
        },
        {
            "name": "Deep Relationship Link",
            "query": "Who is Mia dating, and how is her partner related to the Liam/Sophie drama?",
            "expected_slugs": ["gossip_chat"],
            "keywords": ["Ben", "Noah", "brother"]
        },

        # 3. Aggregator & Persona Check
        {
            "name": "Type-Aware Transcript Persona",
            "query": "Summarize the major decisions made in the Office Incident chat.",
            "expected_slugs": ["office_chat"],
            "keywords": ["Redis", "Billing", "GDPR"]
        }
    ]

    print("\n" + "="*80)
    print("CONVERSATION INTELLIGENCE TESTS")
    print("="*80)

    passed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test['name']}")
        print(f"Query: '{test['query']}'")
        
        try:
            response, _, _, _ = await agent.chat(test['query'])
            print(f"Response (preview): {response[:250]}...")
            
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
    asyncio.run(test_conversation_intelligence())