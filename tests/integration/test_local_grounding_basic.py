"""
Test script for pre-fetch grounding on local models.
Run inside container: docker exec -it bookmate_app python scripts/test_grounding.py
"""
import asyncio
import os

# Force local provider
os.environ.setdefault("LLM_PROVIDER", "local")
os.environ.setdefault("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")

from src.mcp_client.agent import DocMateAgent

# Test cases: (query, selected_doc, description, check_keywords)
# check_keywords: words that SHOULD appear in a correct answer
# negative_keywords: words that indicate hallucination
TEST_CASES = [
    # --- POSITIVE: Doc selected, should ground in retrieved text ---
    {
        "query": "Who kills Patroclus?",
        "doc": "ili",
        "desc": "POSITIVE: Direct fact lookup (Iliad)",
        "expect": ["Hector"],
        "reject": [],
    },
    {
        "query": "What does Achilles do after Patroclus dies?",
        "doc": "ili",
        "desc": "POSITIVE: Event sequence (Iliad)",
        "expect": [],  # flexible, but should mention grief/rage/armor
        "reject": ["Astyanax"],
    },
    {
        "query": "Who is the Cheshire Cat?",
        "doc": "aiw",
        "desc": "POSITIVE: Entity lookup (Alice in Wonderland)",
        "expect": ["cat", "grin"],
        "reject": [],
    },
    {
        "query": "What happens at the Mad Tea Party?",
        "doc": "aiw",
        "desc": "POSITIVE: Event lookup (Alice in Wonderland)",
        "expect": [],
        "reject": [],
    },
    # --- NEGATIVE: No doc selected, model must use tools ---
    {
        "query": "Tell me about quantum physics",
        "doc": None,
        "desc": "NEGATIVE: No doc, unrelated topic - should say not found or no doc",
        "expect": [],
        "reject": [],
    },
    # --- EDGE: Doc selected but query is vague ---
    {
        "query": "Tell me more about the main character",
        "doc": "ili",
        "desc": "EDGE: Vague follow-up with doc selected",
        "expect": [],
        "reject": [],
    },
]

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


async def run_tests():
    results = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {tc['desc']}")
        print(f"  Query: {tc['query']}")
        print(f"  Doc:   {tc['doc'] or '(none)'}")
        print(f"{'='*70}")

        agent = DocMateAgent(
            provider="local",
            model=os.getenv("OLLAMA_MODEL", "gpt-4o-mini"),
            ephemeral=True,
            internal_mode=True,
        )

        try:
            response, _, _, _ = await agent.chat(
                tc["query"],
                conversation_history=[],
                selected_doc=tc["doc"],
            )
        except Exception as e:
            response = f"ERROR: {e}"

        print(f"\n  RESPONSE:\n  {response[:500]}")

        # Check expected keywords
        status = PASS
        response_lower = response.lower()

        for kw in tc["expect"]:
            if kw.lower() not in response_lower:
                print(f"  [MISSING] Expected keyword '{kw}' not found")
                status = WARN

        for kw in tc["reject"]:
            if kw.lower() in response_lower:
                print(f"  [HALLUCINATION] Rejected keyword '{kw}' found!")
                status = FAIL

        if "error" in response_lower and tc["doc"]:
            print("  [ERROR] Response contains error")
            status = FAIL

        print(f"\n  STATUS: {status}")
        results.append((i, tc["desc"], status))

        await agent.close()

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for num, desc, status in results:
        print(f"  {status:4s}  Test {num}: {desc}")

    passed = sum(1 for _, _, s in results if s == PASS)
    warned = sum(1 for _, _, s in results if s == WARN)
    failed = sum(1 for _, _, s in results if s == FAIL)
    print(f"\n  {passed} passed, {warned} warnings, {failed} failed out of {len(results)} tests")


if __name__ == "__main__":
    asyncio.run(run_tests())
