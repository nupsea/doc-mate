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
    # =========================================================================
    # GRAPH / RELATIONSHIP queries -- Iliad only (re-ingested with graph layer)
    # =========================================================================

    # --- Relationship between two characters ---
    {
        "query": "How are Achilles and Hector related in the story?",
        "doc": "ili",
        "desc": "GRAPH: Achilles-Hector relationship (enemies, duel)",
        "expect": ["Achilles", "Hector"],
        "reject": ["lovers"],
    },
    {
        "query": "What is the relationship between Achilles and Patroclus?",
        "doc": "ili",
        "desc": "GRAPH: Achilles-Patroclus bond (companions/friends)",
        "expect": ["Achilles", "Patroclus"],
        "reject": ["lovers"],
    },
    {
        "query": "How does Agamemnon's conflict with Achilles begin?",
        "doc": "ili",
        "desc": "GRAPH: Agamemnon-Achilles quarrel (Briseis / prize)",
        "expect": ["Agamemnon", "Achilles"],
        "reject": [],
    },
    # --- Entity role / identity ---
    {
        "query": "Who is Priam and what role does he play?",
        "doc": "ili",
        "desc": "GRAPH: Entity lookup - Priam (king of Troy, Hector's father)",
        "expect": ["Priam"],
        "reject": [],
    },
    {
        "query": "What role does Thetis play in the Iliad?",
        "doc": "ili",
        "desc": "GRAPH: Entity role - Thetis (Achilles' mother, divine)",
        "expect": ["Thetis"],
        "reject": [],
    },
    # --- Multi-entity / group queries ---
    {
        "query": "Which gods take sides in the Trojan War and whom do they support?",
        "doc": "ili",
        "desc": "GRAPH: Multi-entity - divine factions (Zeus, Athena, Apollo...)",
        "expect": [],
        "reject": [],
    },
    # --- Causal / event-chain ---
    {
        "query": "What chain of events leads from the quarrel to Patroclus's death?",
        "doc": "ili",
        "desc": "GRAPH: Causal chain (quarrel -> withdrawal -> Patroclus fights -> death)",
        "expect": ["Patroclus"],
        "reject": [],
    },
    {
        "query": "Why does Achilles return to battle?",
        "doc": "ili",
        "desc": "GRAPH: Motivation / cause (Patroclus killed -> rage -> returns)",
        "expect": ["Achilles"],
        "reject": [],
    },
    # --- Negative: relationship question but wrong book ---
    {
        "query": "How are Achilles and Hector related?",
        "doc": None,
        "desc": "NEGATIVE: Relationship query with no doc selected",
        "expect": [],
        "reject": [],
    },
    # --- Edge: broad thematic relationship ---
    {
        "query": "How does the theme of honour connect Achilles, Agamemnon, and Hector?",
        "doc": "ili",
        "desc": "EDGE: Thematic relationship across three characters",
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
            response, _, _ = await agent.chat(
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
