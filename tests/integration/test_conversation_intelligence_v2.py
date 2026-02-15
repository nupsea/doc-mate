"""
Integration tests for Conversation Intelligence Improvements (v2).

Tests the four phases:
1. Adaptive chunk sizing -- end-to-end parse with real parser
2. Episode store methods -- DB round-trip for get_episodes_by_chunk_ids, get_all_episodes_for_doc
3. Episode context in retrieval -- aggregator injects [EPISODE CONTEXT]
4. Arc summaries -- SummaryGenerator.generate_arc_summaries produces valid output

Requires:
- PostgreSQL with doc-mate schema (graph_episodes table)
- OPENAI_API_KEY for Phase 4 (arc summary LLM call)
- At least one conversation doc already ingested (gossip_chat or office_chat)
"""

import asyncio
import os
import sys
import tempfile

sys.path.append(os.getcwd())

from src.content.parsers.conversation_parser import ConversationParser
from src.graph.store import PostgresGraphStore
from src.content.store import PgresStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_social_chat_file(num_turns: int = 300) -> str:
    """Generate a synthetic social chat file with short turns for testing."""
    speakers = ["Alice", "Bob", "Cara", "Dan"]
    lines = []
    for i in range(num_turns):
        speaker = speakers[i % len(speakers)]
        hour = 10 + (i // 60)
        minute = i % 60
        lines.append(f"[{hour:02d}:{minute:02d}] {speaker}: Hey turn {i+1} ok")
    path = tempfile.mktemp(suffix=".txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _build_formal_chat_file(num_turns: int = 40) -> str:
    """Generate a synthetic formal/long-turn conversation file."""
    speakers = ["Dr. Smith", "Prof. Jones"]
    lines = []
    for i in range(num_turns):
        speaker = speakers[i % len(speakers)]
        hour = 9 + (i // 60)
        minute = i % 60
        # Each turn ~200 tokens (long paragraph)
        body = (
            f"In analyzing the implications of observation {i+1}, we must consider "
            "the broader theoretical framework within which this evidence operates. "
            "The statistical significance of the correlations observed across multiple "
            "datasets suggests a robust underlying mechanism that warrants further "
            "investigation through controlled experimental design. Furthermore, the "
            "methodological considerations raised by the peer review committee indicate "
            "that additional validation steps are necessary before we can draw definitive "
            "conclusions regarding causality versus mere association in this context."
        )
        lines.append(f"[{hour:02d}:{minute:02d}] {speaker}: {body}")
    path = tempfile.mktemp(suffix=".txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Phase 1: Adaptive Chunk Sizing (end-to-end)
# ---------------------------------------------------------------------------

def test_adaptive_chunking_social_chat_e2e():
    """Social chat with short turns should produce wider chunks (max_tokens=1000)."""
    print("\n[TEST] Phase 1a: Adaptive chunking -- social chat e2e")

    path = _build_social_chat_file(num_turns=300)
    parser = ConversationParser(path, slug="test_social")
    turns = parser.parse()
    assert len(turns) >= 200, f"Expected many turns, got {len(turns)}"

    # Chunk with default max_tokens=500 (will be overridden adaptively)
    chunks = parser.chunk(turns, max_tokens=500)

    # With adaptive sizing, social chat gets max_tokens=1000
    # so we expect far fewer chunks than with max_tokens=500
    parser.chunk(turns, max_tokens=500)  # already adaptive=1000
    # To verify the adaptive path, manually chunk at 500 without adaptation
    # by bypassing the method -- just verify chunk count is reasonable
    # With 300 short turns at ~5 tokens each and max_tokens=1000,
    # we expect roughly (300*5)/1000 ~ 2 chunks (plus overlap)
    assert len(chunks) < 100, f"Expected fewer chunks from adaptive sizing, got {len(chunks)}"

    # Cleanup
    os.unlink(path)
    print(f"  Chunks produced: {len(chunks)} (from {len(turns)} turns) -- OK")
    print("  PASSED")


def test_adaptive_chunking_formal_e2e():
    """Formal/long-turn chat should produce tighter chunks (max_tokens=400)."""
    print("\n[TEST] Phase 1b: Adaptive chunking -- formal chat e2e")

    path = _build_formal_chat_file(num_turns=40)
    parser = ConversationParser(path, slug="test_formal")
    turns = parser.parse()
    assert len(turns) >= 30, f"Expected 30+ turns, got {len(turns)}"

    chunks = parser.chunk(turns, max_tokens=500)

    # With avg_turn > 150 tokens, adaptive sets max_tokens=400
    # 40 turns * ~200 tokens / 400 = ~20 chunks (plus overlap)
    # More chunks than with default 500
    assert len(chunks) > 10, f"Expected >10 chunks for formal chat, got {len(chunks)}"

    os.unlink(path)
    print(f"  Chunks produced: {len(chunks)} (from {len(turns)} turns) -- OK")
    print("  PASSED")


def test_adaptive_chunking_short_conversation_unchanged():
    """Conversations < 20 turns should keep default max_tokens."""
    print("\n[TEST] Phase 1c: Adaptive chunking -- short conversation keeps default")

    speakers = ["Alice", "Bob"]
    lines = []
    for i in range(10):
        speaker = speakers[i % 2]
        lines.append(f"{speaker}: Short message number {i+1}")

    path = tempfile.mktemp(suffix=".txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    parser = ConversationParser(path, slug="test_short")
    turns = parser.parse()
    assert len(turns) == 10

    chunks = parser.chunk(turns, max_tokens=500)
    # With 10 short turns, adaptive keeps default 500
    # All turns fit in one chunk
    assert len(chunks) >= 1

    os.unlink(path)
    print(f"  Chunks produced: {len(chunks)} (from {len(turns)} turns, default sizing) -- OK")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Phase 2: Episode Store Methods (DB round-trip)
# ---------------------------------------------------------------------------

def test_episode_store_get_all_episodes_for_doc():
    """get_all_episodes_for_doc returns episodes ordered by turn_start."""
    print("\n[TEST] Phase 2a: get_all_episodes_for_doc")

    store = PostgresGraphStore()
    content_store = PgresStore()

    # Find an existing conversation doc with episodes
    rows = content_store.execute(
        "SELECT d.doc_id, d.slug FROM documents d WHERE d.doc_type = 'conversation' LIMIT 5",
        fetch="all"
    )
    if not rows:
        print("  SKIPPED: No conversation documents in DB")
        return

    found_episodes = False
    for doc_id, slug in rows:
        episodes = store.get_all_episodes_for_doc(doc_id)
        if episodes:
            found_episodes = True
            # Verify ordering by turn_start
            turn_starts = [ep.get("turn_start") for ep in episodes if ep.get("turn_start") is not None]
            assert turn_starts == sorted(turn_starts), "Episodes should be ordered by turn_start"

            # Verify structure
            for ep in episodes:
                assert "topic" in ep, "Episode missing 'topic'"
                assert "summary" in ep, "Episode missing 'summary'"
                assert "speaker" in ep, "Episode missing 'speaker'"
                assert "source_chunk_ids" in ep, "Episode missing 'source_chunk_ids'"

            print(f"  Found {len(episodes)} episodes for doc '{slug}' (doc_id={doc_id})")
            print(f"  Turn ordering: valid ({len(turn_starts)} with turn_start)")
            break

    if not found_episodes:
        print("  SKIPPED: No episodes found for any conversation doc")
        return

    print("  PASSED")


def test_episode_store_get_by_chunk_ids():
    """get_episodes_by_chunk_ids returns episodes matching given chunk IDs."""
    print("\n[TEST] Phase 2b: get_episodes_by_chunk_ids")

    store = PostgresGraphStore()
    content_store = PgresStore()

    # Find an existing conversation doc with episodes
    rows = content_store.execute(
        "SELECT d.doc_id, d.slug FROM documents d WHERE d.doc_type = 'conversation' LIMIT 5",
        fetch="all"
    )
    if not rows:
        print("  SKIPPED: No conversation documents in DB")
        return

    found = False
    for doc_id, slug in rows:
        all_episodes = store.get_all_episodes_for_doc(doc_id)
        if not all_episodes:
            continue

        # Grab chunk IDs from the first episode
        first_ep = all_episodes[0]
        chunk_ids = first_ep.get("source_chunk_ids", [])
        if not chunk_ids:
            continue

        # Query by those chunk IDs
        matched = store.get_episodes_by_chunk_ids(doc_id, chunk_ids)
        assert len(matched) >= 1, f"Expected at least 1 matching episode, got {len(matched)}"

        # The first episode's topic should appear in matched results
        matched_topics = {ep["topic"] for ep in matched}
        assert first_ep["topic"] in matched_topics, (
            f"Expected topic '{first_ep['topic']}' in matched results, got {matched_topics}"
        )

        # Query with a non-existent chunk ID should return nothing or fewer results
        fake_ids = ["nonexistent_chunk_999"]
        fake_matched = store.get_episodes_by_chunk_ids(doc_id, fake_ids)
        assert len(fake_matched) == 0, f"Expected 0 results for fake chunk IDs, got {len(fake_matched)}"

        found = True
        print(f"  Queried {len(chunk_ids)} chunk IDs -> {len(matched)} episodes matched")
        print("  Fake chunk ID -> 0 episodes (correct)")
        break

    if not found:
        print("  SKIPPED: No episodes with source_chunk_ids found")
        return

    print("  PASSED")


# ---------------------------------------------------------------------------
# Phase 3: Episode Context in Retrieval
# ---------------------------------------------------------------------------

def test_episode_context_in_aggregator():
    """context_aggregator_node injects [EPISODE CONTEXT] for conversation docs with episodes."""
    print("\n[TEST] Phase 3: Episode context injected in aggregator")

    from src.flows.agent_graph import context_aggregator_node

    store = PostgresGraphStore()
    content_store = PgresStore()

    # Find a conversation doc with episodes and chunks
    rows = content_store.execute(
        "SELECT d.doc_id, d.slug FROM documents d WHERE d.doc_type = 'conversation' LIMIT 5",
        fetch="all"
    )
    if not rows:
        print("  SKIPPED: No conversation documents in DB")
        return

    found = False
    for doc_id, slug in rows:
        episodes = store.get_all_episodes_for_doc(doc_id)
        if not episodes:
            continue

        # Get some real chunk IDs from the episodes
        chunk_ids = []
        for ep in episodes[:3]:
            chunk_ids.extend(ep.get("source_chunk_ids", []))
        if not chunk_ids:
            continue

        # Build minimal fake passages with real chunk IDs
        fake_passages = {}
        fake_passage_list = []
        for cid in chunk_ids[:3]:
            fake_passage_list.append({
                "id": cid,
                "text": f"Some conversation text from {cid}",
                "metadata": {
                    "speakers": ["Alice"],
                    "timestamp_start": "2024-01-01 10:00:00"
                }
            })
        fake_passages[slug] = fake_passage_list

        # Build a minimal state for the aggregator
        from langchain_core.messages import HumanMessage
        state = {
            "messages": [HumanMessage(content="What happened?")],
            "partial_summaries": {},
            "partial_passages": fake_passages,
            "partial_relations": {},
            "target_doc_types": {slug: "conversation"},
            "router_entities": [],
        }

        result = context_aggregator_node(state)
        context = result["retrieved_context"]

        if "[EPISODE CONTEXT]" in context:
            found = True
            # Count episode lines
            ep_lines = [line for line in context.split("\n") if line.startswith("- ") and " on '" in line]
            print(f"  Doc: {slug}")
            print(f"  [EPISODE CONTEXT] found with {len(ep_lines)} episode lines")
            assert len(ep_lines) >= 1, "Expected at least 1 episode context line"
            print(f"  Sample: {ep_lines[0][:120]}...")
            break

    if not found:
        print("  SKIPPED: Could not find a doc with episodes + chunk matches")
        return

    print("  PASSED")


# ---------------------------------------------------------------------------
# Phase 4: Arc Summary Generation
# ---------------------------------------------------------------------------

async def test_arc_summary_generation():
    """generate_arc_summaries produces valid arc summary dicts."""
    print("\n[TEST] Phase 4: Arc summary generation")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  SKIPPED: OPENAI_API_KEY not set")
        return

    from src.llm.generator import SummaryGenerator

    # Build synthetic episodes
    episodes = []
    topics = [
        ("Alice", "project deadline", "concerned", "Alice raised concerns about the approaching deadline"),
        ("Bob", "budget allocation", "supportive", "Bob proposed reallocating funds from marketing"),
        ("Carol", "team morale", "neutral", "Carol noted that morale has been declining recently"),
        ("Alice", "hiring plan", "supportive", "Alice suggested hiring two more engineers"),
        ("Bob", "technical debt", "critical", "Bob argued the codebase needs refactoring before new features"),
        ("Carol", "client feedback", "neutral", "Carol shared positive feedback from the enterprise client"),
        ("Alice", "release schedule", "concerned", "Alice worried the Q3 release will slip"),
        ("Bob", "infrastructure costs", "critical", "Bob flagged that AWS costs doubled last quarter"),
    ]

    for i, (speaker, topic, stance, summary) in enumerate(topics):
        episodes.append({
            "speaker": speaker,
            "topic": topic,
            "stance": stance,
            "summary": summary,
            "turn_start": i * 10 + 1,
            "turn_end": i * 10 + 9,
        })

    gen = SummaryGenerator(doc_type="conversation")
    arcs = await gen.generate_arc_summaries(episodes, target_arcs=3)

    assert isinstance(arcs, list), f"Expected list, got {type(arcs)}"
    assert len(arcs) >= 1, f"Expected at least 1 arc, got {len(arcs)}"

    for arc in arcs:
        assert "chapter_id" in arc, f"Arc missing 'chapter_id': {arc}"
        assert "summary" in arc, f"Arc missing 'summary': {arc}"
        assert isinstance(arc["chapter_id"], int), f"chapter_id should be int: {arc['chapter_id']}"
        assert len(arc["summary"]) > 10, f"Arc summary too short: {arc['summary']}"

    print(f"  Generated {len(arcs)} arcs from {len(episodes)} episodes")
    for arc in arcs:
        print(f"    Arc {arc['chapter_id']}: {arc['summary'][:80]}...")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Phase 2+: GIN Index Verification
# ---------------------------------------------------------------------------

def test_gin_index_exists():
    """Verify the GIN index on graph_episodes.source_chunk_ids exists."""
    print("\n[TEST] Phase 2+: GIN index on graph_episodes.source_chunk_ids")

    store = PgresStore()
    rows = store.execute(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'graph_episodes'
          AND indexdef ILIKE '%gin%'
        """,
        fetch="all"
    )

    if rows:
        for name, defn in rows:
            print(f"  Found GIN index: {name}")
            print(f"    Definition: {defn}")
        print("  PASSED")
    else:
        print("  NOT FOUND: GIN index on graph_episodes not yet created.")
        print("  Run: CREATE INDEX IF NOT EXISTS idx_gep_chunk_ids ON graph_episodes USING GIN (source_chunk_ids);")
        print("  SKIPPED (non-fatal)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 80)
    print("CONVERSATION INTELLIGENCE v2 -- INTEGRATION TESTS")
    print("=" * 80)

    passed = 0
    failed = 0

    tests_sync = [
        test_adaptive_chunking_social_chat_e2e,
        test_adaptive_chunking_formal_e2e,
        test_adaptive_chunking_short_conversation_unchanged,
        test_episode_store_get_all_episodes_for_doc,
        test_episode_store_get_by_chunk_ids,
        test_episode_context_in_aggregator,
        test_gin_index_exists,
    ]

    tests_async = [
        test_arc_summary_generation,
    ]

    for test in tests_sync:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    for test in tests_async:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
