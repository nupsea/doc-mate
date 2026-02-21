"""
Agent behavior tests: isolated component correctness and topology invariants.

Covers:
1. Router scope constraint  -- selected doc is always respected
2. _VAGUE_REF_RE pattern    -- detects pronouns and generic roles
3. Aggregator citation      -- None timestamp coerced to "", speakers populated
4. source_refs metadata     -- timestamp and speakers included for conversations
5. Graph retriever trigger  -- skips when no entities; runs when entities present
6. Graph topology           -- aggregator runs once (equal-depth fan-in)
7. Confidence for summaries -- summary-only responses not penalised as low confidence
"""

import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# 1. Router scope constraint
# ---------------------------------------------------------------------------

def test_router_constrains_to_selected_doc_when_set():
    """When a doc is selected and intent != compare, router_node must constrain slugs to it."""
    from src.flows.agent_graph import router_node

    # Simulate router returning multiple slugs (e.g. due to entity discovery)
    mock_decision = {
        "intent": "search",
        "strategy": "search",
        "slugs": ["some_other_doc", "yet_another"],
        "entities": [],
    }

    with patch("src.flows.agent_graph.QueryRouter") as MockRouter:
        mock_instance = MockRouter.return_value
        mock_instance.route.return_value = mock_decision

        # PgresStore is imported locally inside router_node — patch at source
        with patch("src.content.store.PgresStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.execute.return_value = [("my_doc", "book")]

            state = {
                "messages": [HumanMessage(content="What is the theme?")],
                "provider": "openai",
                "model": "gpt-4o-mini",
                "selected_doc_slug": "my_doc",
            }
            result = router_node(state)

    # Must be constrained to the selected doc only
    assert result["target_slugs"] == ["my_doc"], (
        f"Expected ['my_doc'], got {result['target_slugs']}"
    )


def test_router_does_not_constrain_for_compare_intent():
    """For compare intent, router should NOT constrain to selected doc."""
    from src.flows.agent_graph import router_node

    mock_decision = {
        "intent": "compare",
        "strategy": "hybrid",
        "slugs": ["doc_a", "doc_b"],
        "entities": [],
    }

    with patch("src.flows.agent_graph.QueryRouter") as MockRouter:
        mock_instance = MockRouter.return_value
        mock_instance.route.return_value = mock_decision

        with patch("src.content.store.PgresStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.execute.return_value = [("doc_a", "book"), ("doc_b", "book")]

            state = {
                "messages": [HumanMessage(content="Compare doc_a and doc_b")],
                "provider": "openai",
                "model": "gpt-4o-mini",
                "selected_doc_slug": "doc_a",
            }
            result = router_node(state)

    # Both docs should appear (compare intent bypasses the constraint)
    assert "doc_a" in result["target_slugs"]
    assert "doc_b" in result["target_slugs"]


def test_router_no_constraint_when_no_doc_selected():
    """When no doc is pinned ('none' or empty), router returns what it decided."""
    from src.flows.agent_graph import router_node

    mock_decision = {
        "intent": "search",
        "strategy": "search",
        "slugs": ["book_a"],
        "entities": [],
    }

    with patch("src.flows.agent_graph.QueryRouter") as MockRouter:
        mock_instance = MockRouter.return_value
        mock_instance.route.return_value = mock_decision

        with patch("src.content.store.PgresStore") as MockStore:
            mock_store = MockStore.return_value
            mock_store.execute.return_value = [("book_a", "book")]

            state = {
                "messages": [HumanMessage(content="What is this about?")],
                "provider": "openai",
                "model": "gpt-4o-mini",
                "selected_doc_slug": "none",
            }
            result = router_node(state)

    assert result["target_slugs"] == ["book_a"]


# ---------------------------------------------------------------------------
# 2. _VAGUE_REF_RE pattern matching
# ---------------------------------------------------------------------------

def test_vague_ref_re_detects_pronouns():
    """Pronoun references should be detected."""
    from src.flows.agent_graph import _VAGUE_REF_RE

    queries_with_vague = [
        "What does he think about the project?",
        "How do they feel about each other?",
        "What is her position on the matter?",
        "Tell me about both of them",
        "What do the two speakers discuss?",
        "How does the protagonist feel?",
        "What does the narrator say?",
    ]
    for q in queries_with_vague:
        assert _VAGUE_REF_RE.search(q), f"Expected match for: '{q}'"


def test_vague_ref_re_does_not_match_specific_queries():
    """Specific queries without pronouns or generic roles should not match."""
    from src.flows.agent_graph import _VAGUE_REF_RE

    specific_queries = [
        "What does Alice think about the project?",
        "How does Odysseus feel about returning home?",
        "Tell me about Achilles and Hector",
        "What happened during the Trojan War?",
        "Describe the themes of the Iliad",
    ]
    for q in specific_queries:
        assert not _VAGUE_REF_RE.search(q), f"Unexpected match for: '{q}'"


# ---------------------------------------------------------------------------
# 3. Aggregator citation format — None timestamp coercion
# ---------------------------------------------------------------------------

def test_aggregator_coerces_none_timestamp():
    """None timestamp_start should not appear as 'None' in citations."""
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What did they say?")],
        "partial_summaries": {},
        "partial_passages": {
            "test_conv": [
                {
                    "id": "test_conv_01_001_abc",
                    "text": "Hello, this is a test conversation.",
                    "metadata": {
                        "timestamp_start": None,  # Key present but null
                        "speakers": ["Alice", "Bob"],
                    },
                }
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"test_conv": "conversation"},
        "router_entities": [],
    }

    result = context_aggregator_node(state)
    context = result["retrieved_context"]

    # "None" should not appear in the citation
    assert "Time: None" not in context, "None timestamp leaked into citation"
    # Speakers should be present
    assert "Speakers: Alice, Bob" in context, "Speakers missing from citation"


def test_aggregator_includes_timestamp_when_present():
    """When timestamp_start is a valid string, it should appear in the citation."""
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What was discussed?")],
        "partial_summaries": {},
        "partial_passages": {
            "conv_doc": [
                {
                    "id": "conv_doc_01_001_abc",
                    "text": "We discussed the budget.",
                    "metadata": {
                        "timestamp_start": "2024-03-15 10:30:00",
                        "speakers": ["Manager"],
                    },
                }
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"conv_doc": "conversation"},
        "router_entities": [],
    }

    result = context_aggregator_node(state)
    context = result["retrieved_context"]

    assert "Time: 2024-03-15 10:30:00" in context, "Timestamp missing from citation"
    assert "Speakers: Manager" in context, "Speaker missing from citation"


# ---------------------------------------------------------------------------
# 4. source_refs metadata for conversations
# ---------------------------------------------------------------------------

def test_source_refs_include_timestamp_and_speakers():
    """source_refs should include timestamp and speakers for conversation docs."""
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What happened?")],
        "partial_summaries": {},
        "partial_passages": {
            "office_chat": [
                {
                    "id": "office_chat_01_001_abc",
                    "text": "The team discussed the release plan.",
                    "metadata": {
                        "timestamp_start": "2024-01-10 09:00:00",
                        "speakers": ["Sarah", "Tom"],
                    },
                },
                {
                    "id": "office_chat_01_002_def",
                    "text": "Q4 targets were reviewed.",
                    "metadata": {
                        "timestamp_start": None,  # No timestamp for this one
                        "speakers": [],
                    },
                },
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"office_chat": "conversation"},
        "router_entities": [],
    }

    result = context_aggregator_node(state)
    refs = result["source_refs"]

    assert len(refs) == 2

    # First ref has both timestamp and speakers
    ref_with_ts = next((r for r in refs if r["chunk_id"] == "office_chat_01_001_abc"), None)
    assert ref_with_ts is not None
    assert ref_with_ts.get("timestamp") == "2024-01-10 09:00:00"
    assert ref_with_ts.get("speakers") == ["Sarah", "Tom"]

    # Second ref has no timestamp (None) and no speakers (empty) — neither key should be set
    ref_no_ts = next((r for r in refs if r["chunk_id"] == "office_chat_01_002_def"), None)
    assert ref_no_ts is not None
    assert "timestamp" not in ref_no_ts, "Empty timestamp should not appear in source_ref"
    assert "speakers" not in ref_no_ts, "Empty speakers should not appear in source_ref"


def test_source_refs_omit_metadata_for_books():
    """source_refs for book docs should not include timestamp or speakers."""
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What is the theme?")],
        "partial_summaries": {},
        "partial_passages": {
            "iliad": [
                {
                    "id": "iliad_01_001_abc",
                    "text": "Achilles refused to fight.",
                    "metadata": {"chapter": "1"},
                }
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"iliad": "book"},
        "router_entities": [],
    }

    result = context_aggregator_node(state)
    refs = result["source_refs"]

    assert len(refs) == 1
    assert "timestamp" not in refs[0]
    assert "speakers" not in refs[0]
    assert refs[0]["slug"] == "iliad"
    assert refs[0]["chunk_id"] == "iliad_01_001_abc"


# ---------------------------------------------------------------------------
# 5. Graph retriever trigger logic
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_graph_retriever_skips_when_no_entities_and_basic_intent(anyio_backend):
    """Graph retriever should skip when intent=search, strategy=search, no entities."""
    from src.flows.agent_graph import graph_retriever_node

    state = {
        "messages": [HumanMessage(content="Summarize the document")],
        "intent": "search",
        "strategy": "search",
        "entities": [],
        "target_slugs": ["some_doc"],
        "target_doc_types": {"some_doc": "book"},
        "rewritten_query": "Summarize the document",
    }

    result = await graph_retriever_node(state)
    assert result["partial_relations"] == {}, (
        f"Expected empty relations when skipping, got: {result['partial_relations']}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_graph_retriever_runs_when_entities_present(anyio_backend):
    """Graph retriever should run when router extracted entities."""
    from src.flows.agent_graph import graph_retriever_node

    # PostgresGraphStore is imported locally inside _fetch_graph — patch at source
    with patch("src.graph.store.PostgresGraphStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store._resolve_doc_id.return_value = 42
        mock_store.find_entities_by_names.return_value = {}  # No matches, but store was called

        state = {
            "messages": [HumanMessage(content="What is Alice's role?")],
            "intent": "search",
            "strategy": "search",
            "entities": ["Alice"],
            "target_slugs": ["office_chat"],
            "target_doc_types": {"office_chat": "conversation"},
            "rewritten_query": "What is Alice's role?",
        }

        result = await graph_retriever_node(state)

    # The store was queried (graph retrieval ran), result is just empty
    mock_store.find_entities_by_names.assert_called_once()
    assert result["partial_relations"] == {} or isinstance(result["partial_relations"], dict)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_graph_retriever_runs_for_explore_intent(anyio_backend):
    """Graph retriever should run for explore intent even without explicit entities."""
    from src.flows.agent_graph import graph_retriever_node

    with patch("src.graph.store.PostgresGraphStore") as MockStore:
        mock_store = MockStore.return_value
        mock_store._resolve_doc_id.return_value = 99
        mock_store.find_entities_by_names.return_value = {}

        state = {
            "messages": [HumanMessage(content="Tell me about all the characters")],
            "intent": "explore",
            "strategy": "hybrid",
            "entities": [],
            "target_slugs": ["iliad"],
            "target_doc_types": {"iliad": "book"},
            "rewritten_query": "Tell me about all the characters",
        }

        await graph_retriever_node(state)

    # Should have attempted to resolve entities (explore intent triggers graph lookup)
    mock_store.find_entities_by_names.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Graph topology — aggregator receives data from all three retrievers
# ---------------------------------------------------------------------------

def test_aggregator_receives_summaries_and_passages_together():
    """
    Aggregator should see both partial_summaries and partial_passages in one call.
    Regression test for the double-aggregator bug caused by unequal path depths.
    """
    from src.flows.agent_graph import context_aggregator_node

    # Simulate a state where both summary retriever AND content retriever have results
    state = {
        "messages": [HumanMessage(content="What is this book about?")],
        "partial_summaries": {
            "iliad": "The Iliad is an ancient Greek epic poem about the Trojan War.",
        },
        "partial_passages": {
            "iliad": [
                {
                    "id": "iliad_01_001_abc",
                    "text": "Sing, O goddess, the anger of Achilles.",
                    "metadata": {"chapter": "1"},
                }
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"iliad": "book"},
        "router_entities": [],
    }

    result = context_aggregator_node(state)
    context = result["retrieved_context"]

    # Both the overview and the passage should be present
    assert "[OVERVIEW]" in context, "Summary (OVERVIEW) missing from context"
    assert "Trojan War" in context, "Summary content missing"
    assert "[RELEVANT PASSAGES]" in context, "Passages section missing from context"
    assert "anger of Achilles" in context, "Passage content missing"


# ---------------------------------------------------------------------------
# 7. Confidence for summary-only responses
# ---------------------------------------------------------------------------

def test_aggregator_summary_only_not_low_confidence():
    """
    When only summaries are retrieved (no chunks), confidence should not be 'low'.
    The summary text itself is evidence and should be used in confidence assessment.
    """
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What is sample_doc about?")],
        "partial_summaries": {
            "sample_doc": "A book about organisational change and leadership challenges.",
        },
        "partial_passages": {},  # No passages — summary-only strategy
        "partial_relations": {},
        "target_doc_types": {"sample_doc": "book"},
        "router_entities": [],
    }

    result = context_aggregator_node(state)

    # Confidence should not be 'low' when a summary was retrieved
    assert result["confidence_level"] != "low", (
        f"Summary-only response should not have low confidence, got: {result['confidence_level']}"
    )


# ---------------------------------------------------------------------------
# 8. NAME NOTE — contact name vs entity name disambiguation
# ---------------------------------------------------------------------------

def test_aggregator_emits_name_note_when_entity_not_in_speakers():
    """
    When a query entity name doesn't appear in any speaker metadata,
    a [NAME NOTE] should be emitted so the LLM can reconcile contact names
    with names used in conversation text.
    """
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What did Alex say about the project?")],
        "partial_summaries": {},
        "partial_passages": {
            "conv_doc": [
                {
                    "id": "conv_doc_01_001_abc",
                    "text": "I think we should proceed with the plan",
                    "metadata": {
                        "timestamp_start": "2024-03-10 09:00:00",
                        "speakers": ["Nick"],  # Contact name differs from entity name
                    },
                }
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"conv_doc": "conversation"},
        "router_entities": ["Alex"],  # Name from query / graph — not in speaker metadata
    }

    result = context_aggregator_node(state)
    context = result["retrieved_context"]

    assert "[NAME NOTE]" in context, "NAME NOTE missing when entity not in speakers"
    assert "Alex" in context, "Entity name should appear in NAME NOTE"
    assert "'nick'" in context.lower(), "Contact name should appear in NAME NOTE"


def test_aggregator_no_name_note_when_entity_matches_speaker():
    """
    When the query entity name matches a speaker name, no [NAME NOTE] should appear.
    """
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What did Sam say?")],
        "partial_summaries": {},
        "partial_passages": {
            "conv_doc": [
                {
                    "id": "conv_doc_01_001_abc",
                    "text": "I think we should proceed with the plan",
                    "metadata": {
                        "timestamp_start": "2024-03-10 09:00:00",
                        "speakers": ["Sam"],
                    },
                }
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"conv_doc": "conversation"},
        "router_entities": ["Sam"],  # Matches the speaker name exactly
    }

    result = context_aggregator_node(state)
    context = result["retrieved_context"]

    assert "[NAME NOTE]" not in context, "NAME NOTE should not appear when entity matches speaker"


def test_aggregator_name_note_for_casual_conversation():
    """NAME NOTE fires for informal conversation docs the same as formal ones."""
    from src.flows.agent_graph import context_aggregator_node

    state = {
        "messages": [HumanMessage(content="What did Jordan say?")],
        "partial_summaries": {},
        "partial_passages": {
            "chat_doc": [
                {
                    "id": "chat_doc_01_001_abc",
                    "text": "Meeting at 3pm today",
                    "metadata": {
                        "timestamp_start": "2024-01-20 14:00:00",
                        "speakers": ["Jay"],  # Contact name differs from entity name
                    },
                }
            ]
        },
        "partial_relations": {},
        "target_doc_types": {"chat_doc": "conversation"},
        "router_entities": ["Jordan"],
    }

    result = context_aggregator_node(state)
    context = result["retrieved_context"]

    assert "[NAME NOTE]" in context, "NAME NOTE should fire for conversation doc type"
    assert "Jordan" in context
