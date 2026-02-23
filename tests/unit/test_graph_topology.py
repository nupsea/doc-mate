"""
LangGraph topology tests for the Doc-Mate RRG (Router-Retriever-Generator) pipeline.

These tests verify the graph wiring using lightweight stub nodes so no external
services (DB, LLM, Qdrant) are required.  They cover:

  1. Node call counts     — every node fires exactly once per invocation
  2. Execution ordering   — dependency ordering is respected
  3. Fan-in correctness   — aggregator sees data from ALL three retrievers in one call
  4. Retriever skip logic — nodes that should be no-ops return empty dicts
  5. State propagation    — each node's output is visible to downstream nodes
  6. Error isolation      — a graceful-failing retriever does not block the others

Why stub-based (not full integration)?
  Using stub nodes that return minimal state lets us test the WIRING independently
  of business logic, making failures unambiguous: if the aggregator fires twice, it
  is a topology bug, not a DB or LLM issue.

Production topology under test (from create_agent_graph):

    router
      └─► query_rewriter
              ├─► summary_retriever ─┐
              ├─► content_retriever ─┼─► aggregator ─► generator ─► END
              └─► graph_retriever  ──┘

All three retrievers branch from query_rewriter (equal depth = 3) so the aggregator
fires exactly once.  The previous topology had summary_retriever at depth 2,
causing a double-aggregator bug.
"""

import asyncio
from collections import Counter
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Minimal state required by AgentState to keep LangGraph happy
# ---------------------------------------------------------------------------

_BASE_STATE = {
    "messages": [HumanMessage(content="What is this doc about?")],
    "provider": "openai",
    "model": "gpt-4o-mini",
    "selected_doc_slug": "none",
    "intent": "",
    "strategy": "",
    "target_slugs": [],
    "target_doc_types": {},
    "entities": [],
    "router_entities": [],
    "partial_summaries": {},
    "partial_passages": {},
    "partial_relations": {},
    "confidence_level": "",
    "confidence_score": 0.0,
    "evidence_gaps": [],
    "query_type": "",
    "rewritten_query": "",
    "retrieved_context": "",
    "source_refs": [],
}

# Default return values for each stub node
_ROUTER_OUT = {
    "intent": "search",
    "strategy": "search",
    "target_slugs": ["test_doc"],
    "target_doc_types": {"test_doc": "book"},
    "entities": [],
    "router_entities": [],
}
_REWRITER_OUT = {"rewritten_query": "What is this doc about?"}
_SUMMARY_OUT = {"partial_summaries": {"test_doc": "A great book about things."}}
_CONTENT_OUT = {"partial_passages": {"test_doc": [{"id": "c1", "text": "Passage.", "metadata": {}}]}}
_GRAPH_OUT = {"partial_relations": {"test_doc": ["A related_to B"]}}
_AGG_OUT = {
    "retrieved_context": "assembled context",
    "confidence_level": "medium",
    "confidence_score": 0.5,
    "evidence_gaps": [],
    "query_type": "broad",
    "source_refs": [],
}
_GEN_OUT = {"messages": [AIMessage(content="Here is your answer.")]}


# ---------------------------------------------------------------------------
# Stub factory
# ---------------------------------------------------------------------------

def _make_stubs(call_log: list, call_counts: Counter, *, overrides: dict = None):
    """
    Build a dict suitable for patch.multiple('src.flows.agent_graph', ...).

    Each entry maps the module-level function name to a stub.
    Stubs append a short label to call_log and increment call_counts.

    overrides: {module_fn_name: callable(state)} replaces the entire stub logic
               for that node (call tracking still applies).
    """
    overrides = overrides or {}

    def _sync(label, module_key, default_ret):
        custom = overrides.get(module_key)
        if custom is not None:
            def stub(state, _fn=custom, _l=label):
                call_log.append(_l)
                call_counts[_l] += 1
                return _fn(state)
        else:
            def stub(state, _v=default_ret, _l=label):
                call_log.append(_l)
                call_counts[_l] += 1
                return _v
        stub.__name__ = label
        return stub

    def _async(label, module_key, default_ret):
        custom = overrides.get(module_key)
        if custom is not None:
            async def stub(state, _fn=custom, _l=label):
                call_log.append(_l)
                call_counts[_l] += 1
                result = _fn(state)
                return (await result) if asyncio.iscoroutine(result) else result
        else:
            async def stub(state, _v=default_ret, _l=label):
                call_log.append(_l)
                call_counts[_l] += 1
                return _v
        stub.__name__ = label
        return stub

    return {
        "router_node":             _sync("router",     "router_node",             _ROUTER_OUT),
        "query_rewriter_node":     _async("rewriter",  "query_rewriter_node",     _REWRITER_OUT),
        "summary_retriever_node":  _async("summary",   "summary_retriever_node",  _SUMMARY_OUT),
        "content_retriever_node":  _async("content",   "content_retriever_node",  _CONTENT_OUT),
        "graph_retriever_node":    _async("graph",     "graph_retriever_node",    _GRAPH_OUT),
        "context_aggregator_node": _sync("aggregator", "context_aggregator_node", _AGG_OUT),
        "generator_node":          _sync("generator",  "generator_node",          _GEN_OUT),
    }


async def _run(overrides: dict = None, initial: dict = None):
    """Patch node functions, build the production graph, invoke it, return final state."""
    call_log: list = []
    call_counts: Counter = Counter()
    stubs = _make_stubs(call_log, call_counts, overrides=overrides)
    state = dict(_BASE_STATE, **(initial or {}))
    with patch.multiple("src.flows.agent_graph", **stubs):
        from src.flows.agent_graph import create_agent_graph  # noqa: PLC0415
        graph = create_agent_graph()
        final = await graph.ainvoke(state)
    return final, call_log, call_counts


# ---------------------------------------------------------------------------
# 1. Node call counts
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_every_node_called_exactly_once(anyio_backend):
    """Each of the 7 pipeline nodes must fire exactly once per invocation."""
    _, _, call_counts = await _run()

    expected = {"router", "rewriter", "summary", "content", "graph", "aggregator", "generator"}
    assert set(call_counts.keys()) == expected, (
        f"Node set mismatch: got {set(call_counts.keys())}"
    )
    for node, count in call_counts.items():
        assert count == 1, f"Node '{node}' called {count} times, expected 1"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_aggregator_called_exactly_once(anyio_backend):
    """
    Regression: equal-depth fan-in must prevent the aggregator from firing
    multiple times (the prior topology had summary_retriever at depth 2).
    """
    _, _, call_counts = await _run()
    assert call_counts["aggregator"] == 1, (
        f"Aggregator fired {call_counts['aggregator']} times — depth mismatch?"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_generator_called_exactly_once(anyio_backend):
    """Generator depends only on the aggregator and must fire exactly once."""
    _, _, call_counts = await _run()
    assert call_counts["generator"] == 1, (
        f"Generator fired {call_counts['generator']} times"
    )


# ---------------------------------------------------------------------------
# 2. Execution ordering
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_router_runs_before_query_rewriter(anyio_backend):
    _, call_log, _ = await _run()
    assert call_log.index("router") < call_log.index("rewriter"), (
        f"Expected router before rewriter, got: {call_log}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_query_rewriter_runs_before_all_retrievers(anyio_backend):
    """query_rewriter must complete before summary, content, and graph retrievers start."""
    _, call_log, _ = await _run()
    rewriter_pos = call_log.index("rewriter")
    for retriever in ("summary", "content", "graph"):
        pos = call_log.index(retriever)
        assert rewriter_pos < pos, (
            f"rewriter (pos {rewriter_pos}) must precede '{retriever}' (pos {pos})"
        )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_all_retrievers_complete_before_aggregator(anyio_backend):
    """The aggregator must fire only after all three retrievers have completed."""
    _, call_log, _ = await _run()
    agg_pos = call_log.index("aggregator")
    for retriever in ("summary", "content", "graph"):
        pos = call_log.index(retriever)
        assert pos < agg_pos, (
            f"'{retriever}' (pos {pos}) must precede aggregator (pos {agg_pos})"
        )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_aggregator_runs_before_generator(anyio_backend):
    """Generator depends on retrieved_context which comes from the aggregator."""
    _, call_log, _ = await _run()
    assert call_log.index("aggregator") < call_log.index("generator"), (
        f"aggregator must precede generator, got: {call_log}"
    )


# ---------------------------------------------------------------------------
# 3. Fan-in correctness — aggregator sees ALL retriever data in one call
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_aggregator_receives_all_three_retriever_outputs(anyio_backend):
    """
    The aggregator must see partial_summaries, partial_passages, AND
    partial_relations in a single call — not across multiple calls.
    """
    received: list = []  # Captures the state each time aggregator is called

    def capturing_aggregator(state):
        received.append({
            "partial_summaries": dict(state.get("partial_summaries", {})),
            "partial_passages": dict(state.get("partial_passages", {})),
            "partial_relations": dict(state.get("partial_relations", {})),
        })
        return _AGG_OUT

    await _run(overrides={"context_aggregator_node": capturing_aggregator})

    assert len(received) == 1, f"Aggregator called {len(received)} times, expected 1"
    snap = received[0]
    assert snap["partial_summaries"] == _SUMMARY_OUT["partial_summaries"], (
        "partial_summaries missing or wrong when aggregator ran"
    )
    assert snap["partial_passages"] == _CONTENT_OUT["partial_passages"], (
        "partial_passages missing or wrong when aggregator ran"
    )
    assert snap["partial_relations"] == _GRAPH_OUT["partial_relations"], (
        "partial_relations missing or wrong when aggregator ran"
    )


# ---------------------------------------------------------------------------
# 4. Retriever skip logic
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_summary_retriever_skips_for_search_strategy():
    """summary_retriever_node returns empty when strategy='search'."""
    from src.flows.agent_graph import summary_retriever_node  # noqa: PLC0415

    result = await summary_retriever_node({"strategy": "search", "target_slugs": ["x"]})
    assert result["partial_summaries"] == {}, (
        f"Expected empty summaries for strategy=search, got: {result['partial_summaries']}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_content_retriever_skips_for_summary_strategy():
    """content_retriever_node returns empty when strategy='summary'."""
    from src.flows.agent_graph import content_retriever_node  # noqa: PLC0415

    state = {
        "strategy": "summary",
        "target_slugs": ["x"],
        "target_doc_types": {},
        "messages": [HumanMessage(content="overview")],
        "rewritten_query": "",
        "router_entities": [],
    }
    result = await content_retriever_node(state)
    assert result["partial_passages"] == {}, (
        f"Expected empty passages for strategy=summary, got: {result['partial_passages']}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_graph_retriever_skips_when_no_entities_and_basic_intent():
    """graph_retriever_node returns empty when intent=search, strategy=search, no entities."""
    from src.flows.agent_graph import graph_retriever_node  # noqa: PLC0415

    state = {
        "messages": [HumanMessage(content="Summarize this")],
        "intent": "search",
        "strategy": "search",
        "entities": [],
        "target_slugs": ["x"],
        "target_doc_types": {"x": "book"},
        "rewritten_query": "Summarize this",
    }
    result = await graph_retriever_node(state)
    assert result["partial_relations"] == {}, (
        f"Expected empty relations, got: {result['partial_relations']}"
    )


# ---------------------------------------------------------------------------
# 5. State propagation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_router_slugs_reach_all_retrievers(anyio_backend):
    """target_slugs set by the router must be visible to all three retrievers."""
    slugs_seen: dict = {}

    def custom_router(state):
        return dict(_ROUTER_OUT, target_slugs=["specific_doc"])

    async def spy_summary(state):
        slugs_seen["summary"] = list(state.get("target_slugs", []))
        return _SUMMARY_OUT

    async def spy_content(state):
        slugs_seen["content"] = list(state.get("target_slugs", []))
        return _CONTENT_OUT

    async def spy_graph(state):
        slugs_seen["graph"] = list(state.get("target_slugs", []))
        return _GRAPH_OUT

    await _run(overrides={
        "router_node": custom_router,
        "summary_retriever_node": spy_summary,
        "content_retriever_node": spy_content,
        "graph_retriever_node": spy_graph,
    })

    for node, slugs in slugs_seen.items():
        assert slugs == ["specific_doc"], (
            f"'{node}' saw target_slugs={slugs}, expected ['specific_doc']"
        )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_rewritten_query_reaches_content_and_graph_retrievers(anyio_backend):
    """rewritten_query from query_rewriter must be visible to content and graph retrievers."""
    marker = "Alice and Bob discuss the budget?"
    queries_seen: dict = {}

    async def custom_rewriter(state):
        return {"rewritten_query": marker}

    async def spy_content(state):
        queries_seen["content"] = state.get("rewritten_query", "")
        return _CONTENT_OUT

    async def spy_graph(state):
        queries_seen["graph"] = state.get("rewritten_query", "")
        return _GRAPH_OUT

    await _run(overrides={
        "query_rewriter_node": custom_rewriter,
        "content_retriever_node": spy_content,
        "graph_retriever_node": spy_graph,
    })

    assert queries_seen.get("content") == marker, (
        f"content_retriever saw '{queries_seen.get('content')}', expected '{marker}'"
    )
    assert queries_seen.get("graph") == marker, (
        f"graph_retriever saw '{queries_seen.get('graph')}', expected '{marker}'"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_retrieved_context_from_aggregator_reaches_generator(anyio_backend):
    """retrieved_context set by the aggregator must be visible to the generator."""
    marker = "UNIQUE_CONTEXT_MARKER_xyz"
    context_seen: list = []

    def custom_aggregator(state):
        return dict(_AGG_OUT, retrieved_context=marker)

    def spy_generator(state):
        context_seen.append(state.get("retrieved_context", ""))
        return _GEN_OUT

    await _run(overrides={
        "context_aggregator_node": custom_aggregator,
        "generator_node": spy_generator,
    })

    assert len(context_seen) == 1, f"Generator called {len(context_seen)} times, expected 1"
    assert context_seen[0] == marker, (
        f"Generator received '{context_seen[0]}', expected '{marker}'"
    )


# ---------------------------------------------------------------------------
# 6. Error isolation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_graceful_retriever_failure_does_not_block_pipeline(anyio_backend):
    """
    If one retriever returns an empty result (simulating a graceful error),
    the other retrievers, aggregator, and generator must still be called once.
    """
    async def empty_content(state):
        return {"partial_passages": {}}   # No chunks — simulates DB/search failure

    _, _, call_counts = await _run(overrides={"content_retriever_node": empty_content})

    assert call_counts["aggregator"] == 1, "Aggregator must still fire once"
    assert call_counts["generator"] == 1, "Generator must still fire once"
    assert call_counts["summary"] == 1, "summary_retriever must still have been called"
    assert call_counts["graph"] == 1, "graph_retriever must still have been called"
