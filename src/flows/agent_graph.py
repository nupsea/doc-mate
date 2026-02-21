"""
LangGraph orchestration for Doc-Mate agent.
Router-Retriever-Generator (RRG) architecture.
"""

import logging
import operator
import re
import asyncio
from typing import Annotated, List, TypedDict, Dict, Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.flows.router import QueryRouter
from src.flows.document_query import search_document_content
from src.flows.confidence import RetrievalConfidenceAssessor

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _extractive_condense(text: str, query_terms: List[str], max_chars: int = 500) -> str:
    """Keep sentences that mention query terms or entities. Fast, no LLM."""
    if len(text) <= max_chars:
        return text
    sentences = _SENTENCE_RE.split(text)
    if len(sentences) <= 1:
        return text[:max_chars] + " [...]"
    terms_lower = [t.lower() for t in query_terms if len(t) > 2]
    kept = []
    kept_len = 0
    for s in sentences:
        s_lower = s.lower()
        if any(t in s_lower for t in terms_lower):
            kept.append(s)
            kept_len += len(s)
    # Always include the first sentence for context if nothing matched
    if not kept:
        kept.append(sentences[0])
        kept_len = len(sentences[0])
    # If kept text is still very long, trim to budget
    result = " ".join(kept)
    if len(result) > max_chars:
        result = result[:max_chars] + " [...]"
    if len(kept) < len(sentences):
        result += " [...]"
    return result


# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    """Deterministic state for Modular Type-Aware RRG flow."""
    messages: Annotated[List[BaseMessage], operator.add]
    provider: str
    model: str
    selected_doc_slug: str

    # Router Outputs
    intent: str
    strategy: str
    target_slugs: List[str]
    target_doc_types: Dict[str, str]  # {slug: doc_type}
    entities: List[str]
    router_entities: List[str]  # Entities extracted by the router

    # Partial Retrieval Results
    partial_summaries: Dict[str, str]
    partial_passages: Dict[str, List[Dict]]
    partial_relations: Dict[str, List[str]]

    # Confidence Assessment
    confidence_level: str  # "high", "medium", "low"
    confidence_score: float  # 0.0 to 1.0
    evidence_gaps: List[str]  # Entities not found (for ENTITY queries)
    query_type: str  # "broad", "entity", "inference", "factual"

    # Query Rewriting
    rewritten_query: str  # Original query with vague references resolved to entity names

    # Final Aggregated Input
    retrieved_context: str
    source_refs: List[Dict]  # [{slug, chunk_id, snippet}] for note provenance

# -----------------------------------------------------------------------------
# 1. Router Node
# -----------------------------------------------------------------------------

def router_node(state: AgentState) -> Dict[str, Any]:
    """Classifies query and extracts target documents/entities with metadata."""
    query = state["messages"][-1].content
    selected = state.get("selected_doc_slug")
    history = state.get("messages", [])  # Full history
    
    router = QueryRouter(provider=state["provider"], model=state["model"])
    decision = router.route(query, selected, history=history)
    
    slugs = decision.get("slugs", [])
    entities = decision.get("entities", [])
    intent = decision.get("intent", "chat")

    logger.info("Router: intent=%s, strategy=%s, slugs=%s", intent, decision.get('strategy'), slugs)

    # When the user has explicitly selected a document, constrain to it unless
    # they are explicitly asking to compare across documents.
    if selected and selected != "none" and intent != "compare":
        if slugs != [selected]:
            logger.info("Router: constraining to selected doc '%s' (was: %s)", selected, slugs)
        slugs = [selected]
    elif entities:
        # Entity-Based Document Discovery only when no explicit doc is pinned
        try:
            from src.graph.store import PostgresGraphStore
            graph_store = PostgresGraphStore()
            discovered_slugs = graph_store.find_docs_by_entities(entities)

            if discovered_slugs:
                added = []
                for ds in discovered_slugs:
                    if ds not in slugs:
                        slugs.append(ds)
                        added.append(ds)
                if added:
                    logger.info("Discovered docs via entities: %s", added)
        except Exception as e:
            logger.warning("Entity-based routing failed: %s", e)
    
    # Enrich with doc_types from DB (Crucial for Type-Aware Retrieval)
    doc_types = {}
    if slugs:
        try:
            from src.content.store import PgresStore
            store = PgresStore()
            rows = store.execute("SELECT slug, doc_type FROM documents WHERE slug = ANY(%s)", (slugs,), fetch="all")
            if rows:
                for row in rows:
                    doc_types[row[0]] = row[1]
        except Exception as e:
            logger.warning("Failed to fetch doc types: %s", e)

    return {
        "intent": decision.get("intent", "chat"),
        "strategy": decision.get("strategy", "search"),
        "target_slugs": slugs,
        "target_doc_types": doc_types,
        "entities": entities,
        "router_entities": entities
    }

# -----------------------------------------------------------------------------
# 1b. Query Rewriter Node
# -----------------------------------------------------------------------------

# Detects queries that contain vague references that could be resolved with
# known entity names (works for any doc type: books, conversations, scripts…)
_VAGUE_REF_RE = re.compile(
    r'\b('
    r'he|she|they|them|their|him|her'          # pronouns
    r'|both|each\s+other'                       # mutual references
    r'|the\s+(?:two|three|\d+|main|other|first|second)'  # "the two X"
    r'|the\s+(?:speaker|character|protagonist|antagonist'
    r'|narrator|author|hero|villain|person|people|subject)s?'  # generic roles
    r')\b',
    re.IGNORECASE,
)


async def query_rewriter_node(state: AgentState) -> Dict[str, Any]:
    """
    Resolves vague references in the user query to concrete entity names
    drawn from the knowledge graph of the target document(s).

    Works for any document type:
      - Conversations: "the two speakers"  → "Sarah and Mike"
      - Books:         "both characters"   → "Achilles and Hector"
      - Scripts:       "he"                → "Hamlet"

    Runs after routing (target docs known) but before retrieval so the
    enriched query flows into BM25, vector, and graph search.
    Skips the LLM call entirely when no vague references are detected.
    """
    query = state["messages"][-1].content
    target_slugs = state.get("target_slugs", [])

    # Fast path: no vague references → nothing to rewrite
    if not target_slugs or not _VAGUE_REF_RE.search(query):
        return {"rewritten_query": query}

    def _rewrite() -> str:
        try:
            from src.graph.store import PostgresGraphStore
            store = PostgresGraphStore()
            entity_lines = []
            for slug in target_slugs:
                doc_id = store._resolve_doc_id(slug)
                if not doc_id:
                    continue
                rows = store.execute(
                    "SELECT name, entity_type FROM graph_entities "
                    "WHERE doc_id = %s ORDER BY entity_id LIMIT 30",
                    (doc_id,),
                    fetch="all",
                )
                if rows:
                    names = ", ".join(f"{name} ({etype})" for name, etype in rows)
                    entity_lines.append(f"[{slug}] {names}")
        except Exception as e:
            logger.warning("Query rewriter: entity fetch failed: %s", e)
            return query

        if not entity_lines:
            return query  # No graph data — nothing to resolve against

        entity_context = "\n".join(entity_lines)
        prompt = (
            "You are a search query optimizer for a document retrieval system.\n\n"
            f"Known entities in the target document(s):\n{entity_context}\n\n"
            f"Original query: {query}\n\n"
            "Rewrite the query to replace vague or ambiguous references "
            "(pronouns, generic roles like 'the two speakers', 'both characters', "
            "'the protagonist', 'he', 'she', 'they') with the actual entity names "
            "from the list above — but only where you can confidently infer the mapping.\n"
            "If the query is already specific, or you cannot confidently resolve a "
            "reference, leave that part unchanged.\n"
            "Return ONLY the rewritten query, nothing else."
        )

        from langchain_core.messages import HumanMessage as HM
        from src.llm.config import LLMConfig
        cfg = LLMConfig.from_env()
        if state["provider"] == "openai":
            llm = ChatOpenAI(model=cfg.router_model, temperature=0)
        else:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=cfg.ollama_model, temperature=0, base_url=cfg.ollama_base_url)

        result = llm.invoke([HM(content=prompt)]).content.strip()
        # Sanity guard: reject wildly long or empty responses
        if not result or len(result) > len(query) * 5:
            return query
        return result

    rewritten = await asyncio.to_thread(_rewrite)
    if rewritten != query:
        logger.info("Query rewritten: '%s' → '%s'", query, rewritten)
    else:
        logger.info("Query rewriter: no changes for '%s'", query)
    return {"rewritten_query": rewritten}


# -----------------------------------------------------------------------------
# 2. Specialized Retrieval Nodes
# -----------------------------------------------------------------------------

# ... imports ...

async def summary_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Fetches high-level summaries. Vital for 'summary' and 'hybrid' strategies."""
    if state["strategy"] not in ["summary", "hybrid"]:
        logger.info("Summary retriever: skipping (strategy=%s)", state["strategy"])
        return {"partial_summaries": {}}

    from src.content.store import PgresStore

    def _fetch_summaries():
        store = PgresStore()
        results = {}
        for slug in state["target_slugs"]:
            summary = store.get_document_summary(slug)
            if summary:
                results[slug] = summary
                logger.info("Summary retriever: fetched summary for '%s' (%d chars)", slug, len(summary))
            else:
                logger.info("Summary retriever: no summary found for '%s'", slug)
        return results

    results = await asyncio.to_thread(_fetch_summaries)
    return {"partial_summaries": results}


async def content_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Type-Aware Content Retrieval.
    - Books: Standard Hybrid Search
    - Conversations: Hybrid Search + Context Expansion (Window retrieval)
    - Tech Docs: Hybrid Search
    """
    if state["strategy"] not in ["search", "hybrid"]:
        return {"partial_passages": {}}

    # Use the rewritten query if vague references were resolved; fall back to original
    query = state.get("rewritten_query") or state["messages"][-1].content
    hint_entities = state.get("router_entities", [])
    
    from src.flows.document_query import get_adjacent_chunks
    
    def _fetch_content():
        results = {}
        num_docs = len(state["target_slugs"])
        for slug in state["target_slugs"]:
            doc_type = state["target_doc_types"].get(slug, "book")

            if doc_type == "conversation":
                limit = 3
            elif num_docs >= 3:
                limit = 3
            elif num_docs == 2:
                limit = 4
            else:
                limit = 5
                
            search_res = search_document_content(query, slug, limit=limit, hint_entities=hint_entities)
            chunks = search_res.get("chunks", [])
            
            # Context Expansion for Conversations
            if doc_type == "conversation" and chunks:
                expanded_chunks = []
                seen_ids = set()
                
                for chunk in chunks:
                    if chunk['id'] in seen_ids:
                        continue
                    
                    adjacent = get_adjacent_chunks(slug, chunk['id'], before=1, after=1)
                    for adj in adjacent:
                        if adj['id'] not in seen_ids:
                            expanded_chunks.append(adj)
                            seen_ids.add(adj['id'])
                    
                    expanded_chunks.append(chunk)
                    seen_ids.add(chunk['id'])
                
                expanded_chunks.sort(key=lambda x: x['id'])
                results[slug] = expanded_chunks
            else:
                if chunks:
                    results[slug] = chunks
        return results

    results = await asyncio.to_thread(_fetch_content)
    return {"partial_passages": results}


async def graph_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetches Knowledge Graph relationships.
    Skips for 'tech_doc' or 'report' where narrative graphs usually don't exist.
    Runs for: explore/compare intents, hybrid strategy, or any query with router-extracted entities.
    """
    entities = state.get("entities", [])
    should_run = (
        state["intent"] in ["explore", "compare"]
        or state["strategy"] == "hybrid"
        or bool(entities)
    )
    if not should_run:
        logger.info(
            "Graph retriever: skipping (intent=%s, strategy=%s, no entities extracted)",
            state["intent"], state["strategy"],
        )
        return {"partial_relations": {}}

    from src.graph.store import PostgresGraphStore

    def _fetch_graph():
        store = PostgresGraphStore()
        results = {}

        graph_entities = state.get("entities", [])
        # Use the rewritten query for fallback entity extraction — it has vague
        # references already replaced with actual names (e.g. "Sarah and Mike")
        q = state.get("rewritten_query") or state["messages"][-1].content
        if not graph_entities:
            graph_entities = re.findall(r'\b[A-Z][a-z]+\b', q)
            logger.info("Graph: Fallback entity extraction from rewritten query: %s", graph_entities)
        else:
            logger.info("Graph: Using router entities: %s", graph_entities)

        for slug in state["target_slugs"]:
            doc_type = state["target_doc_types"].get(slug, "book")

            if doc_type in ["tech_doc", "report"]:
                continue

            doc_id = store._resolve_doc_id(slug)
            if not doc_id:
                continue

            name_map = store.find_entities_by_names(doc_id, graph_entities)
            if not name_map:
                logger.info("Graph: No entities matched in DB for doc '%s' (searched: %s)", slug, graph_entities)
            else:
                logger.info("Graph: Matched %d entities in doc '%s': %s", len(name_map), slug, list(name_map.keys()))

            rel_list = []

            for name, eid in name_map.items():
                hops = 2 if doc_type == "conversation" else 1
                related = store.find_related_entities(eid, hops=hops)

                for r in related:
                    rel_list.append(f"{name} {r['relation_type']} {r['name']} ({r['entity_type']})")

            if rel_list:
                logger.info("Graph: Found %d relationships in '%s'", len(rel_list), slug)
                results[slug] = list(set(rel_list))
            else:
                logger.info("Graph: No relationships found in '%s'", slug)
        return results

    results = await asyncio.to_thread(_fetch_graph)
    return {"partial_relations": results}

# -----------------------------------------------------------------------------
# 3. Aggregator Node
# -----------------------------------------------------------------------------

def context_aggregator_node(state: AgentState) -> Dict[str, Any]:
    """
    Merges partial results with deduplication, type-formatting, and confidence assessment.
    """
    summaries = state.get("partial_summaries", {})
    passages = state.get("partial_passages", {})
    relations = state.get("partial_relations", {})
    doc_types = state.get("target_doc_types", {})
    query_entities = state.get("router_entities", [])
    query = state["messages"][-1].content

    # Collect all chunks for confidence assessment
    all_chunks = []
    all_scores = []

    combined = []
    seen_content_hashes = set()

    all_slugs = sorted(set(list(summaries.keys()) + list(passages.keys()) + list(relations.keys())))
    num_target_docs = len(all_slugs)

    for slug in all_slugs:
        dtype = doc_types.get(slug, "unknown")
        combined.append(f"=== DOCUMENT: {slug} (Type: {dtype}) ===")

        if slug in summaries:
            overview_text = summaries[slug]
            # Truncate overviews when comparing 3+ documents
            if num_target_docs >= 3:
                words = overview_text.split()
                if len(words) > 150:
                    overview_text = " ".join(words[:150]) + " [...]"
            combined.append(f"\n[OVERVIEW]\n{overview_text}")

        if slug in relations:
            combined.append("\n[KNOWN RELATIONSHIPS]")
            unique_rels = sorted(list(set(relations[slug])))
            # Filter to relationships relevant to the query
            query_lower = query.lower()
            entity_names_lower = [e.lower() for e in query_entities]
            filtered_rels = [
                r for r in unique_rels
                if any(name in r.lower() for name in entity_names_lower)
                or any(word in r.lower() for word in query_lower.split() if len(word) > 3)
            ]
            combined.extend((filtered_rels or unique_rels)[:10])

        # Episode context for conversation documents
        if dtype == "conversation" and slug in passages:
            try:
                from src.graph.store import PostgresGraphStore
                from src.content.store import PgresStore
                ep_store = PostgresGraphStore()
                doc_id = PgresStore()._resolve_doc_id(slug)
                if doc_id:
                    chunk_ids = [p['id'] for p in passages[slug]]
                    episodes = ep_store.get_episodes_by_chunk_ids(doc_id, chunk_ids)
                    if episodes:
                        # Deduplicate by (speaker, topic)
                        seen_ep = set()
                        ep_lines = []
                        for ep in episodes:
                            key = (ep.get("speaker", ""), ep.get("topic", ""))
                            if key in seen_ep:
                                continue
                            seen_ep.add(key)
                            stance_str = f" (stance: {ep['stance']})" if ep.get("stance") else ""
                            speaker = ep.get("speaker", "Unknown")
                            topic = ep.get("topic", "general")
                            summary = ep.get("summary", "")
                            ep_lines.append(f"- {speaker} on '{topic}'{stance_str}: {summary}")
                        if ep_lines:
                            combined.append("\n[EPISODE CONTEXT]")
                            combined.extend(ep_lines)
            except Exception as e:
                logger.warning("Episode context failed for %s: %s", slug, e)

        if slug in passages:
            combined.append("\n[RELEVANT PASSAGES]")
            for i, p in enumerate(passages[slug], 1):
                text_hash = hash(p['text'][:50])
                if text_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(text_hash)

                all_chunks.append(p)
                if "score" in p:
                    all_scores.append(p["score"])

                # Condense passage text for the prompt (full text already in all_chunks).
                # Conversations/scripts keep full text; others get extractive condense.
                display_text = p['text']
                if dtype not in ("conversation", "script"):
                    query_terms = list(query_entities) + [w for w in query.split() if len(w) > 3]
                    display_text = _extractive_condense(display_text, query_terms)

                # Format based on doc type
                meta = p.get("metadata", {})
                if dtype == "conversation":
                    # Use `or ""` to coerce None (key present but null) to empty string
                    timestamp = meta.get("timestamp_start") or ""
                    speakers = meta.get("speakers") or []
                    speaker_str = ", ".join(speakers) if speakers else "Unknown"
                    time_str = f", Time: {timestamp}" if timestamp else ""
                    # Match the citation format described in config.yaml so the LLM copies it faithfully
                    combined.append(f"{i}. {display_text}\n   [Speakers: {speaker_str}{time_str}, Source: {slug}, ID: {p['id']}]")
                else:
                    section = meta.get("chapter", meta.get("section", ""))
                    section_str = f", Section: {section}" if section else ""
                    combined.append(f"{i}. {display_text}\n   [Source: {slug}, Chunk: {p['id']}{section_str}]")

        # For conversation docs: flag when queried entity names don't appear as
        # speaker names. This surfaces the contact-name vs real-name discrepancy so the
        # LLM can reconcile them using the [NAME NOTE] instruction in config.yaml.
        if dtype == "conversation" and query_entities and slug in passages:
            speakers_in_slug: set = set()
            for p in passages[slug]:
                for spk in (p.get("metadata", {}).get("speakers") or []):
                    speakers_in_slug.add(spk.strip().lower())
            entities_not_as_speakers = [
                e for e in query_entities
                if e.lower() not in speakers_in_slug
            ]
            if entities_not_as_speakers and speakers_in_slug:
                entity_list = ", ".join(entities_not_as_speakers)
                speaker_list = ", ".join(f"'{s}'" for s in sorted(speakers_in_slug))
                combined.append(
                    f"\n[NAME NOTE] '{entity_list}' appears in conversation text "
                    f"but not in [Speakers: ...] metadata for these passages. "
                    f"Contact names used in these passages: {speaker_list}. "
                    f"One of these contact names may refer to the same person."
                )

        combined.append("\n" + "-"*40 + "\n")

    # Build source_refs for note provenance
    source_refs = []
    for slug_key in passages:
        dtype = doc_types.get(slug_key, "unknown")
        for p in passages[slug_key]:
            ref: dict = {
                "slug": slug_key,
                "chunk_id": p["id"],
                "snippet": p["text"][:150],
            }
            if dtype == "conversation":
                meta = p.get("metadata", {})
                ts = meta.get("timestamp_start") or ""
                spk = meta.get("speakers") or []
                if ts:
                    ref["timestamp"] = ts
                if spk:
                    ref["speakers"] = spk
            source_refs.append(ref)

    # For summary-only strategy, summaries ARE the evidence — use them as synthetic
    # chunks for confidence assessment so the score isn't unfairly low.
    chunks_for_assessment = all_chunks
    if not all_chunks and summaries:
        chunks_for_assessment = [
            {"id": f"{slug}_summary", "text": summary}
            for slug, summary in summaries.items()
            if summary
        ]

    # Assess confidence
    assessor = RetrievalConfidenceAssessor()
    confidence = assessor.assess(
        query=query,
        query_entities=query_entities,
        retrieved_chunks=chunks_for_assessment,
        retrieval_scores=all_scores if all_scores else None,
    )

    logger.info("Confidence: %s", confidence.coverage_summary)

    return {
        "retrieved_context": "\n".join(combined),
        "confidence_level": confidence.level.value,
        "confidence_score": confidence.score,
        "evidence_gaps": confidence.evidence_gaps,
        "query_type": confidence.query_type.value,
        "source_refs": source_refs,
    }

# -----------------------------------------------------------------------------
# 4. Generator Node
# -----------------------------------------------------------------------------

def generator_node(state: AgentState) -> Dict[str, Any]:
    """Final answer generation with confidence-adaptive prompting."""
    context = state.get("retrieved_context", "")
    messages = state["messages"]
    doc_types = state.get("target_doc_types", {})
    confidence_level = state.get("confidence_level", "medium")
    confidence_score = state.get("confidence_score", 0.5)
    evidence_gaps = state.get("evidence_gaps", [])

    # Load prompt configuration
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "mcp_client" / "prompts" / "config.yaml"
    with open(config_path) as f:
        prompt_config = yaml.safe_load(f)

    # Determine persona based on doc type
    primary_type = "book"
    if doc_types:
        types = list(doc_types.values())
        for t in ["conversation", "script", "tech_doc", "report"]:
            if t in types:
                primary_type = t
                break

    personas = prompt_config.get("personas", {})
    persona = personas.get(primary_type, personas.get("default", "You are a helpful document assistant."))

    # Build evidence note based on gaps
    evidence_notes = prompt_config.get("evidence_notes", {})
    evidence_note = ""
    if evidence_gaps:
        gap_template = evidence_notes.get("entity_gaps", "")
        evidence_note = gap_template.format(gap_list=", ".join(evidence_gaps))
    elif confidence_level == "low":
        if not context.strip():
            evidence_note = evidence_notes.get("no_results", "")
        else:
            evidence_note = evidence_notes.get("sparse_results", "")

    # Select confidence-based prompt template
    confidence_prompts = prompt_config.get("confidence_prompts", {})
    base_prompt = confidence_prompts.get(confidence_level, confidence_prompts.get("medium", ""))

    # Assemble final system prompt with all placeholders
    system_content = base_prompt.format(
        persona=persona,
        context=context if context else "(No relevant passages found)",
        evidence_note=evidence_note,
    )

    # Add query-type specific guidance
    query_type = state.get("query_type", "broad")
    query_type_guidance = prompt_config.get("query_type_guidance", {})
    if query_type in query_type_guidance:
        system_content += "\n" + query_type_guidance[query_type]

    # Log for debugging
    logger.debug("Generator: confidence=%s, score=%.2f, query_type=%s, gaps=%s", confidence_level, confidence_score, query_type, evidence_gaps)

    from src.llm.config import LLMConfig
    config = LLMConfig.from_env()
    
    if state["provider"] == "openai":
        # Use configured generator model (defaults to OPENAI_MODEL)
        llm = ChatOpenAI(model=config.openai_model, temperature=0.1)
    else:
        from langchain_ollama import ChatOllama
        # Use configured ollama settings
        llm = ChatOllama(
            model=config.ollama_model, 
            temperature=0, 
            base_url=config.ollama_base_url
        )

    input_messages = [SystemMessage(content=system_content)] + messages
    return {"messages": [llm.invoke(input_messages)]}

# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------

def create_agent_graph() -> CompiledStateGraph:
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("summary_retriever", summary_retriever_node)
    workflow.add_node("content_retriever", content_retriever_node)
    workflow.add_node("graph_retriever", graph_retriever_node)
    workflow.add_node("aggregator", context_aggregator_node)
    workflow.add_node("generator", generator_node)

    # Define Edges
    workflow.set_entry_point("router")

    # router → query_rewriter (always); rewriter is a fast no-op when no vague refs found.
    # All three retrievers branch from query_rewriter so every path to the aggregator
    # has the same depth (router→rewriter→retriever→aggregator = 3 hops).
    # This is critical: unequal depths caused LangGraph to fire the aggregator multiple
    # times (once per completed path), producing empty/incorrect generator output.
    workflow.add_edge("router", "query_rewriter")
    workflow.add_edge("query_rewriter", "summary_retriever")
    workflow.add_edge("query_rewriter", "content_retriever")
    workflow.add_edge("query_rewriter", "graph_retriever")

    # Fan-in to Aggregator (all three paths now arrive at equal depth)
    workflow.add_edge("summary_retriever", "aggregator")
    workflow.add_edge("content_retriever", "aggregator")
    workflow.add_edge("graph_retriever", "aggregator")
    
    # Final Flow
    workflow.add_edge("aggregator", "generator")
    workflow.add_edge("generator", END)
    
    return workflow.compile()

agent_graph = create_agent_graph()
