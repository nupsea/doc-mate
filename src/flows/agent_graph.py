"""
LangGraph orchestration for Doc-Mate agent.
Router-Retriever-Generator (RRG) architecture.
"""

import operator
import asyncio
from typing import Annotated, List, TypedDict, Dict, Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.flows.router import QueryRouter
from src.flows.document_query import search_document_content
from src.flows.confidence import RetrievalConfidenceAssessor

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

    # Final Aggregated Input
    retrieved_context: str

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
    
    print(f"[RRG] Router: intent={decision.get('intent')}, strategy={decision.get('strategy')}, slugs={slugs}")
    
    # Entity-Based Document Discovery (Content-Based Routing)
    if entities:
        try:
            from src.graph.store import PostgresGraphStore
            graph_store = PostgresGraphStore()
            discovered_slugs = graph_store.find_docs_by_entities(entities)
            
            if discovered_slugs:
                # Add discovered slugs, avoiding duplicates
                added = []
                for ds in discovered_slugs:
                    if ds not in slugs:
                        slugs.append(ds)
                        added.append(ds)
                if added:
                    print(f"[RRG] Discovered docs via entities: {added}")
        except Exception as e:
            print(f"[RRG] Entity-based routing failed: {e}")
    
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
            print(f"[RRG] Failed to fetch doc types: {e}")

    return {
        "intent": decision.get("intent", "chat"),
        "strategy": decision.get("strategy", "search"),
        "target_slugs": slugs,
        "target_doc_types": doc_types,
        "entities": entities,
        "router_entities": entities
    }

# -----------------------------------------------------------------------------
# 2. Specialized Retrieval Nodes
# -----------------------------------------------------------------------------

# ... imports ...

async def summary_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Fetches high-level summaries. Vital for 'summary' and 'hybrid' strategies."""
    if state["strategy"] not in ["summary", "hybrid"]:
        return {"partial_summaries": {}}
        
    from src.content.store import PgresStore
    
    def _fetch_summaries():
        store = PgresStore()
        results = {}
        for slug in state["target_slugs"]:
            summary = store.get_document_summary(slug)
            if summary:
                results[slug] = summary
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
        
    query = state["messages"][-1].content
    hint_entities = state.get("router_entities", [])
    
    from src.flows.document_query import get_adjacent_chunks
    
    def _fetch_content():
        results = {}
        for slug in state["target_slugs"]:
            doc_type = state["target_doc_types"].get(slug, "book")
            
            limit = 5
            if doc_type == "conversation":
                limit = 3
                
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
    """
    if state["intent"] not in ["explore", "compare"] and state["strategy"] != "hybrid":
        return {"partial_relations": {}}
        
    from src.graph.store import PostgresGraphStore
    
    def _fetch_graph():
        store = PostgresGraphStore()
        results = {}
        
        entities = state.get("entities", [])
        if not entities:
            import re
            q = state["messages"][-1].content
            # Fallback: simple extraction if router missed them
            entities = re.findall(r'\b[A-Z][a-z]+\b', q)
            print(f"[RRG] Graph: Fallback entity extraction: {entities}")
        else:
            print(f"[RRG] Graph: Using router entities: {entities}")

        for slug in state["target_slugs"]:
            doc_type = state["target_doc_types"].get(slug, "book")
            
            if doc_type in ["tech_doc", "report"]:
                continue
                
            doc_id = store._resolve_doc_id(slug)
            if not doc_id:
                continue
                
            name_map = store.find_entities_by_names(doc_id, entities)
            if not name_map and entities:
                print(f"[RRG] Graph: No entities found in DB for doc {slug} matching {entities}")
                
            rel_list = []
            
            for name, eid in name_map.items():
                hops = 2 if doc_type == "conversation" else 1
                related = store.find_related_entities(eid, hops=hops)
                    
                for r in related:
                    rel_list.append(f"{name} {r['relation_type']} {r['name']} ({r['entity_type']})")
            
            if rel_list:
                print(f"[RRG] Graph: Found {len(rel_list)} relationships in {slug}")
                results[slug] = list(set(rel_list))
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

    for slug in sorted(set(list(summaries.keys()) + list(passages.keys()) + list(relations.keys()))):
        dtype = doc_types.get(slug, "unknown")
        combined.append(f"=== DOCUMENT: {slug} (Type: {dtype}) ===")

        if slug in summaries:
            combined.append(f"\n[OVERVIEW]\n{summaries[slug]}")

        if slug in relations:
            combined.append("\n[KNOWN RELATIONSHIPS]")
            unique_rels = sorted(list(set(relations[slug])))
            combined.extend(unique_rels[:15])

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
                print(f"[RRG] Episode context failed for {slug}: {e}")

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

                # Format based on doc type
                meta = p.get("metadata", {})
                if dtype == "conversation":
                    timestamp = meta.get("timestamp_start", "")
                    speakers = meta.get("speakers", [])
                    speaker_str = ", ".join(speakers) if speakers else "Unknown"
                    combined.append(f"{i}. [{timestamp}] {speaker_str}:\n   {p['text']}\n   [Source: {slug}, ID: {p['id']}]")
                else:
                    section = meta.get("chapter", meta.get("section", ""))
                    section_str = f", Section: {section}" if section else ""
                    combined.append(f"{i}. {p['text']}\n   [Source: {slug}, Chunk: {p['id']}{section_str}]")

        combined.append("\n" + "-"*40 + "\n")

    # Assess confidence
    assessor = RetrievalConfidenceAssessor()
    confidence = assessor.assess(
        query=query,
        query_entities=query_entities,
        retrieved_chunks=all_chunks,
        retrieval_scores=all_scores if all_scores else None,
    )

    print(f"[RRG] Confidence: {confidence.coverage_summary}")

    return {
        "retrieved_context": "\n".join(combined),
        "confidence_level": confidence.level.value,
        "confidence_score": confidence.score,
        "evidence_gaps": confidence.evidence_gaps,
        "query_type": confidence.query_type.value,
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
        for t in ["conversation", "script", "tech_doc", "report", "social_chat"]:
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
    print(f"[RRG] Generator: confidence={confidence_level}, score={confidence_score:.2f}, query_type={query_type}, gaps={evidence_gaps}")

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

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("summary_retriever", summary_retriever_node)
    workflow.add_node("content_retriever", content_retriever_node)
    workflow.add_node("graph_retriever", graph_retriever_node)
    workflow.add_node("aggregator", context_aggregator_node)
    workflow.add_node("generator", generator_node)
    
    # Define Edges
    workflow.set_entry_point("router")
    
    # Parallel Retrieval
    workflow.add_edge("router", "summary_retriever")
    workflow.add_edge("router", "content_retriever")
    workflow.add_edge("router", "graph_retriever")
    
    # Fan-in to Aggregator
    workflow.add_edge("summary_retriever", "aggregator")
    workflow.add_edge("content_retriever", "aggregator")
    workflow.add_edge("graph_retriever", "aggregator")
    
    # Final Flow
    workflow.add_edge("aggregator", "generator")
    workflow.add_edge("generator", END)
    
    return workflow.compile()

agent_graph = create_agent_graph()
