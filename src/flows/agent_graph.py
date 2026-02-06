"""
LangGraph orchestration for Doc-Mate agent.
Router-Retriever-Generator (RRG) architecture.
"""

import operator
import os
from typing import Annotated, List, TypedDict, Dict, Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.flows.router import QueryRouter
from src.flows.document_query import search_document_content

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
    router_entities: List[str] # Entities extracted by the router
    
    # Partial Retrieval Results
    partial_summaries: Dict[str, str]
    partial_passages: Dict[str, List[Dict]]
    partial_relations: Dict[str, List[str]]
    
    # Final Aggregated Input
    retrieved_context: str

# -----------------------------------------------------------------------------
# 1. Router Node
# -----------------------------------------------------------------------------

def router_node(state: AgentState) -> Dict[str, Any]:
    """Classifies query and extracts target documents/entities with metadata."""
    query = state["messages"][-1].content
    selected = state.get("selected_doc_slug")
    
    router = QueryRouter(provider=state["provider"], model=state["model"])
    decision = router.route(query, selected)
    
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
            with store.conn.cursor() as cur:
                # Safe SQL with ANY
                cur.execute("SELECT slug, doc_type FROM documents WHERE slug = ANY(%s)", (slugs,))
                for row in cur.fetchall():
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

def summary_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Fetches high-level summaries. Vital for 'summary' and 'hybrid' strategies."""
    if state["strategy"] not in ["summary", "hybrid"]:
        return {"partial_summaries": {}}
        
    from src.content.store import PgresStore
    store = PgresStore()
    results = {}
    
    for slug in state["target_slugs"]:
        summary = store.get_document_summary(slug)
        if summary:
            results[slug] = summary
            
    return {"partial_summaries": results}


def content_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Type-Aware Content Retrieval.
    - Books: Standard Hybrid Search
    - Conversations: Hybrid Search + Context Expansion (Window retrieval)
    - Tech Docs: Hybrid Search
    """
    if state["strategy"] not in ["search", "hybrid"]:
        return {"partial_passages": {}}
        
    query = state["messages"][-1].content
    results = {}
    hint_entities = state.get("router_entities", [])
    
    from src.flows.document_query import get_adjacent_chunks
    
    for slug in state["target_slugs"]:
        doc_type = state["target_doc_types"].get(slug, "book")
        
        limit = 5
        if doc_type == "conversation":
            limit = 3  # Fewer initial hits, but we expand them
            
        search_res = search_document_content(query, slug, limit=limit, hint_entities=hint_entities)
        chunks = search_res.get("chunks", [])
        
        # Context Expansion for Conversations
        if doc_type == "conversation" and chunks:
            expanded_chunks = []
            seen_ids = set()
            
            for chunk in chunks:
                if chunk['id'] in seen_ids:
                    continue
                
                # Add preceding context
                adjacent = get_adjacent_chunks(slug, chunk['id'], window=1)
                for adj in adjacent:
                    if adj['id'] not in seen_ids:
                        expanded_chunks.append(adj)
                        seen_ids.add(adj['id'])
                
                # Add main chunk
                expanded_chunks.append(chunk)
                seen_ids.add(chunk['id'])
            
            # Sort by ID to restore chronological flow
            expanded_chunks.sort(key=lambda x: x['id'])
            results[slug] = expanded_chunks
        else:
            if chunks:
                results[slug] = chunks
            
    return {"partial_passages": results}


def graph_retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Fetches Knowledge Graph relationships.
    Skips for 'tech_doc' or 'report' where narrative graphs usually don't exist.
    """
    # Only run for explore/compare or hybrid strategy
    if state["intent"] not in ["explore", "compare"] and state["strategy"] != "hybrid":
        return {"partial_relations": {}}
        
    from src.graph.store import PostgresGraphStore
    store = PostgresGraphStore()
    results = {}
    
    entities = state.get("entities", [])
    if not entities:
        import re
        query = state["messages"][-1].content
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)

    for slug in state["target_slugs"]:
        doc_type = state["target_doc_types"].get(slug, "book")
        
        if doc_type in ["tech_doc", "report"]:
            continue
            
        doc_id = store._resolve_doc_id(slug)
        if not doc_id or not entities:
            continue
            
        name_map = store.find_entities_by_names(doc_id, entities)
        rel_list = []
        
        for name, eid in name_map.items():
            # Conversations often have deep, implicit chains (A dates B, brother of C)
            # So we use 2-hops by default for conversations, 1-hop for others.
            hops = 2 if doc_type == "conversation" else 1
            related = store.find_related_entities(eid, hops=hops)
                
            for r in related:
                rel_list.append(f"{name} {r['relation_type']} {r['name']} ({r['entity_type']})")
        
        if rel_list:
            results[slug] = list(set(rel_list)) # Dedupe relations
            
    return {"partial_relations": results}

# -----------------------------------------------------------------------------
# 3. Aggregator Node
# -----------------------------------------------------------------------------

def context_aggregator_node(state: AgentState) -> Dict[str, Any]:
    """
    Merges partial results with Deduplication and Type-Formatting.
    """
    summaries = state.get("partial_summaries", {})
    passages = state.get("partial_passages", {})
    relations = state.get("partial_relations", {})
    doc_types = state.get("target_doc_types", {})
    
    combined = []
    
    # Global dedupe set for text content
    seen_content_hashes = set()
    
    for slug in sorted(set(list(summaries.keys()) + list(passages.keys()) + list(relations.keys()))):
        dtype = doc_types.get(slug, "unknown")
        combined.append(f"=== DOCUMENT: {slug} (Type: {dtype}) ===")
        
        if slug in summaries:
            combined.append(f"\n[OVERVIEW]\n{summaries[slug]}")
            
        if slug in relations:
            combined.append("\n[KNOWN RELATIONSHIPS]")
            unique_rels = sorted(list(set(relations[slug])))
            combined.extend(unique_rels[:15]) # Increased limit for deep graphs
            
        if slug in passages:
            combined.append("\n[RELEVANT PASSAGES]")
            for i, p in enumerate(passages[slug], 1):
                # Content Deduplication
                text_hash = hash(p['text'][:50]) # Simple hash of start
                if text_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(text_hash)
                
                citation_extras = ""
                meta = p.get("metadata", {})
                
                if dtype == "conversation":
                    # Conversation format: [Time] Speaker: Message
                    timestamp = meta.get("timestamp_start", "")
                    speaker = meta.get("speaker", "Unknown")
                    combined.append(f"{i}. [{timestamp}] {speaker}: {p['text']}\n   Source: {slug}, ID: {p['id']}")
                else:
                    # Book/Standard format
                    if "chapter" in meta:
                        citation_extras = f", Section: {meta['chapter']}"
                    combined.append(f"{i}. {p['text']}\n   Source: {slug}, Chunk: {p['id']}{citation_extras}")
        
        combined.append("\n" + "-"*40 + "\n")

    return {"retrieved_context": "\n".join(combined)}

# -----------------------------------------------------------------------------
# 4. Generator Node
# -----------------------------------------------------------------------------

def generator_node(state: AgentState) -> Dict[str, Any]:
    """Final answer generation with Type-Aware Persona."""
    context = state.get("retrieved_context", "")
    messages = state["messages"]
    doc_types = state.get("target_doc_types", {})
    
    # Determine primary doc type
    primary_type = "book"
    if doc_types:
        # Simple heuristic: priority to conversation -> script -> book
        types = list(doc_types.values())
        if "conversation" in types:
            primary_type = "conversation"
        elif "script" in types:
            primary_type = "script"
        elif "tech_doc" in types:
            primary_type = "tech_doc"
    
    # Select Persona
    persona_map = {
        "conversation": "You are analyzing a transcript. Pay attention to speakers, timestamps, and the flow of dialogue.",
        "script": "You are analyzing a screenplay. Focus on scene descriptions, dialogue, and character actions.",
        "tech_doc": "You are a technical assistant. Be precise with code, APIs, and specifications.",
        "book": "You are a literary assistant. Focus on themes, narrative, and character development."
    }
    persona = persona_map.get(primary_type, persona_map["book"])
    
    if context:
        system_content = (
            f"{persona}\n"
            "Answer the user's question using ONLY the provided context. "
            "Attribute facts to the correct document. "
            "If the answer is not in the context, say the documents do not specify.\n\n"
            f"RETRIEVED CONTEXT:\n{context}"
        )
    else:
        system_content = "You are Doc-Mate. Answer the user's question politely."

    if state["provider"] == "openai":
        llm = ChatOpenAI(model=state["model"], temperature=0.1)
    else:
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_HOST_URL", "http://host.docker.internal:11434")
        llm = ChatOllama(model=state["model"], temperature=0, base_url=base_url)

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
