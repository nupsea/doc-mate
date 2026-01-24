"""
LangChain tool definitions for Doc-Mate.
Wraps core logic from document_query.py into agent-ready tools.
"""

from typing import List
from langchain_core.tools import tool
from src.flows.document_query import (
    search_document_content,
    get_document_summary,
    get_document_chapters
)

# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------

@tool
def search_multiple_documents(query: str, doc_identifiers: List[str], limit_per_doc: int = 3) -> str:
    """
    Search across multiple documents (2-5) simultaneously. Required for comparisons 
    ('Compare X and Y') or multi-source queries. More efficient than multiple single searches.
    
    Args:
        query: Specific search terms (concrete concepts work better than abstract ones).
        doc_identifiers: List of document SLUGS to search (e.g. ['ili', 'ody']).
        limit_per_doc: Results per document (default 3).
    """
    results = []
    
    for doc_slug in doc_identifiers:
        # Re-use the existing logic
        result = search_document_content(query, doc_slug, limit=limit_per_doc)
        
        if result.get("error"):
            results.append(f"Error searching '{doc_slug}': {result['error']}")
            continue
            
        chunks = result.get("chunks", [])
        if not chunks:
            results.append(f"No results found in '{doc_slug}'.")
            continue
            
        # Format chunks for the LLM
        doc_output = f"--- Results from '{doc_slug}' ---"
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            # Construct a clear citation string
            citation = f"[Source: {doc_slug}, Chunk: {chunk['id']}"
            if "chapter" in meta:
                citation += f", Section: {meta['chapter']}"
            if "speaker" in meta:
                citation += f", Speaker: {meta['speaker']}"
            citation += "]"
            
            doc_output += f"\n{i}. {chunk['text']}\n   {citation}\n"
        results.append(doc_output)

    return "\n\n".join(results)


@tool
def search_document(query: str, doc_identifier: str, limit: int = 5) -> str:
    """
    Search a single document. For comparisons or multiple sources, use 'search_multiple_documents'.
    
    Args:
        query: Specific search terms.
        doc_identifier: The document SLUG (e.g. 'ili').
        limit: Number of results (default 5).
    """
    result = search_document_content(query, doc_identifier, limit=limit)
    
    if result.get("error"):
        return f"Error: {result['error']}"
        
    chunks = result.get("chunks", [])
    if not chunks:
        return f"No results found for '{query}' in document '{doc_identifier}'."
        
    output = f"--- Search Results for '{query}' in '{doc_identifier}' ---\\n"
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        citation = f"[Source: {doc_identifier}, Chunk: {chunk['id']}"
        if "chapter" in meta:
            citation += f", Section: {meta['chapter']}"
        citation += "]"
        
        output += f"\n{i}. {chunk['text']}\n   {citation}\n"
        
    return output


@tool
def get_summary(doc_identifier: str) -> str:
    """
    Get the high-level summary of a document.
    Use for questions like "What is this book about?" or "Summarize X".
    
    Args:
        doc_identifier: The document SLUG.
    """
    result = get_document_summary(doc_identifier)
    summary = result.get("summary")
    if not summary:
        return f"No summary available for document '{doc_identifier}'."
    return f"--- Summary of '{doc_identifier}' ---\\n{summary}"


@tool
def get_structure(doc_identifier: str) -> str:
    """
    Get the chapter/section structure of a document.
    
    Args:
        doc_identifier: The document SLUG.
    """
    result = get_document_chapters(doc_identifier)
    chapters = result.get("chapters", [])
    
    if not chapters:
        return f"No structure information available for '{doc_identifier}'."
        
    output = f"--- Structure of '{doc_identifier}' ---\\n"
    for chap in chapters:
        output += f"- {chap['chapter_id']}: {chap['summary']}\n"
    return output

# List of tools to bind to the LLM
ALL_TOOLS = [search_multiple_documents, search_document, get_summary, get_structure]
