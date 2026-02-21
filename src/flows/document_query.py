"""
Document query script - plain Python.
"""

import logging
from datetime import datetime
from src.content.store import PgresStore
from src.search.adaptive import AdaptiveRetriever

logger = logging.getLogger(__name__)


def _parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object. Returns None if parsing fails."""
    if not timestamp_str:
        return None
    try:
        # Try common timestamp formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None


def _diversify_conversation_results(chunks, target_count=None):
    """
    Diversify conversation search results to avoid repetition.

    Strategies:
    1. Temporal spreading - don't cluster results in same time window
    2. Speaker balancing - if speaker metadata exists, balance across speakers
    3. Limit results per time window

    Args:
        chunks: List of chunk dictionaries with text and metadata
        target_count: Number of results to return (defaults to len(chunks)//2)

    Returns:
        Diversified list of chunks
    """
    if not chunks or len(chunks) <= 3:
        return chunks  # Too few to diversify

    if target_count is None:
        target_count = max(5, int(len(chunks) * 0.7))  # Keep ~70% for diversity

    # Extract timestamps and speakers
    chunks_with_meta = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        timestamp_str = (
            metadata.get("timestamp")
            or metadata.get("created_at")
            or metadata.get("timestamp_start")
        )
        timestamp = _parse_timestamp(timestamp_str)
        speaker = metadata.get("speaker") or metadata.get("author")

        chunks_with_meta.append({
            "chunk": chunk,
            "timestamp": timestamp,
            "speaker": speaker,
            "original_rank": len(chunks_with_meta)  # Preserve search ranking
        })

    # Sort by timestamp if available (temporal ordering)
    chunks_with_meta.sort(key=lambda x: (
        x["timestamp"] if x["timestamp"] else datetime.max,
        x["original_rank"]
    ))

    # Diversify selection with dynamic constraints
    selected = []
    speaker_counts = {}
    last_timestamp = None

    for item in chunks_with_meta:
        # Relax constraints progressively when under quota
        under_quota = len(selected) < target_count // 2

        # Check speaker balance (relax limit when under quota)
        speaker = item["speaker"]
        max_per_speaker = 3 if under_quota else 2
        if speaker:
            if speaker_counts.get(speaker, 0) >= max_per_speaker:
                continue  # Skip - too many from this speaker
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        # Check temporal spacing (relax gap when under quota)
        min_gap = 60 if under_quota else 300
        timestamp = item["timestamp"]
        if timestamp and last_timestamp:
            gap = abs((timestamp - last_timestamp).total_seconds())
            if gap < min_gap:
                continue  # Skip - too close in time to previous result

        # Select this chunk
        selected.append(item["chunk"])
        last_timestamp = timestamp

        if len(selected) >= target_count:
            break

    logger.debug(f"Diversified {len(chunks)} conversation results to {len(selected)}")
    return selected


def validate_document_exists(doc_identifier: str | int):
    """Validate that document exists in database."""
    store = PgresStore()
    doc_id = store._resolve_doc_id(doc_identifier)

    if not doc_id:
        raise ValueError(f"Document not found: {doc_identifier}")

    return {"doc_id": doc_id, "identifier": doc_identifier}


# Global singleton for retriever to avoid reloading models
_RETRIEVER = None

def get_retriever():
    """Get or create singleton retriever instance."""
    global _RETRIEVER
    if _RETRIEVER is None:
        logger.info("Initializing AdaptiveRetriever (singleton)...")
        _RETRIEVER = AdaptiveRetriever()
    return _RETRIEVER

def preload_retriever():
    """Preload the retriever model at startup."""
    logger.info("Preloading retriever model...")
    get_retriever()
    logger.info("Retriever model preloaded successfully.")


def search_document_content(query: str, doc_identifier: str | int, limit: int = 5, hint_entities: list[str] = None):
    """
    Search document content using hybrid search (BM25 + vector).
    Returns chunk IDs and fetches chunk text from Qdrant.
    """
    logger.info(f"Searching for: '{query}' in document: {doc_identifier}")

    try:
        # Validate document exists and get doc_id
        store = PgresStore()
        doc_id = store._resolve_doc_id(doc_identifier)
        if not doc_id:
            return {
                "query": query,
                "document": doc_identifier,
                "chunk_ids": [],
                "chunks": [],
                "num_results": 0,
                "error": f"Document not found: {doc_identifier}",
            }

        # Resolve doc_type early so search can adapt weights
        row = store.execute("SELECT doc_type FROM documents WHERE doc_id = %s", (doc_id,), fetch="one")
        doc_type = row[0] if row else None

        retriever = get_retriever()
        # Pass doc_identifier as doc_slug to filter search results DURING search, not after
        chunk_ids = retriever.id_search(
            query, topk=limit, doc_slug=doc_identifier,
            hint_entities=hint_entities, doc_type=doc_type,
        )

        logger.info(f"Hybrid search returned {len(chunk_ids)} chunk IDs for document '{doc_identifier}'")
        for i, cid in enumerate(chunk_ids[:10], 1):  # Show first 10
            logger.debug(f"  {i}. {cid}")

        # Fetch full chunk details from Qdrant by ID
        chunks_full = retriever.vec.get_chunks_by_ids(chunk_ids)

        # Preserve full metadata for citations and type-aware formatting
        chunks_with_text = []
        for chunk in chunks_full:
            text = chunk["text"]
            chunks_with_text.append(
                {
                    "id": chunk["id"],
                    "text": text, # Pass full text, aggregator will handle truncation if needed
                    "metadata": chunk.get("metadata", {}),
                }
            )

        logger.info(f"Retrieved {len(chunks_with_text)} chunks with text for '{doc_identifier}'")

        # Apply diversity filtering for conversation documents
        # This prevents repetitive results from the same time window/speaker
        if doc_type == "conversation" and len(chunks_with_text) > 5:
            logger.info("Applying diversity filtering for conversation document")
            chunks_with_text = _diversify_conversation_results(chunks_with_text)

        return {
            "query": query,
            "document": doc_identifier,
            "chunk_ids": [c["id"] for c in chunks_with_text],
            "chunks": chunks_with_text,
            "num_results": len(chunks_with_text),
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "query": query,
            "document": doc_identifier,
            "chunk_ids": [],
            "chunks": [],
            "num_results": 0,
            "error": str(e),
        }


def get_document_chapters(doc_identifier: str | int):
    """Get all chapter/section summaries for a document."""
    store = PgresStore()
    chapters = store.get_all_chapter_summaries(doc_identifier)

    return {
        "chapters": [
            {"chapter_id": ch_id, "summary": summary} for ch_id, summary in chapters
        ],
        "num_chapters": len(chapters),
    }


def get_document_summary(doc_identifier: str | int):
    """Get overall document summary."""
    store = PgresStore()
    summary = store.get_document_summary(doc_identifier)

    return {"summary": summary, "length": len(summary) if summary else 0}


def get_adjacent_chunks(doc_slug: str, chunk_id: str, before: int = 1, after: int = 1) -> list[dict]:
    """
    Fetch adjacent chunks (before and after) based on chunk_id sequence.
    Useful for expanding conversation context.
    
    Args:
        doc_slug: Slug of the document
        chunk_id: The pivot chunk ID
        before: Number of chunks to fetch before
        after: Number of chunks to fetch after
    """
    try:
        # Assuming chunk_ids are sequential like "slug_01_001_hash"
        # Format: {slug}_{section}_{index}_{hash}
        parts = chunk_id.split('_')
        if len(parts) < 3:
            return []
            
        slug = parts[0]
        section = parts[1]
            
        # We need the prefix to rebuild IDs
        # Note: hash part is unique to each chunk, so we can't easily guess it.
        # INSTEAD: In the DB-backed world, we can query BM25_doc_lens for sequential IDs.
        
        store = PgresStore()
        # Fetch IDs in the same section, sorted by ID
        rows = store.execute(
            "SELECT chunk_id FROM bm25_doc_lens WHERE chunk_id LIKE %s ORDER BY chunk_id",
            (f"{slug}_{section}_%",),
            fetch="all"
        )
        
        if not rows:
            return []
            
        all_ids = [row[0] for row in rows]
        try:
            pivot_idx = all_ids.index(chunk_id)
        except ValueError:
            return []
            
        # Calculate range
        start = max(0, pivot_idx - before)
        end = min(len(all_ids), pivot_idx + after + 1)
        
        target_ids = [all_ids[i] for i in range(start, end) if i != pivot_idx]
            
        if not target_ids:
            return []

        # Fetch from Qdrant
        retriever = get_retriever()
        chunks = retriever.vec.get_chunks_by_ids(target_ids)
        
        # Sort results to match target_ids order
        id_to_chunk = {c['id']: c for c in chunks}
        sorted_chunks = [id_to_chunk[tid] for tid in target_ids if tid in id_to_chunk]
        
        return sorted_chunks
    except Exception as e:
        logger.warning(f"Failed to get adjacent chunks for {chunk_id}: {e}")
        return []

def query_document(
    doc_identifier: str | int,
    query: str = None,
    include_chapters: bool = True,
    include_document_summary: bool = True,
    search_limit: int = 5,
):
    """
    Query a document with optional search and summary retrieval.
    """
    logger.info(f"Starting query for document: {doc_identifier}")

    validation = validate_document_exists(doc_identifier)
    logger.info(f"Document validated - ID: {validation['doc_id']}")

    results = {"doc_id": validation["doc_id"]}

    if query:
        results["search"] = search_document_content(query, doc_identifier, search_limit)
        logger.info(
            f"Search completed - Found {results['search']['num_results']} results"
        )

    if include_chapters:
        results["chapters"] = get_document_chapters(doc_identifier)
        logger.info(
            f"Retrieved {results['chapters']['num_chapters']} chapter summaries"
        )

    if include_document_summary:
        results["document_summary"] = get_document_summary(doc_identifier)
        logger.info(
            f"Retrieved document summary ({results['document_summary']['length']} chars)"
        )

    logger.info("Query complete")
    return results


if __name__ == "__main__":
    # Example 1: Get all summaries
    result1 = query_document(
        doc_identifier="mma", include_chapters=True, include_document_summary=True
    )
    print(f"\nQuery result: Found {result1['chapters']['num_chapters']} chapters")
    print(f"Document summary preview: {result1['document_summary']['summary'][:150]}...")

    # Example 2: Search with summaries
    result2 = query_document(
        doc_identifier="ody",
        query="odysseus journey home",
        include_chapters=False,
        include_document_summary=True,
    )
    print(f"\nSearch results for: '{result2['search']['query']}'")
    print(f"Found {result2['search']['num_results']} matching chunks:\n")
    for i, chunk in enumerate(result2["search"]["chunks"], 1):
        print(f"{i}. [{chunk['id']}]")
        print(f"   {chunk['text']}\n")
