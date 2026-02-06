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
        target_count = max(5, len(chunks) // 2)  # Return ~half for diversity

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

    # Diversify selection
    selected = []
    speaker_counts = {}
    last_timestamp = None
    MIN_TIME_GAP_SECONDS = 300  # 5 minutes between selected chunks

    for item in chunks_with_meta:
        # Check speaker balance (max 2 per speaker if we have speakers)
        speaker = item["speaker"]
        if speaker:
            if speaker_counts.get(speaker, 0) >= 2:
                continue  # Skip - too many from this speaker
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        # Check temporal spacing
        timestamp = item["timestamp"]
        if timestamp and last_timestamp:
            gap = abs((timestamp - last_timestamp).total_seconds())
            if gap < MIN_TIME_GAP_SECONDS:
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

        retriever = get_retriever()
        # Pass doc_identifier as doc_slug to filter search results DURING search, not after
        chunk_ids = retriever.id_search(
            query, topk=limit, doc_slug=doc_identifier, hint_entities=hint_entities
        )

        logger.debug(f"Hybrid search returned {len(chunk_ids)} chunk IDs for document '{doc_identifier}':")
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

        logger.debug(f"Retrieved {len(chunks_with_text)} chunks with text")

        # Apply diversity filtering for conversation documents
        # This prevents repetitive results from the same time window/speaker
        with store.conn.cursor() as cur:
            cur.execute("SELECT doc_type FROM documents WHERE doc_id = %s", (doc_id,))
            result_row = cur.fetchone()
            doc_type = result_row[0] if result_row else None

        if doc_type == "conversation" and len(chunks_with_text) > 5:
            logger.debug("Applying diversity filtering for conversation document")
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


def get_adjacent_chunks(doc_slug: str, chunk_id: str, window: int = 2) -> list[dict]:
    """
    Fetch adjacent chunks (before and after) based on chunk_id sequence.
    Useful for expanding conversation context.
    """
    try:
        # Assuming chunk_ids are sequential like "slug_01_001", "slug_01_002"
        # We can reconstruct IDs or use a DB lookup if chunks are stored with order.
        # For now, we'll try a Qdrant ID lookup pattern if possible, or fall back to DB.
        
        # Parse the ID to get the index
        # Format: {slug}_chk_{index} or similar. Let's assume standard format.
        parts = chunk_id.rsplit('_', 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return []
            
        base_id = parts[0]
        current_idx = int(parts[1])
        
        target_ids = []
        for i in range(current_idx - window, current_idx + window + 1):
            if i == current_idx or i < 0:
                continue
            # Zero-pad the index to match 6-digit standard (e.g., 000012)
            target_ids.append(f"{base_id}_{i:06d}")
            
        # Fetch from Qdrant
        retriever = get_retriever()
        chunks = retriever.vec.get_chunks_by_ids(target_ids)
        return chunks
    except Exception as e:
        logger.warning(f"Failed to get adjacent chunks: {e}")
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
