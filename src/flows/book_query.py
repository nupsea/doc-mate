"""
Book query script - plain Python.
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


def validate_book_exists(book_identifier: str | int):
    """Validate that book exists in database."""
    store = PgresStore()
    book_id = store._resolve_book_id(book_identifier)

    if not book_id:
        raise ValueError(f"Book not found: {book_identifier}")

    return {"book_id": book_id, "identifier": book_identifier}


def search_book_content(query: str, book_identifier: str | int, limit: int = 5):
    """
    Search book content using hybrid search (BM25 + vector).
    Returns chunk IDs and fetches chunk text from Qdrant.
    """
    logger.info(f"Searching for: '{query}' in book: {book_identifier}")

    try:
        # Validate book exists and get book_id
        store = PgresStore()
        book_id = store._resolve_book_id(book_identifier)
        if not book_id:
            return {
                "query": query,
                "book": book_identifier,
                "chunk_ids": [],
                "chunks": [],
                "num_results": 0,
                "error": f"Book not found: {book_identifier}",
            }

        retriever = AdaptiveRetriever()
        # Pass book_identifier as book_slug to filter search results DURING search, not after
        chunk_ids = retriever.id_search(
            query, topk=limit, book_slug=book_identifier
        )

        logger.debug(f"Hybrid search returned {len(chunk_ids)} chunk IDs for book '{book_identifier}':")
        for i, cid in enumerate(chunk_ids[:10], 1):  # Show first 10
            logger.debug(f"  {i}. {cid}")

        # Fetch full chunk details from Qdrant by ID
        chunks_full = retriever.vec.get_chunks_by_ids(chunk_ids)

        # Truncate text and preserve metadata for citations
        chunks_with_text = []
        for chunk in chunks_full:
            text = chunk["text"]
            chunks_with_text.append(
                {
                    "id": chunk["id"],
                    "text": text[:800] + "..." if len(text) > 800 else text,
                    "metadata": chunk.get("metadata", {}),  # Keep metadata for citations
                }
            )

        logger.debug(f"Retrieved {len(chunks_with_text)} chunks with text")

        # Apply diversity filtering for conversation documents
        # This prevents repetitive results from the same time window/speaker
        with store.conn.cursor() as cur:
            cur.execute("SELECT doc_type FROM books WHERE book_id = %s", (book_id,))
            result_row = cur.fetchone()
            doc_type = result_row[0] if result_row else None

        if doc_type == "conversation" and len(chunks_with_text) > 5:
            logger.debug("Applying diversity filtering for conversation document")
            chunks_with_text = _diversify_conversation_results(chunks_with_text)

        return {
            "query": query,
            "book": book_identifier,
            "chunk_ids": [c["id"] for c in chunks_with_text],
            "chunks": chunks_with_text,
            "num_results": len(chunks_with_text),
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "query": query,
            "book": book_identifier,
            "chunk_ids": [],
            "chunks": [],
            "num_results": 0,
            "error": str(e),
        }


def get_chapter_summaries(book_identifier: str | int):
    """Get all chapter summaries for a book."""
    store = PgresStore()
    chapters = store.get_all_chapter_summaries(book_identifier)

    return {
        "chapters": [
            {"chapter_id": ch_id, "summary": summary} for ch_id, summary in chapters
        ],
        "num_chapters": len(chapters),
    }


def get_book_summary(book_identifier: str | int):
    """Get overall book summary."""
    store = PgresStore()
    summary = store.get_book_summary(book_identifier)

    return {"summary": summary, "length": len(summary) if summary else 0}


def query_book(
    book_identifier: str | int,
    query: str = None,
    include_chapters: bool = True,
    include_book_summary: bool = True,
    search_limit: int = 5,
):
    """
    Query a book with optional search and summary retrieval.
    """
    logger.info(f"Starting query for book: {book_identifier}")

    validation = validate_book_exists(book_identifier)
    logger.info(f"Book validated - ID: {validation['book_id']}")

    results = {"book_id": validation["book_id"]}

    if query:
        results["search"] = search_book_content(query, book_identifier, search_limit)
        logger.info(
            f"Search completed - Found {results['search']['num_results']} results"
        )

    if include_chapters:
        results["chapters"] = get_chapter_summaries(book_identifier)
        logger.info(
            f"Retrieved {results['chapters']['num_chapters']} chapter summaries"
        )

    if include_book_summary:
        results["book_summary"] = get_book_summary(book_identifier)
        logger.info(
            f"Retrieved book summary ({results['book_summary']['length']} chars)"
        )

    logger.info("Query complete")
    return results


if __name__ == "__main__":
    # Example 1: Get all summaries
    result1 = query_book(
        book_identifier="mma", include_chapters=True, include_book_summary=True
    )
    print(f"\nQuery result: Found {result1['chapters']['num_chapters']} chapters")
    print(f"Book summary preview: {result1['book_summary']['summary'][:150]}...")

    # Example 2: Search with summaries
    result2 = query_book(
        book_identifier="ody",
        query="odysseus journey home",
        include_chapters=False,
        include_book_summary=True,
    )
    print(f"\nSearch results for: '{result2['search']['query']}'")
    print(f"Found {result2['search']['num_results']} matching chunks:\n")
    for i, chunk in enumerate(result2["search"]["chunks"], 1):
        print(f"{i}. [{chunk['id']}]")
        print(f"   {chunk['text']}\n")
