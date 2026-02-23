"""
Integration tests for conversation metadata retrieval.

Requires: Postgres + Qdrant running (make dev), with at least one
          conversation document ingested (e.g. sample_teams_meeting).
"""
import pytest
from src.content.store import PgresStore


def _conversation_slug() -> str | None:
    """Return the slug of any ingested conversation document, or None."""
    try:
        store = PgresStore()
        row = store.execute(
            "SELECT slug FROM books WHERE doc_type = 'conversation' LIMIT 1",
            fetch="one",
        )
        return row[0] if row else None
    except Exception:
        return None


@pytest.fixture(scope="module")
def conversation_slug():
    slug = _conversation_slug()
    if slug is None:
        pytest.skip("No conversation document found in database")
    return slug


def test_search_results_include_required_fields(conversation_slug):
    """search() must return dicts with id, text, score, and metadata keys."""
    from src.search.vec import SemanticRetriever

    retriever = SemanticRetriever()
    results = retriever.search("meeting discussion", topk=3, book_slug=conversation_slug)

    if not results:
        pytest.skip(f"No search results found for slug '{conversation_slug}'")

    for result in results:
        assert "id" in result, f"Missing 'id' in result: {result.keys()}"
        assert "text" in result, f"Missing 'text' in result: {result.keys()}"
        assert "score" in result, f"Missing 'score' in result: {result.keys()}"
        assert "metadata" in result, f"Missing 'metadata' in result: {result.keys()}"


def test_get_chunks_by_ids_includes_metadata(conversation_slug):
    """get_chunks_by_ids() must return dicts with id, text, and metadata keys."""
    from src.search.vec import SemanticRetriever

    retriever = SemanticRetriever()
    chunk_ids = retriever.id_search("meeting", topk=3)

    if not chunk_ids:
        pytest.skip("No chunks found via id_search")

    chunks = retriever.get_chunks_by_ids(chunk_ids)
    for chunk in chunks:
        assert "id" in chunk, f"Missing 'id' in chunk: {chunk.keys()}"
        assert "text" in chunk, f"Missing 'text' in chunk: {chunk.keys()}"
        assert "metadata" in chunk, f"Missing 'metadata' in chunk: {chunk.keys()}"


def test_conversation_chunks_have_speaker_or_timestamp_metadata(conversation_slug):
    """Conversation chunks should carry speaker or timestamp metadata when ingested."""
    from src.search.vec import SemanticRetriever

    retriever = SemanticRetriever()
    results = retriever.search("discussion", topk=5, book_slug=conversation_slug)

    if not results:
        pytest.skip(f"No results for slug '{conversation_slug}'")

    # At least one chunk should carry conversation-specific metadata
    has_convo_meta = any(
        "speakers" in r.get("metadata", {})
        or "timestamp_start" in r.get("metadata", {})
        or "speaker" in r.get("metadata", {})
        for r in results
    )
    assert has_convo_meta, (
        "No conversation metadata (speakers/timestamp) found in any retrieved chunk. "
        "Ensure the document was ingested with a conversation parser."
    )
