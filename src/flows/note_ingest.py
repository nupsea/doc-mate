"""
Lightweight indexing pipeline for notes.

Notes skip the heavy document ingestion path (no file parsing, no chapter
detection, no ingestion profiles). Instead: chunk markdown -> index BM25 +
vector -> optionally extract graph entities.

Called on both note creation and update.
"""

import asyncio
import hashlib
import logging
from src.content.store import PgresStore
from src.search.hybrid import FusionRetriever

logger = logging.getLogger(__name__)


def _chunk_markdown(content: str, slug: str, max_tokens: int = 300) -> list[dict]:
    """Split markdown content into chunks suitable for indexing.

    Strategy: split on double-newlines (paragraph boundaries), then merge
    small paragraphs until reaching max_tokens. This preserves natural
    thought boundaries in notes.
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    chunks = []
    current_text = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(para.split())

        if current_tokens + para_tokens > max_tokens and current_text:
            # Flush current chunk
            chunk_id = _make_chunk_id(slug, len(chunks))
            chunks.append({
                "id": chunk_id,
                "text": current_text.strip(),
                "num_tokens": current_tokens,
                "metadata": {"doc_type": "note", "chunk_index": len(chunks)},
            })
            current_text = para
            current_tokens = para_tokens
        else:
            separator = "\n\n" if current_text else ""
            current_text += separator + para
            current_tokens += para_tokens

    # Flush remaining
    if current_text.strip():
        chunk_id = _make_chunk_id(slug, len(chunks))
        chunks.append({
            "id": chunk_id,
            "text": current_text.strip(),
            "num_tokens": current_tokens,
            "metadata": {"doc_type": "note", "chunk_index": len(chunks)},
        })

    return chunks


def _make_chunk_id(slug: str, index: int) -> str:
    """Generate a deterministic chunk ID for a note chunk."""
    raw = f"{slug}_{index:03d}"
    hash_suffix = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{slug}_{index:02d}_001_{hash_suffix}"


def _clear_old_index(slug: str, doc_id: int):
    """Remove old BM25 + vector + graph data for a note before re-indexing."""
    store = PgresStore()

    # BM25 cleanup (cascade from documents doesn't help here since we keep the doc)
    store.execute(
        "DELETE FROM bm25_index WHERE doc_id = %s", (doc_id,), commit=True
    )
    store.execute(
        "DELETE FROM bm25_doc_lens WHERE doc_id = %s", (doc_id,), commit=True
    )

    # Graph cleanup
    from src.graph.store import PostgresGraphStore
    graph_store = PostgresGraphStore()
    graph_store.delete_graph_for_doc(doc_id)

    # Qdrant vector cleanup
    from src.flows.document_ingest import _cleanup_qdrant_vectors
    _cleanup_qdrant_vectors(slug)


async def index_note(slug: str, content: str, extract_graph: bool = True):
    """Index a note for search. Called on create AND update.

    Steps:
        1. Clear old index data
        2. Chunk the markdown content
        3. Build BM25 + vector indexes
        4. Optionally extract graph entities (if content is substantial)
        5. Update document metadata (num_chunks, num_chars)
    """
    store = PgresStore()
    doc_id = store._resolve_doc_id(slug)
    if not doc_id:
        logger.warning("Document not found for slug: %s", slug)
        return

    # 1. Clear old indexes
    await asyncio.to_thread(_clear_old_index, slug, doc_id)

    # 2. Chunk the content
    chunks = _chunk_markdown(content, slug)
    if not chunks:
        logger.warning("No chunks generated for note: %s", slug)
        return

    logger.info("Indexing note '%s': %d chunks, %d chars", slug, len(chunks), len(content))

    # 3. Build search indexes (BM25 + vector) in parallel
    retriever = FusionRetriever()
    await asyncio.gather(
        asyncio.to_thread(retriever.bm25.build_index, chunks, doc_id),
        asyncio.to_thread(retriever.vec.build_index, chunks),
    )

    # 4. Update document metadata
    store.execute(
        "UPDATE documents SET num_chunks = %s, num_chars = %s WHERE doc_id = %s",
        (len(chunks), len(content), doc_id),
        commit=True,
    )

    # 5. Graph extraction (optional, skip for very short notes)
    total_tokens = sum(c.get("num_tokens", 0) for c in chunks)
    if extract_graph and total_tokens >= 100:
        try:
            await _extract_note_graph(doc_id, chunks)
        except Exception as e:
            # Graph extraction failure should not block note saving
            logger.warning("Graph extraction failed (non-fatal): %s", e)
    else:
        logger.debug("Skipping graph extraction (%d tokens < 100 threshold)", total_tokens)

    logger.info("Done: %d chunks indexed for '%s'", len(chunks), slug)


async def _extract_note_graph(doc_id: int, chunks: list):
    """Extract entities and relationships from note chunks."""
    from src.graph.extractor import EntityExtractor
    from src.graph.resolver import EntityResolver
    from src.graph.store import PostgresGraphStore

    extractor = EntityExtractor(provider="openai", batch_size=4)
    resolver = EntityResolver()
    graph_store = PostgresGraphStore()

    entities, relationships = await extractor.extract_from_chunks(chunks, doc_type="note")

    if not entities and not relationships:
        logger.debug("No entities extracted from note")
        return

    entities, relationships = resolver.resolve_kinship_references(entities, relationships)
    resolved_entities, name_mapping = resolver.resolve(entities)

    for rel in relationships:
        if rel.source_entity in name_mapping:
            rel.source_entity = name_mapping[rel.source_entity]
        if rel.target_entity in name_mapping:
            rel.target_entity = name_mapping[rel.target_entity]

    e_ids = graph_store.store_entities(doc_id, resolved_entities)
    r_count = graph_store.store_relationships(doc_id, relationships)
    logger.info("Graph: %d entities, %d relationships", len(e_ids), r_count)
