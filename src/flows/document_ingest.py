"""
Document ingestion pipeline - supports books, scripts, conversations, tech docs, reports.

Backward compatible with book ingestion.
"""

import asyncio
import time
from pathlib import Path

from src.content.reader import GutenbergReader, PDFReader
from src.content.parsers import get_parser
from src.content.store import PgresStore
from src.llm.generator import SummaryGenerator
from src.search.hybrid import FusionRetriever
from src.graph.extractor import EntityExtractor
from src.graph.resolver import EntityResolver
from src.graph.store import PostgresGraphStore


def validate_inputs(slug: str, file_path: str, title: str, force_update: bool = False):
    """Validate inputs before processing."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not slug or not slug.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid slug format: {slug}")

    store = PgresStore()
    exists = store.document_exists(slug)

    if exists and not force_update:
        raise ValueError(
            f"Document with slug '{slug}' already exists. Use force_update=True to overwrite."
        )

    return {
        "slug": slug,
        "file_path": file_path,
        "title": title,
        "exists": exists,
        "file_size": path.stat().st_size,
    }


def read_and_parse(
    slug: str,
    file_path: str,
    doc_type: str = 'book',
    split_pattern: str = None,
    max_tokens: int = 500,
    overlap: int = 100,
):
    """
    Read file and parse into chunks using appropriate parser.

    Args:
        doc_type: 'book', 'script', 'conversation', 'tech_doc', 'report'
        For books: Uses old reader for backward compatibility
        For other types: Uses new parsers
    """
    path = Path(file_path)
    file_extension = path.suffix.lower()

    # For books, use old readers (backward compatible)
    if doc_type == 'book':
        if file_extension == ".pdf":
            reader = PDFReader(file_path, slug, split_pattern=split_pattern)
        else:
            reader = GutenbergReader(file_path, slug, split_pattern=split_pattern)
        chunks = reader.parse(max_tokens=max_tokens, overlap=overlap)

    else:
        # For other types, use new parsers
        print(f"[PARSE] Creating parser for {doc_type}...")
        parser = get_parser(file_path, doc_type, slug, split_pattern=split_pattern)
        print("[PARSE] Parsing document structure...")
        parsed = parser.parse()
        print(f"[PARSE] Parsed {len(parsed)} sections. Creating chunks...")
        # Different parsers have different chunking parameters
        try:
            if doc_type == 'conversation':
                chunks = parser.chunk(parsed, max_tokens=max_tokens, overlap_turns=2)
            elif doc_type == 'script':
                chunks = parser.chunk(parsed)  # Scripts use scene-based chunking
            else:
                chunks = parser.chunk(parsed, max_tokens=max_tokens, overlap=overlap)
            print(f"[PARSE] Created {len(chunks)} chunks successfully")
        except Exception as e:
            print(f"[PARSE ERROR] Chunking failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    return {
        "chunks": chunks,
        "num_chunks": len(chunks),
        "num_chars": sum(len(c.get("text", "")) for c in chunks),
        "num_tokens": sum(c.get("num_tokens", 0) for c in chunks),
    }


async def generate_summaries(
    chunks: list, doc_type: str = 'book', ephemeral: bool = False, slug: str = None,
    provider: str = None, summary_sampling: str = "auto", summary_semaphore: int = 2,
):
    """Generate chapter and document summaries.

    For long conversations (100+ chunks), samples ~30 chunks evenly distributed
    across the conversation to save ~95% of summary LLM calls.
    """
    from src.monitoring.tracer import disable_tracing, init_phoenix_tracing, is_phoenix_enabled

    # Disable tracing if ephemeral mode is enabled
    tracing_was_enabled = is_phoenix_enabled()
    if ephemeral and tracing_was_enabled:
        print("[EPHEMERAL] Disabling Phoenix tracing for summarization...")
        disable_tracing()

    try:
        # Determine sampling threshold and size based on mode
        summary_chunks = chunks
        if summary_sampling == "aggressive":
            threshold, sample_size = 50, 20
        else:
            threshold, sample_size = 100, 30

        if doc_type == "conversation" and len(chunks) >= threshold:
            step = len(chunks) / sample_size
            indices = [int(i * step) for i in range(sample_size)]
            summary_chunks = [chunks[i] for i in indices]
            print(f"[SUMMARY] Long conversation: sampling {len(summary_chunks)} of {len(chunks)} chunks for summary")

        gen = SummaryGenerator(doc_type=doc_type, provider=provider)
        gen.semaphore = asyncio.Semaphore(summary_semaphore)
        chapter_summaries, document_summary = await gen.summarize_hierarchy(summary_chunks)

        if slug:
            store_summaries_to_db(slug, chapter_summaries, document_summary)

        return {
            "chapter_summaries": chapter_summaries,
            "document_summary": document_summary,
            "num_chapters": len(chapter_summaries),
        }
    finally:
        # Re-enable tracing if it was disabled
        if ephemeral and tracing_was_enabled:
            print("[EPHEMERAL] Re-enabling Phoenix tracing...")
            init_phoenix_tracing()

def store_document_metadata(
    slug: str,
    title: str,
    author: str,
    num_chunks: int,
    num_chars: int,
    doc_type: str = 'book',
    metadata: dict = None,
    force_update: bool = False,
    is_ephemeral: bool = False,
):
    """Store document metadata to database."""
    store = PgresStore()

    if force_update and store.document_exists(slug):
        # delete_document handles DB cleanup (cascading summaries/graph + explicit BM25)
        store.delete_document(slug)
        # Also purge stale Qdrant vectors for this slug
        _cleanup_qdrant_vectors(slug)

    # Use new store_document method for multi-format support
    doc_id = store.store_document(
        slug=slug,
        title=title,
        doc_type=doc_type,
        author=author,
        num_chunks=num_chunks,
        num_chars=num_chars,
        metadata=metadata,
        is_ephemeral=is_ephemeral
    )
    return {"doc_id": doc_id, "slug": slug}


def _cleanup_qdrant_vectors(slug: str):
    """Remove Qdrant vectors whose payload 'id' matches the slug prefix."""
    try:
        from src.search.vec import SemanticRetriever
        from qdrant_client import models

        vec = SemanticRetriever()
        if not vec.qdrant.collection_exists(vec.COLLECTION):
            return
        vec.qdrant.delete(
            collection_name=vec.COLLECTION,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id", match=models.MatchText(text=slug)
                    )
                ]
            ),
        )
    except Exception as e:
        print(f"[WARNING] Qdrant cleanup for '{slug}' failed: {e}")

def store_summaries_to_db(slug: str, chapter_summaries: list, document_summary: str):
    """Store document summaries to database."""
    store = PgresStore()
    if chapter_summaries and document_summary:
        store.store_summaries(slug, chapter_summaries, document_summary)

async def build_search_indexes(chunks: list, doc_id: int):
    """Build BM25 and vector search indexes in parallel."""
    retriever = FusionRetriever()

    # Run BM25 and Vector indexing concurrently
    # Both methods now handle batching internally
    await asyncio.gather(
        asyncio.to_thread(retriever.bm25.build_index, chunks, doc_id),
        asyncio.to_thread(retriever.vec.build_index, chunks)
    )

    return {
        "bm25_indexed": len(chunks),
        "vector_indexed": len(chunks),
        "new_chunks": len(chunks),
    }


async def _extract_entities_batch(extractor, resolver, store, doc_id, chunks, doc_type, batch_size):
    """Extract entities and relationships from chunks in batches."""
    total_entities_count = 0
    total_rels_count = 0

    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i : i + batch_size]
        print(f"[GRAPH] Processing chunks {i} to {min(i+batch_size, len(chunks))} ({len(chunk_batch)} chunks)...")

        entities, relationships = await extractor.extract_from_chunks(chunk_batch, doc_type)

        if not entities and not relationships:
            print(f"[GRAPH] Batch {i} yielded no entities. Continuing...")
            continue

        entities, relationships = resolver.resolve_kinship_references(entities, relationships)
        resolved_entities, name_mapping = resolver.resolve(entities)

        for rel in relationships:
            if rel.source_entity in name_mapping:
                rel.source_entity = name_mapping[rel.source_entity]
            if rel.target_entity in name_mapping:
                rel.target_entity = name_mapping[rel.target_entity]

        e_ids = store.store_entities(doc_id, resolved_entities)
        r_count = store.store_relationships(doc_id, relationships)
        print(f"[GRAPH] Stored batch: {len(e_ids)} entities, {r_count} relationships")

        total_entities_count += len(e_ids)
        total_rels_count += r_count

    return total_entities_count, total_rels_count


async def _extract_episodes_batch(extractor, store, doc_id, chunks, batch_size):
    """Extract episodes from chunks in batches."""
    episodes_count = 0
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i : i + batch_size]
        episodes = await extractor.extract_episodes(chunk_batch)
        cnt = store.store_episodes(doc_id, episodes)
        episodes_count += cnt
    return episodes_count


async def build_graph_index(
    chunks: list, doc_id: int, doc_type: str = 'book', ephemeral: bool = False,
    graph_provider: str = "openai", graph_coverage: str = "full",
    graph_sample_pct: float = 1.0, graph_batch_size: int = 8, graph_semaphore: int = 2,
):
    """
    Extract, resolve, and store graph entities and relationships incrementally.
    For conversations, runs entity and episode extraction in parallel.
    For long conversations (100+ chunks), generates arc summaries that replace per-chunk chapter summaries.
    """
    if graph_coverage == "skip" or graph_provider == "skip":
        print("[GRAPH] Skipping graph extraction (profile: search_only)")
        return {"entities": 0, "relationships": 0, "episodes": 0, "status": "skipped", "summary_text": ""}

    # Sample chunks if profile requests it
    graph_chunks = chunks
    if graph_coverage == "sampled" and graph_sample_pct < 1.0:
        sample_size = max(1, int(len(chunks) * graph_sample_pct))
        step = len(chunks) / sample_size
        indices = [int(i * step) for i in range(sample_size)]
        graph_chunks = [chunks[i] for i in indices]
        print(f"[GRAPH] Sampled {len(graph_chunks)} of {len(chunks)} chunks ({graph_sample_pct*100:.0f}%)")

    print(f"[GRAPH] Starting graph extraction for {len(graph_chunks)} chunks (provider={graph_provider})...")

    # Initialize components
    model_name = None if graph_provider == "local" else "gpt-4o-mini"
    extractor = EntityExtractor(provider=graph_provider, model_name=model_name, batch_size=graph_batch_size)
    extractor.semaphore = asyncio.Semaphore(graph_semaphore)
    resolver = EntityResolver()
    store = PostgresGraphStore()

    # Clear existing graph for this doc if it exists (re-ingest)
    store.delete_graph_for_doc(doc_id)

    BATCH_SIZE = 40

    if doc_type == "conversation":
        # Run entity extraction and episode extraction in parallel
        print("[GRAPH] Running entity + episode extraction in parallel...")
        (total_entities_count, total_rels_count), episodes_count = await asyncio.gather(
            _extract_entities_batch(extractor, resolver, store, doc_id, graph_chunks, doc_type, BATCH_SIZE),
            _extract_episodes_batch(extractor, store, doc_id, graph_chunks, BATCH_SIZE)
        )
        print(f"[GRAPH] Stored {episodes_count} total episodes")

        # Arc summaries for long conversations (100+ chunks)
        if len(graph_chunks) >= 100:
            print(f"[GRAPH] Long conversation ({len(graph_chunks)} chunks): generating arc summaries...")
            all_episodes = store.get_all_episodes_for_doc(doc_id)
            if all_episodes:
                gen = SummaryGenerator(doc_type=doc_type)
                target_arcs = min(15, max(8, len(all_episodes) // 10))
                arc_summaries = await gen.generate_arc_summaries(all_episodes, target_arcs=target_arcs)
                if arc_summaries:
                    # Replace per-chunk chapter summaries with arc summaries
                    content_store = PgresStore()
                    content_store.execute(
                        "DELETE FROM chapter_summaries WHERE doc_id = %s",
                        (doc_id,),
                        commit=True
                    )
                    content_store.store_summaries(doc_id, arc_summaries, content_store.get_document_summary(doc_id) or "")
                    print(f"[GRAPH] Replaced chapter summaries with {len(arc_summaries)} arc summaries")
    else:
        total_entities_count, total_rels_count = await _extract_entities_batch(
            extractor, resolver, store, doc_id, graph_chunks, doc_type, BATCH_SIZE
        )
        episodes_count = 0

    return {
        "entities": total_entities_count,
        "relationships": total_rels_count,
        "episodes": episodes_count,
        "status": "persisted",
        "summary_text": ""
    }


def verify_ingestion(slug: str, expected_chapters: int, allow_arc_summaries: bool = False):
    """Verify document was ingested correctly.

    Args:
        allow_arc_summaries: If True, skip chapter count matching (arc summaries
            replace per-chunk summaries with a different count).
    """
    store = PgresStore()

    if not store.document_exists(slug):
        raise ValueError(f"Document verification failed: {slug} not found in database")

    if not store.summaries_exist(slug):
        raise ValueError(
            f"Summaries verification failed: no summaries found for {slug}"
        )

    chapters = store.get_all_chapter_summaries(slug)
    actual_chapters = len(chapters)

    if not allow_arc_summaries and actual_chapters != expected_chapters:
        raise ValueError(
            f"Chapter count mismatch: expected {expected_chapters}, got {actual_chapters}"
        )

    document_summary = store.get_document_summary(slug)

    return {
        "status": "success",
        "slug": slug,
        "chapters_verified": actual_chapters,
        "document_summary_length": len(document_summary) if document_summary else 0,
    }


async def ingest_document(
    slug: str,
    file_path: str,
    title: str,
    doc_type: str = 'book',
    author: str = None,
    split_pattern: str = None,
    max_tokens: int = 500,
    overlap: int = 100,
    force_update: bool = False,
    ephemeral: bool = False,
    profile_name: str = None,
):
    """
    Ingest any document type: validate -> parse -> summarize -> store -> build indexes -> verify.

    Args:
        doc_type: 'book', 'script', 'conversation', 'tech_doc', 'report'
        ephemeral: If True, disable Phoenix tracing during summarization
        profile_name: Ingestion profile controlling speed/quality tradeoffs
    """
    from src.flows.ingest_profiles import PROFILES, DEFAULT_PROFILE
    profile = PROFILES.get(profile_name or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])

    t_total = time.time()
    print(f"Starting ingestion for: {title} (type: {doc_type}, slug: {slug})")
    print(f"[INGEST] Profile: {profile.label} (summary={profile.summary_provider}, graph={profile.graph_provider}/{profile.graph_coverage})")

    validation = validate_inputs(slug, file_path, title, force_update)
    print(f"Validation passed - File size: {validation['file_size']} bytes")

    t_parse = time.time()
    parse_result = read_and_parse(slug, file_path, doc_type, split_pattern, max_tokens, overlap)
    parse_elapsed = time.time() - t_parse
    print(
        f"Parsed {parse_result['num_chunks']} chunks, {parse_result['num_chars']} chars ({parse_elapsed:.1f}s)"
    )

    # Extract metadata from parser (for non-book types)
    metadata = None
    if doc_type != 'book':
        try:
            parser = get_parser(file_path, doc_type, slug, split_pattern=split_pattern)
            metadata = parser.extract_metadata()
        except Exception:
            metadata = {}

    # Store initial metadata to get doc_id for graph
    db_result = store_document_metadata(
        slug,
        title,
        author,
        parse_result["num_chunks"],
        parse_result["num_chars"],
        doc_type,
        metadata,
        force_update,
        is_ephemeral=ephemeral
    )
    print(f"Stored metadata to database - Document ID: {db_result['doc_id']}")

    # Run Summaries (LLM), Search (CPU/IO), and Graph (LLM) in PARALLEL
    # SummaryGenerator handles its own semaphore (2 concurrent)
    # EntityExtractor handles its own semaphore (2 concurrent)
    # Search indexing handles its own threads
    t_parallel = time.time()
    print("[INGEST] Starting parallel execution: Summaries | Search Indexes | Knowledge Graph...")

    summary_result, search_result, graph_stats = await asyncio.gather(
        generate_summaries(
            parse_result["chunks"], doc_type=doc_type, ephemeral=ephemeral, slug=slug,
            provider=profile.summary_provider,
            summary_sampling=profile.summary_sampling,
            summary_semaphore=profile.summary_semaphore,
        ),
        build_search_indexes(parse_result["chunks"], db_result["doc_id"]),
        build_graph_index(
            parse_result["chunks"], db_result["doc_id"], doc_type, ephemeral=ephemeral,
            graph_provider=profile.graph_provider,
            graph_coverage=profile.graph_coverage,
            graph_sample_pct=profile.graph_sample_pct,
            graph_batch_size=profile.graph_batch_size,
            graph_semaphore=profile.graph_semaphore,
        )
    )

    parallel_elapsed = time.time() - t_parallel
    print(f"[INGEST] Parallel execution completed in {parallel_elapsed:.1f}s")
    print(
        f"Generated {summary_result['num_chapters']} section summaries + overall summary"
    )
    print(
        f"Built search indexes - BM25: {search_result['bm25_indexed']}, Vector: {search_result['vector_indexed']} chunks"
    )
    print(f"Built Knowledge Graph - Entities: {graph_stats['entities']}, Relations: {graph_stats['relationships']}, Episodes: {graph_stats['episodes']}")

    # For long conversations with arc summaries, chapter counts may differ
    is_long_conversation = doc_type == "conversation" and parse_result["num_chunks"] >= 100
    verify_result = verify_ingestion(slug, summary_result["num_chapters"], allow_arc_summaries=is_long_conversation)

    total_elapsed = time.time() - t_total
    print(f"\n[INGEST] Completed '{title}' ({slug}) in {total_elapsed:.1f}s | "
          f"chunks={parse_result['num_chunks']}, chapters={verify_result['chapters_verified']}, "
          f"entities={graph_stats['entities']}, relations={graph_stats['relationships']}, "
          f"episodes={graph_stats['episodes']} | "
          f"parse={parse_elapsed:.1f}s, parallel={parallel_elapsed:.1f}s")

    return {
        "slug": slug,
        "doc_id": db_result["doc_id"],
        "title": title,
        "doc_type": doc_type,
        "chapters": verify_result["chapters_verified"],
        "chunks": parse_result["num_chunks"],
        "search_indexed": search_result["bm25_indexed"],
        "graph_stats": graph_stats,
        "status": "success",
    }


async def ingest_book(
    slug: str,
    file_path: str,
    title: str,
    author: str = None,
    split_pattern: str = None,
    max_tokens: int = 500,
    overlap: int = 100,
    force_update: bool = False,
):
    """
    Ingest a book (backward compatible).

    DEPRECATED: Use ingest_document() for multi-format support.
    """
    return await ingest_document(
        slug=slug,
        file_path=file_path,
        title=title,
        doc_type='book',
        author=author,
        split_pattern=split_pattern,
        max_tokens=max_tokens,
        overlap=overlap,
        force_update=force_update
    )


if __name__ == "__main__":
    result = asyncio.run(
        ingest_book(
            slug="ody",
            file_path="DATA/the_odyssey.txt",
            title="The Odyssey",
            author="Homer",
            split_pattern=r"^(?:BOOK [IVXLCDM]+)\s*\n",
            force_update=False,
        )
    )
    print(f"\nIngestion complete: {result}")
