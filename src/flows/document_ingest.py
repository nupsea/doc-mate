"""
Document ingestion pipeline - supports books, scripts, conversations, tech docs, reports.

Backward compatible with book ingestion.
"""

import asyncio
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


async def generate_summaries(chunks: list, doc_type: str = 'book', ephemeral: bool = False, slug: str = None):
    """Generate chapter and document summaries."""
    from src.monitoring.tracer import disable_tracing, init_phoenix_tracing, is_phoenix_enabled

    # Disable tracing if ephemeral mode is enabled
    tracing_was_enabled = is_phoenix_enabled()
    if ephemeral and tracing_was_enabled:
        print("[EPHEMERAL] Disabling Phoenix tracing for summarization...")
        disable_tracing()

    try:
        gen = SummaryGenerator(doc_type=doc_type)
        chapter_summaries, document_summary = await gen.summarize_hierarchy(chunks)
        
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
        store.delete_document(slug)

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

def store_summaries_to_db(slug: str, chapter_summaries: list, document_summary: str):
    """Store document summaries to database."""
    store = PgresStore()
    if chapter_summaries and document_summary:
        store.store_summaries(slug, chapter_summaries, document_summary)

async def build_search_indexes(chunks: list):
    """Build BM25 and vector search indexes in parallel."""
    retriever = FusionRetriever()

    # Run BM25 and Vector indexing concurrently
    # Both methods now handle batching internally
    await asyncio.gather(
        asyncio.to_thread(retriever.bm25.build_index, chunks),
        asyncio.to_thread(retriever.vec.build_index, chunks)
    )

    return {
        "bm25_indexed": len(chunks),
        "vector_indexed": len(chunks),
        "new_chunks": len(chunks),
    }


async def build_graph_index(chunks: list, doc_id: int, doc_type: str = 'book', ephemeral: bool = False):
    """
    Extract, resolve, and store graph entities and relationships incrementally.
    """
    print(f"[GRAPH] Starting graph extraction for {len(chunks)} chunks...")
    
    # Initialize components
    # Note: Using small batch size to avoid context limits and improve precision
    extractor = EntityExtractor(model_name="gpt-4o-mini", batch_size=8)
    resolver = EntityResolver()
    store = PostgresGraphStore()

    # Clear existing graph for this doc if it exists (re-ingest)
    store.delete_graph_for_doc(doc_id)

    # Incremental Processing
    BATCH_SIZE = 40  # Process 40 chunks at a time (5 extractor calls of 8)
    total_entities_count = 0
    total_rels_count = 0
    
    for i in range(0, len(chunks), BATCH_SIZE):
        chunk_batch = chunks[i : i + BATCH_SIZE]
        print(f"[GRAPH] Processing chunks {i} to {min(i+BATCH_SIZE, len(chunks))} ({len(chunk_batch)} chunks)...")
        
        # 1. Extraction
        entities, relationships = await extractor.extract_from_chunks(chunk_batch, doc_type)
        
        if not entities and not relationships:
            print(f"[GRAPH] Batch {i} yielded no entities. Continuing...")
            continue

        # 1b. Kinship Resolution (Pre-process)
        # Link "Mia's sister" to "Mia" before main resolution
        entities, relationships = resolver.resolve_kinship_references(entities, relationships)

        # 2. Resolution (Local to this batch for now, effectively)
        # Note: Global resolution is better, but incremental is safer for large docs.
        # Exact name matches will merge in DB automatically.
        resolved_entities, name_mapping = resolver.resolve(entities)
        
        # Update relationships with resolved names
        for rel in relationships:
            if rel.source_entity in name_mapping:
                rel.source_entity = name_mapping[rel.source_entity]
            if rel.target_entity in name_mapping:
                rel.target_entity = name_mapping[rel.target_entity]

        # 3. Storage (Immediate)
        e_ids = store.store_entities(doc_id, resolved_entities)
        r_count = store.store_relationships(doc_id, relationships)
        print(f"[GRAPH] Stored batch: {len(e_ids)} entities, {r_count} relationships")
        
        total_entities_count += len(e_ids)
        total_rels_count += r_count

    # 4. Episode Extraction (if conversation) - done at end for context
    episodes_count = 0
    if doc_type == "conversation":
        print("[GRAPH] Extracting conversation episodes...")
        # Process episodes in larger batches or all at once?
        # Episodes usually need more sequential context. 
        # Using the same batch strategy for safety.
        for i in range(0, len(chunks), BATCH_SIZE):
            chunk_batch = chunks[i : i + BATCH_SIZE]
            episodes = await extractor.extract_episodes(chunk_batch)
            cnt = store.store_episodes(doc_id, episodes)
            episodes_count += cnt
        print(f"[GRAPH] Stored {episodes_count} total episodes")

    return {
        "entities": total_entities_count,
        "relationships": total_rels_count,
        "episodes": episodes_count,
        "status": "persisted",
        "summary_text": ""
    }


def verify_ingestion(slug: str, expected_chapters: int):
    """Verify document was ingested correctly."""
    store = PgresStore()

    if not store.document_exists(slug):
        raise ValueError(f"Document verification failed: {slug} not found in database")

    if not store.summaries_exist(slug):
        raise ValueError(
            f"Summaries verification failed: no summaries found for {slug}"
        )

    chapters = store.get_all_chapter_summaries(slug)
    actual_chapters = len(chapters)

    if actual_chapters != expected_chapters:
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
):
    """
    Ingest any document type: validate -> parse -> summarize -> store -> build indexes -> verify.

    Args:
        doc_type: 'book', 'script', 'conversation', 'tech_doc', 'report'
        ephemeral: If True, disable Phoenix tracing during summarization
    """
    print(f"Starting ingestion for: {title} (type: {doc_type}, slug: {slug})")

    validation = validate_inputs(slug, file_path, title, force_update)
    print(f"Validation passed - File size: {validation['file_size']} bytes")

    parse_result = read_and_parse(slug, file_path, doc_type, split_pattern, max_tokens, overlap)
    print(
        f"Parsed {parse_result['num_chunks']} chunks, {parse_result['num_chars']} chars"
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
    print("[INGEST] Starting parallel execution: Summaries | Search Indexes | Knowledge Graph...")
    
    summary_result, search_result, graph_stats = await asyncio.gather(
        generate_summaries(parse_result["chunks"], doc_type=doc_type, ephemeral=ephemeral, slug=slug),
        build_search_indexes(parse_result["chunks"]),
        build_graph_index(parse_result["chunks"], db_result["doc_id"], doc_type, ephemeral=ephemeral)
    )
    
    print(
        f"Generated {summary_result['num_chapters']} section summaries + overall summary"
    )
    print(
        f"Built search indexes - BM25: {search_result['bm25_indexed']}, Vector: {search_result['vector_indexed']} chunks"
    )
    print(f"Built Knowledge Graph - Entities: {graph_stats['entities']}, Relations: {graph_stats['relationships']}")

    verify_result = verify_ingestion(slug, summary_result["num_chapters"])
    print(f"Verification complete - Status: {verify_result['status']}")

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
