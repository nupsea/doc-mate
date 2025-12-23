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


def validate_inputs(slug: str, file_path: str, title: str, force_update: bool = False):
    """Validate inputs before processing."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not slug or not slug.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid slug format: {slug}")

    store = PgresStore()
    exists = store.book_exists(slug)

    if exists and not force_update:
        raise ValueError(
            f"Book with slug '{slug}' already exists. Use force_update=True to overwrite."
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
        print(f"[PARSE] Parsing document structure...")
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


async def generate_summaries(chunks: list, doc_type: str = 'book'):
    """Generate chapter and book summaries."""
    gen = SummaryGenerator(doc_type=doc_type)
    chapter_summaries, book_summary = await gen.summarize_hierarchy(chunks)

    return {
        "chapter_summaries": chapter_summaries,
        "book_summary": book_summary,
        "num_chapters": len(chapter_summaries),
    }


def store_to_db(
    slug: str,
    title: str,
    author: str,
    num_chunks: int,
    num_chars: int,
    chapter_summaries: list,
    book_summary: str,
    doc_type: str = 'book',
    metadata: dict = None,
    force_update: bool = False,
):
    """Store document metadata and summaries to database."""
    store = PgresStore()

    if force_update and store.book_exists(slug):
        store.delete_book(slug)

    # Use new store_document method for multi-format support
    book_id = store.store_document(
        slug=slug,
        title=title,
        doc_type=doc_type,
        author=author,
        num_chunks=num_chunks,
        num_chars=num_chars,
        metadata=metadata
    )

    # Store summaries (works for all doc types)
    if chapter_summaries and book_summary:
        store.store_summaries(slug, chapter_summaries, book_summary)

    return {"book_id": book_id, "slug": slug}


def build_search_indexes(chunks: list):
    """Build BM25 and vector search indexes (append to existing indexes)."""
    retriever = FusionRetriever()

    # Load existing BM25 index (if exists)
    try:
        retriever.load_bm25_index()
        print(f"Loaded existing BM25 index with {retriever.bm25.N} documents")

        # Append new chunks to existing index
        existing_chunks = [
            {"id": retriever.bm25.ids[i], "text": retriever.bm25.raw_docs[i]}
            for i in range(retriever.bm25.N)
        ]
        all_chunks = existing_chunks + chunks
        print(f"Building combined index with {len(all_chunks)} total chunks")

    except FileNotFoundError:
        print("No existing BM25 index found, creating new one")
        all_chunks = chunks

    # Rebuild BM25 with all chunks (existing + new)
    retriever.bm25.build_index(all_chunks)
    retriever.bm25.save_index(retriever.bm25_index_path)

    # Build vector index for new chunks only (Qdrant handles appending)
    retriever.vec.build_index(chunks)

    return {
        "bm25_indexed": len(all_chunks),
        "vector_indexed": len(chunks),
        "new_chunks": len(chunks),
    }


def verify_ingestion(slug: str, expected_chapters: int):
    """Verify book was ingested correctly."""
    store = PgresStore()

    if not store.book_exists(slug):
        raise ValueError(f"Book verification failed: {slug} not found in database")

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

    book_summary = store.get_book_summary(slug)

    return {
        "status": "success",
        "slug": slug,
        "chapters_verified": actual_chapters,
        "book_summary_length": len(book_summary) if book_summary else 0,
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
):
    """
    Ingest any document type: validate -> parse -> summarize -> store -> build indexes -> verify.

    Args:
        doc_type: 'book', 'script', 'conversation', 'tech_doc', 'report'
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
        except:
            metadata = {}

    # Generate summaries (for all types)
    summary_result = await generate_summaries(parse_result["chunks"], doc_type=doc_type)
    print(
        f"Generated {summary_result['num_chapters']} section summaries + overall summary"
    )

    db_result = store_to_db(
        slug,
        title,
        author,
        parse_result["num_chunks"],
        parse_result["num_chars"],
        summary_result["chapter_summaries"],
        summary_result["book_summary"],
        doc_type,
        metadata,
        force_update,
    )
    print(f"Stored to database - Document ID: {db_result['book_id']}")

    search_result = build_search_indexes(parse_result["chunks"])
    print(
        f"Built search indexes - BM25: {search_result['bm25_indexed']}, Vector: {search_result['vector_indexed']} chunks"
    )

    verify_result = verify_ingestion(slug, summary_result["num_chapters"])
    print(f"Verification complete - Status: {verify_result['status']}")

    return {
        "slug": slug,
        "book_id": db_result["book_id"],
        "title": title,
        "doc_type": doc_type,
        "chapters": verify_result["chapters_verified"],
        "chunks": parse_result["num_chunks"],
        "search_indexed": search_result["bm25_indexed"],
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
