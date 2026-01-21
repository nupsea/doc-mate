"""
Utility functions for the UI.
"""

import re
from src.content.store import PgresStore


def get_available_documents():
    """Fetch list of documents from database with slug, title, author, chunks, and added_at."""
    try:
        store = PgresStore()
        with store.conn.cursor() as cur:
            cur.execute(
                """
                SELECT slug, title, author, num_chunks, added_at
                FROM documents
                ORDER BY added_at DESC
            """
            )
            docs = cur.fetchall()
        return docs
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []

# Compatibility alias
get_available_books = get_available_documents


def validate_slug(slug: str) -> tuple[bool, str]:
    """
    Validate slug format and check for duplicates.

    Returns:
        (is_valid, error_message)
    """
    if not slug or not slug.strip():
        return False, "Slug cannot be empty"

    slug = slug.strip().lower()

    # Check format (lowercase letters, numbers, hyphens, underscores only)
    if not re.match(r"^[a-z0-9_-]+$", slug):
        return (
            False,
            "Slug must contain only lowercase letters, numbers, hyphens, and underscores",
        )

    # Check length
    if len(slug) < 2 or len(slug) > 20:
        return False, "Slug must be between 2 and 20 characters"

    # Check if slug already exists
    try:
        store = PgresStore()
        with store.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents WHERE slug = %s", (slug,))
            count = cur.fetchone()[0]
            if count > 0:
                return False, f"Slug '{slug}' already exists. Choose a different one."
    except Exception as e:
        return False, f"Error checking slug: {str(e)}"

    return True, ""


def detect_chapter_pattern(file_path: str) -> tuple[str, str]:
    """
    Try to auto-detect chapter pattern from file.

    Returns:
        (pattern, description)
    """
    patterns = [
        (
            r"^(?:CHAPTER [IVXLCDM]+)\s*\n",
            "CHAPTER + Roman numerals (e.g., CHAPTER I, CHAPTER II)",
        ),
        (r"^(?:BOOK [IVXLCDM]+)\s*\n", "BOOK + Roman numerals (e.g., BOOK I, BOOK II)"),
        (
            r"^(?:[IVXLCDM]+\. [A-Z])",
            "Roman numeral + period + title (e.g., I. TITLE, II. TITLE)",
        ),
        (
            r"^(?:Chapter \d+)\s*\n",
            "Chapter + Arabic numerals (e.g., Chapter 1, Chapter 2)",
        ),
        (r"^(?:PART [IVXLCDM]+)\s*\n", "PART + Roman numerals (e.g., PART I, PART II)"),
        (
            r"^(?:\d+\.\s+[A-Z])",
            "Arabic numeral + period + title (e.g., 1. Title, 2. Title)",
        ),
        (r"^(?:\d+\.)\s*$", "Numbered sections only (e.g., 1., 2., 3.)"),
    ]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(100000)  # Read first 100KB

        for pattern, description in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            if len(matches) >= 2:  # Found at least 2 chapters
                return (
                    pattern,
                    f"Detected: {description} (found {len(matches)} matches)",
                )

        return "", "No pattern detected. Please provide custom pattern."
    except Exception as e:
        return "", f"Error reading file: {str(e)}"


def extract_chapter_info_from_chunks(slug: str):
    """
    Analyze chunk IDs to determine section/chapter count.

    Chunk format: slug_section_chunk_hash (e.g., mma_01_001_abc123)

    Returns:
        dict with chapter/section info
    """
    try:
        # In the new DB-backed world, we don't need load_bm25_index
        # but we need to check if there are any chunks for this slug.
        
        # We'll use the store to get chunk IDs for this document
        store = PgresStore()
        with store.conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM bm25_doc_lens WHERE chunk_id LIKE %s", (f"{slug}_%",))
            doc_chunks = [row[0] for row in cur.fetchall()]

        if not doc_chunks:
            return {"status": "error", "message": f"No chunks found for document '{slug}'"}

        # Extract section numbers
        section_numbers = set()
        for chunk_id in doc_chunks:
            # chunk_id format: mma_01_001_abc123
            parts = chunk_id.split("_")
            if len(parts) >= 2:
                section_num = parts[1]
                section_numbers.add(section_num)

        sorted_sections = sorted(section_numbers)

        return {
            "status": "success",
            "total_chunks": len(doc_chunks),
            "total_sections": len(sorted_sections),
            "first_chunk": doc_chunks[0] if doc_chunks else "N/A",
            "last_chunk": doc_chunks[-1] if doc_chunks else "N/A",
            "section_range": (
                f"{sorted_sections[0]} to {sorted_sections[-1]}"
                if sorted_sections
                else "N/A"
            ),
            "sections": sorted_sections,
        }

    except Exception as e:
        return {"status": "error", "message": f"Error analyzing chunks: {str(e)}"}


def format_document_list(docs):
    """Format document list as a dataframe (list of lists for Gradio Dataframe)."""
    if not docs:
        return []

    data = []
    for slug, title, author, num_chunks, added_at in docs:
        # Format date
        if added_at:
            date_str = added_at.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = "Unknown"

        data.append(
            [
                slug,
                title,
                author or "Unknown",
                num_chunks if num_chunks else 0,
                date_str,
            ]
        )

    return data

# Compatibility alias
format_book_list = format_document_list


def delete_document(slug: str) -> tuple[bool, str, int]:
    """
    Delete a document and all its associated data from:
    - PostgreSQL (documents table - cascades to summaries and index)
    - Qdrant vector store

    Returns:
        (success, message, chunks_deleted)
    """
    try:
        from src.search.hybrid import FusionRetriever
        from qdrant_client import models

        store = PgresStore()
        retriever = FusionRetriever()

        # Check if document exists and get info
        with store.conn.cursor() as cur:
            cur.execute("SELECT title, num_chunks FROM documents WHERE slug = %s", (slug,))
            result = cur.fetchone()
            if not result:
                return False, f"Document '{slug}' not found", 0

            doc_title = result[0]
            deleted_chunks = result[1] or 0

        # Delete from PostgreSQL (CASCADE handles summaries, BM25 index needs manual cleanup if no FK)
        # Note: bm25_index uses chunk_id which starts with slug_
        with store.conn.cursor() as cur:
            # First clean up BM25 tables since they use chunk_id strings, not doc_id FK
            cur.execute("DELETE FROM bm25_index WHERE chunk_id LIKE %s", (f"{slug}_%",))
            cur.execute("DELETE FROM bm25_doc_lens WHERE chunk_id LIKE %s", (f"{slug}_%",))
            
            # Now delete from main documents table
            cur.execute("DELETE FROM documents WHERE slug = %s", (slug,))
            store.conn.commit()

        # Delete from Qdrant
        qdrant_success = True
        qdrant_error = ""
        try:
            # FusionRetriever doesn't have qdrant_client directly, it's in vec.qdrant
            retriever.vec.qdrant.delete(
                collection_name=retriever.vec.COLLECTION,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchText(text=slug)
                        )
                    ]
                ),
            )
        except Exception as e:
            qdrant_success = False
            qdrant_error = str(e)

        # Build success message
        if qdrant_success:
            return (
                True,
                f"[SUCCESS] Deleted '{doc_title}' ({deleted_chunks} chunks)",
                deleted_chunks,
            )
        else:
            return (
                True,
                f"[WARNING] Document '{doc_title}' deleted from DB, but Qdrant cleanup failed: {qdrant_error}",
                deleted_chunks,
            )

    except Exception as e:
        return False, f"[ERROR] Failed to delete document: {str(e)}", 0

# Compatibility alias
delete_book = delete_document
