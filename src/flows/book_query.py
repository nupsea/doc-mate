"""
Backward compatibility layer for document querying.
Redirects calls to document_query.py.
"""

from src.flows.document_query import (
    query_document,
    search_document_content,
    get_document_chapters,
    get_document_summary
)

# Map old names to new names
query_book = query_document
search_book_content = search_document_content
get_chapter_summaries = get_document_chapters
get_book_summary = get_document_summary
