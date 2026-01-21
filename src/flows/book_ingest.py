"""
Backward compatibility layer for document ingestion.
Redirects calls to document_ingest.py.
"""

from src.flows.document_ingest import ingest_document, ingest_book

# Explicitly export for star imports
__all__ = ["ingest_document", "ingest_book"]
