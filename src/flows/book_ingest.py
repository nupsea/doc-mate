"""
Backward compatibility layer for document ingestion.
Redirects calls to document_ingest.py.
"""

from src.flows.document_ingest import ingest_document, ingest_book

# Ensure both names are available for older code
ingest_book = ingest_book
ingest_document = ingest_document
