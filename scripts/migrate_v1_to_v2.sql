-- Rename tables and columns to match new 'documents' schema
BEGIN;

-- Rename books -> documents
ALTER TABLE IF EXISTS books RENAME TO documents;
ALTER TABLE IF EXISTS documents RENAME COLUMN book_id TO doc_id;

-- Rename summaries tables and columns
ALTER TABLE IF EXISTS book_summaries RENAME TO document_summaries;
ALTER TABLE IF EXISTS document_summaries RENAME COLUMN book_id TO doc_id;

ALTER TABLE IF EXISTS chapter_summaries RENAME COLUMN book_id TO doc_id;

-- Add BM25 tables if they don't exist
CREATE TABLE IF NOT EXISTS bm25_index (
    term TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    frequency INT NOT NULL,
    PRIMARY KEY (term, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_bm25_term ON bm25_index(term);
CREATE INDEX IF NOT EXISTS idx_bm25_chunk_id ON bm25_index(chunk_id);

CREATE TABLE IF NOT EXISTS bm25_doc_lens (
    chunk_id TEXT PRIMARY KEY,
    doc_len INT NOT NULL
);

-- Update query_metrics
ALTER TABLE IF EXISTS query_metrics RENAME COLUMN book_slug TO doc_slug;
CREATE INDEX IF NOT EXISTS idx_query_metrics_doc_slug ON query_metrics(doc_slug);

COMMIT;
