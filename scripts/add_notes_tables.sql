-- Migration: Add Notes support to Doc-Mate
-- Notes are documents (doc_type='note') with additional metadata
-- Run this on existing databases to enable the notes feature

-- 1. Extend doc_type constraint to include 'note'
ALTER TABLE documents DROP CONSTRAINT IF EXISTS valid_doc_type;
ALTER TABLE documents ADD CONSTRAINT valid_doc_type
  CHECK (doc_type IN ('book', 'script', 'conversation', 'tech_doc', 'report', 'note'));

-- 2. Notes metadata table (extends the documents row)
CREATE TABLE IF NOT EXISTS notes (
    note_id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL UNIQUE REFERENCES documents(doc_id) ON DELETE CASCADE,
    content TEXT NOT NULL,                    -- raw markdown content (source of truth)
    tags TEXT[] DEFAULT '{}',                 -- user-assigned tags
    source_refs JSONB DEFAULT '[]'::jsonb,    -- [{slug, chunk_id, snippet, query}]
    is_pinned BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT NOW(),
    version INT DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_notes_doc_id ON notes(doc_id);
CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_pinned ON notes(is_pinned) WHERE is_pinned = TRUE;
