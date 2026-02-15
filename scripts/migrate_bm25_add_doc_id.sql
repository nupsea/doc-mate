-- Migration: Add doc_id FK with ON DELETE CASCADE to BM25 tables.
-- This ensures BM25 data is automatically cleaned up when a document is deleted.
--
-- Usage: docker exec books_postgres psql -U bookuser -d booksdb -f /scripts/migrate_bm25_add_doc_id.sql
-- Or paste into psql manually.

BEGIN;

-- 1. Add doc_id column (nullable initially so we can backfill)
ALTER TABLE bm25_index ADD COLUMN IF NOT EXISTS doc_id INT;
ALTER TABLE bm25_doc_lens ADD COLUMN IF NOT EXISTS doc_id INT;

-- 2. Backfill doc_id from documents table using slug prefix in chunk_id
UPDATE bm25_index bi
SET doc_id = d.doc_id
FROM documents d
WHERE bi.chunk_id LIKE d.slug || '_%'
  AND bi.doc_id IS NULL;

UPDATE bm25_doc_lens dl
SET doc_id = d.doc_id
FROM documents d
WHERE dl.chunk_id LIKE d.slug || '_%'
  AND dl.doc_id IS NULL;

-- 3. Delete any rows that couldn't be matched (true orphans)
DELETE FROM bm25_index WHERE doc_id IS NULL;
DELETE FROM bm25_doc_lens WHERE doc_id IS NULL;

-- 4. Set NOT NULL constraint
ALTER TABLE bm25_index ALTER COLUMN doc_id SET NOT NULL;
ALTER TABLE bm25_doc_lens ALTER COLUMN doc_id SET NOT NULL;

-- 5. Add foreign key with CASCADE
ALTER TABLE bm25_index
  ADD CONSTRAINT fk_bm25_index_doc_id
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE;

ALTER TABLE bm25_doc_lens
  ADD CONSTRAINT fk_bm25_doc_lens_doc_id
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE;

-- 6. Add index on doc_id for faster cascade deletes and filtered queries
CREATE INDEX IF NOT EXISTS idx_bm25_index_doc_id ON bm25_index(doc_id);
CREATE INDEX IF NOT EXISTS idx_bm25_doc_lens_doc_id ON bm25_doc_lens(doc_id);

COMMIT;
