-- ============================================================================
-- Stage 1: Add Multi-Format Support to Books Table (FIXED)
-- ============================================================================

BEGIN;

-- Add doc_type column (defaults to 'book' for existing records)
ALTER TABLE books
  ADD COLUMN IF NOT EXISTS doc_type VARCHAR(20) DEFAULT 'book';

-- Add metadata column (JSONB for type-specific data)
ALTER TABLE books
  ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

-- Add constraint (with error handling for already existing constraint)
DO $$
BEGIN
  ALTER TABLE books ADD CONSTRAINT valid_doc_type
    CHECK (doc_type IN ('book', 'script', 'conversation', 'tech_doc', 'report'));
EXCEPTION
  WHEN duplicate_object THEN
    RAISE NOTICE 'Constraint valid_doc_type already exists, skipping';
END $$;

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_books_doc_type ON books(doc_type);
CREATE INDEX IF NOT EXISTS idx_books_metadata ON books USING GIN (metadata);

-- Verify changes
DO $$
DECLARE
    doc_type_exists BOOLEAN;
    metadata_exists BOOLEAN;
BEGIN
    -- Check if columns were added
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'books' AND column_name = 'doc_type'
    ) INTO doc_type_exists;

    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'books' AND column_name = 'metadata'
    ) INTO metadata_exists;

    -- Report results
    IF doc_type_exists AND metadata_exists THEN
        RAISE NOTICE '✓ Stage 1 migration successful!';
        RAISE NOTICE '  - doc_type column added';
        RAISE NOTICE '  - metadata column added';
        RAISE NOTICE '  - Indexes created';
        RAISE NOTICE '';
        RAISE NOTICE 'Books table now supports: book, script, conversation, tech_doc, report';
    ELSE
        RAISE EXCEPTION 'Migration failed - columns not created';
    END IF;
END $$;

COMMIT;

-- Show current schema
\d books;
