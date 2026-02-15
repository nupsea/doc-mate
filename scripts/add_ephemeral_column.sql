-- Add is_ephemeral column to documents table for privacy control
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS is_ephemeral BOOLEAN DEFAULT FALSE;

-- Create an index for faster filtering
CREATE INDEX IF NOT EXISTS idx_documents_ephemeral ON documents(is_ephemeral);
