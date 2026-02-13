-- Create tables in the booksdb database
-- This runs only on first Postgres container startup
-- The database 'booksdb' is already created by POSTGRES_DB environment variable

CREATE TABLE IF NOT EXISTS documents (
    doc_id SERIAL PRIMARY KEY,
    slug VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    num_chunks INT,
    num_chars INT,
    doc_type VARCHAR(20) DEFAULT 'book',
    metadata JSONB DEFAULT '{}'::jsonb,
    added_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT valid_doc_type CHECK (doc_type IN ('book', 'script', 'conversation', 'tech_doc', 'report'))
);

-- Indexes for multi-format support
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);

CREATE TABLE IF NOT EXISTS chapter_summaries (
    doc_id INT NOT NULL,
    chapter_id INT NOT NULL,
    summary TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (doc_id, chapter_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_summaries (
    doc_id INT PRIMARY KEY,
    summary TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bm25_index (
    term TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    doc_id INT NOT NULL,
    frequency INT NOT NULL,
    PRIMARY KEY (term, chunk_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bm25_term ON bm25_index(term);
CREATE INDEX IF NOT EXISTS idx_bm25_chunk_id ON bm25_index(chunk_id);
CREATE INDEX IF NOT EXISTS idx_bm25_index_doc_id ON bm25_index(doc_id);

CREATE TABLE IF NOT EXISTS bm25_doc_lens (
    chunk_id TEXT PRIMARY KEY,
    doc_id INT NOT NULL,
    doc_len INT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bm25_doc_lens_doc_id ON bm25_doc_lens(doc_id);

-- Metrics tables for monitoring
CREATE TABLE IF NOT EXISTS query_metrics (
    query_id VARCHAR(100) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    query TEXT NOT NULL,
    response TEXT,
    doc_slug VARCHAR(50),
    latency_ms FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    tool_calls TEXT[],
    num_results INTEGER,
    llm_relevance_score VARCHAR(20),
    llm_reasoning TEXT,
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    user_comment TEXT,
    retry_attempted BOOLEAN DEFAULT FALSE,
    original_query TEXT,
    rephrased_query TEXT,
    retry_results INTEGER,
    fallback_to_context BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp ON query_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_query_metrics_doc_slug ON query_metrics(doc_slug);
CREATE INDEX IF NOT EXISTS idx_query_metrics_success ON query_metrics(success);