-- Create tables for Graph Knowledge Layer
-- Run this to add graph capabilities to the existing schema

-- 1. Entities Table
CREATE TABLE IF NOT EXISTS graph_entities (
    entity_id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL,
    name TEXT NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    source_chunk_ids TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
    CONSTRAINT unique_entity_per_doc UNIQUE (doc_id, name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_ge_doc ON graph_entities(doc_id);
CREATE INDEX IF NOT EXISTS idx_ge_name ON graph_entities(name);
CREATE INDEX IF NOT EXISTS idx_ge_type ON graph_entities(entity_type);

-- 2. Relationships Table
CREATE TABLE IF NOT EXISTS graph_relationships (
    rel_id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL,
    source_entity_id INT NOT NULL,
    target_entity_id INT NOT NULL,
    relation_type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    source_chunk_ids TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (source_entity_id) REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES graph_entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_gr_doc ON graph_relationships(doc_id);
CREATE INDEX IF NOT EXISTS idx_gr_source ON graph_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_gr_target ON graph_relationships(target_entity_id);

-- 3. Episodes Table (for conversations/transcripts)
CREATE TABLE IF NOT EXISTS graph_episodes (
    episode_id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL,
    speaker TEXT,
    stance TEXT,
    topic TEXT,
    summary TEXT,
    turn_start INT,
    turn_end INT,
    timestamp_start TEXT,
    timestamp_end TEXT,
    entity_ids INT[],
    source_chunk_ids TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_gep_doc ON graph_episodes(doc_id);
CREATE INDEX IF NOT EXISTS idx_gep_speaker ON graph_episodes(speaker);
CREATE INDEX IF NOT EXISTS idx_gep_topic ON graph_episodes(topic);
CREATE INDEX IF NOT EXISTS idx_gep_stance ON graph_episodes(stance);
CREATE INDEX IF NOT EXISTS idx_gep_chunk_ids ON graph_episodes USING GIN (source_chunk_ids);
