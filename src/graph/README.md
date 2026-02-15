# Graph Knowledge Layer

The Graph Knowledge Layer enhances Doc-Mate with structured understanding of entities (people, places, concepts) and their relationships. This enables the agent to answer questions like "How is X related to Y?" which are difficult for traditional vector search.

## Architecture

The graph is stored natively in PostgreSQL and integrated into the ingestion and retrieval pipelines.

### 1. Schema (Postgres)

- **`graph_entities`**: Nodes in the graph.
  - `entity_id`: Primary key.
  - `name`: Canonical name (e.g., "Achilles").
  - `entity_type`: e.g., "Person", "Location", "Concept".
  - `description`: LLM-generated summary of the entity's role.
  - `source_chunk_ids`: Array of text chunks where this entity appears.

- **`graph_relationships`**: Edges between nodes.
  - `source_entity_id` -> `target_entity_id`.
  - `relation_type`: e.g., "friend_of", "located_in", "killed_by".
  - `description`: Context for the relationship.

- **`graph_episodes`** (Conversations only):
  - Tracks temporal "episodes" or turns in a conversation with speaker/stance metadata.

### 2. Extraction Pipeline (`src/graph/extractor.py`)

Runs during document ingestion (`document_ingest.py`).

1. **Chunking**: Document is split into text chunks.
2. **LLM Extraction**: `EntityExtractor` sends batches of chunks to `gpt-4o-mini`.
   - Uses strict Pydantic models for structured output.
   - Robust rate-limiting (Semaphore=1, Retries, Jitter) prevents API errors.
3. **Resolution**: `EntityResolver` uses `SentenceTransformer` embeddings to deduplicate entities (e.g., merging "Achilles" and "Pelides").
4. **Storage**: `PostgresGraphStore` persists unique entities and relationships incrementally.

### 3. Retrieval (`src/graph/retriever.py`)

Used by `AdaptiveRetriever` (for search) and `explore_entity_graph` (agent tool).

- **Entity Matching**: Maps query keywords to graph entities.
- **Recursive Traversal**: Uses CTEs (Common Table Expressions) to find:
  - 1-hop neighbors (direct relationships).
  - 2-hop neighbors (indirect connections).
- **Scoring**:
  - Direct hits = 1.0
  - 1-hop = 0.5
  - 2-hop = 0.25

## Usage

### Agent Tool
The agent uses `explore_entity_graph(entity_name, doc_slug)` to explicitly query the graph.

**Example**:
> User: "How is Hector related to Priam?"
> Tool: explore_entity_graph("Hector", "ili")
> Result: "Hector --[son_of]--> Priam"

### Search Fusion
Graph scores are combined with BM25 and Vector scores in `AdaptiveRetriever`:
- **BM25**: 35%
- **Vector**: 35%
- **Graph**: 30%

This ensures that chunks discussing the *relationship* between entities rank higher than chunks that just mention their names separately.
