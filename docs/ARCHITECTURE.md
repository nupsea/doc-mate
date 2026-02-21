# Architecture Reference

Technical architecture and implementation reference for Doc-Mate.

**Last Updated**: February 2026

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI                              │
│  ┌──────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐   │
│  │ Chat │  │ Notes    │  │ Ingest    │  │ Monitoring   │   │
│  └──┬───┘  └──┬───────┘  └──┬────────┘  └──────────────┘   │
└─────┼─────────┼─────────────┼───────────────────────────────┘
      │         │             │
┌─────▼─────────▼─────────────▼───────────────────────────────┐
│              DocMateAgent (MCP Client)                       │
│           LangGraph Router-Retriever-Generator               │
└─────┬───────────────────────────────────┬───────────────────┘
      │                                   │
      │  ┌────────────────────────────┐   │
      │  │      LLM Providers        │   │
      │  │  ┌────────┐  ┌─────────┐  │   │
      │  │  │ OpenAI │  │ Ollama  │  │   │
      │  │  └────────┘  └─────────┘  │   │
      │  └────────────────────────────┘   │
      │                                   │
┌─────▼───────────────────────────────────▼───────────────────┐
│                     Data Layer                               │
│  ┌────────────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │  PostgreSQL    │  │  Qdrant  │  │  BM25 (in-DB)       │ │
│  │  - Documents   │  │  Vector  │  │  - bm25_index       │ │
│  │  - Summaries   │  │  Search  │  │  - bm25_doc_lens    │ │
│  │  - Graph       │  │          │  │                      │ │
│  │  - Notes       │  │          │  │                      │ │
│  │  - Metrics     │  │          │  │                      │ │
│  │  - Ingest Jobs │  │          │  │                      │ │
│  └────────────────┘  └──────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. UI Layer (`src/ui/`)
- **app.py**: Main DocMateUI controller, manages agent lifecycle (per-request instantiation)
- **chat.py**: Chat interface with provider/model/privacy selection, feedback, save-as-note
- **ingest.py**: Document upload and ingestion interface with pattern detection
- **notes.py**: Notes editor with list/editor split view, tag filtering, source tracking
- **monitoring.py**: Metrics dashboard with query performance visualization
- **pattern_builder.py**: Document structure pattern detection for ingestion
- **utils.py**: Document listing, deletion (cascade to Qdrant), slug validation

### 2. Agent / Orchestration (`src/mcp_client/`, `src/flows/`)

**DocMateAgent** (`src/mcp_client/agent.py`):
- Orchestrates queries via LangGraph agent_graph
- Per-request instantiation for thread safety
- Privacy mode enforcement (ephemeral/internal flags)
- Response judging via LLM self-assessment
- Compatibility alias: `BookMateAgent = DocMateAgent`

**LangGraph RRG Pipeline** (`src/flows/agent_graph.py`):
```
Router -> [Summary Retriever, Content Retriever, Graph Retriever] -> Aggregator -> Generator
```
- **Router**: Classifies intent (explore/compare/chat), extracts target docs and entities
- **Summary Retriever**: Fetches document summaries for summary/hybrid strategies
- **Content Retriever**: Type-aware hybrid search with context expansion for conversations
- **Graph Retriever**: Knowledge graph relationship traversal (skips tech_doc/report)
- **Aggregator**: Merges results with deduplication, type-formatting, confidence assessment, episode context
- **Generator**: Confidence-adaptive prompting with query-type guidance

**Supporting Flows**:
- **router.py**: Query classification, entity extraction, conversation-aware history
- **document_query.py**: Hybrid search execution, adjacent chunk retrieval
- **document_ingest.py**: Multi-stage document ingestion pipeline
- **note_ingest.py**: Lightweight note indexing (chunk, BM25, vector, optional graph)
- **confidence.py**: Multi-signal confidence assessment (density, retrieval score, entity coverage, lexical overlap)
- **agent_tools.py**: Tool definitions for agent
- **ingest_profiles.py**: Ingestion configuration profiles

### 3. LLM Providers (`src/llm/`)
- **config.py**: LLMConfig with environment-based configuration
- **generator.py**: Response generation with provider abstraction
- **providers/**:
  - `base.py`: Abstract provider interface
  - `openai_provider.py`: OpenAI API client
  - `local_provider.py`: Ollama local LLM client (timeout: 60s, max_tokens: 2048)
  - `model_router.py`: Provider selection with fallback control
  - `query_classifier.py`: Query type classification

### 4. Search System (`src/search/`)
- **bm25.py**: BM25 keyword search using PostgreSQL-stored term frequencies
- **vec.py**: Qdrant vector search with sentence-transformers embeddings
- **hybrid.py**: Base hybrid search using Reciprocal Rank Fusion (RRF)
- **adaptive.py**: AdaptiveRetriever -- triple hybrid fusion (BM25 + Vector + Graph)
  - With graph: BM25=0.35, Vec=0.35, Graph=0.30
  - Without graph: BM25=0.70, Vec=0.30
  - Query preprocessing (stop word removal, +8% Hit@5)

### 5. Content Processing (`src/content/`)
- **store.py**: PgresStore -- PostgreSQL storage with connection pooling
  - Document CRUD, summaries, BM25 index storage
  - Ingest job tracking (create/complete/fail)
  - Note operations (create, update, delete, list, filter by tag/source/search)
- **db.py**: DatabaseManager with connection pooling
- **reader.py**: File reading utilities
- **parsers/**:
  - `base.py`: Abstract parser interface (BaseParser/DocumentParser)
  - `book_parser.py`: Chapter-based parsing (Gutenberg-style)
  - `script_parser.py`: Scene/dialogue parsing (INT/EXT detection)
  - `conversation_parser.py`: Speaker turn parsing with adaptive chunking
  - `tech_doc_parser.py`: Code block extraction, API reference parsing
  - `report_parser.py`: Section-based parsing
  - `markdown_parser.py`: Markdown/PDF parsing with section awareness, code block reconstruction

### 6. Knowledge Graph (`src/graph/`)
- **store.py**: PostgresGraphStore -- entity, relationship, and episode storage
  - Entity deduplication with chunk_id merging
  - Relationship weight accumulation on conflict
  - Recursive CTE for multi-hop traversal (1-2 hops)
  - Episode storage for conversation documents
- **extractor.py**: EntityExtractor -- LLM-based extraction
  - Doc-type aware schemas (book, script, conversation, tech_doc, report, note); casual conversations auto-detected and given social extraction schema internally
  - Batch processing with semaphore (2 concurrent, batch_size=8)
  - Retry with exponential backoff for rate limits
  - Episode extraction for conversations
- **retriever.py**: GraphRetriever + ConversationGraphRetriever
  - Entity matching: exact, substring, semantic (SentenceTransformer, threshold 0.65)
  - Multi-hop scoring: hop0=1.0, hop1=0.5, hop2=0.25
  - Episode matching for conversations (scored 2.0)
- **resolver.py**: Entity resolution and linking
- **schemas.py**: Pydantic models -- Entity, Relationship, Episode, DOC_TYPE_SCHEMAS

### 7. Monitoring (`src/monitoring/`)
- **metrics.py**: QueryTimer / NoOpQueryTimer, LLM relevance scoring
- **judge.py**: LLM self-assessment of response quality (EXCELLENT/ADEQUATE/POOR)
- **tracer.py**: Phoenix OpenTelemetry integration with disable_tracing support
- **dashboard.py**: Metrics dashboard visualization
- **persistence.py**: Metrics storage

### 8. Prompts (`src/mcp_client/prompts/`)
- **config.yaml**: Prompt templates by document type, confidence levels, personas, evidence notes
- **builder.py**: Conditional prompt assembly with doc-type awareness

## Database Schema

### PostgreSQL

```sql
-- Core document storage
documents (doc_id PK, slug UNIQUE, title, author, num_chunks, num_chars,
           doc_type CHECK('book','script','conversation','tech_doc','report','note'),
           metadata JSONB, added_at)

-- Summaries
chapter_summaries (doc_id FK CASCADE, chapter_id, summary, PK(doc_id, chapter_id))
document_summaries (doc_id PK FK CASCADE, summary)

-- BM25 search indexes (in-database)
bm25_index (term, chunk_id, doc_id FK CASCADE, frequency, PK(term, chunk_id))
bm25_doc_lens (chunk_id PK, doc_id FK CASCADE, doc_len)

-- Knowledge graph
graph_entities (entity_id PK, doc_id FK CASCADE, name, entity_type, description,
                source_chunk_ids TEXT[], metadata JSONB)
graph_relationships (rel_id PK, doc_id FK CASCADE, source_entity_id FK, target_entity_id FK,
                     relation_type, weight, description, source_chunk_ids TEXT[])
graph_episodes (episode_id PK, doc_id FK CASCADE, speaker, stance, topic, summary,
                turn_start, turn_end, timestamp_start, timestamp_end,
                entity_names TEXT[], source_chunk_ids TEXT[])

-- Notes
notes (note_id PK, doc_id FK CASCADE UNIQUE, content, tags TEXT[],
       source_refs JSONB, is_pinned, updated_at, version)

-- Metrics and jobs
query_metrics (query_id PK, timestamp, query, response, doc_slug, latency_ms,
               success, tool_calls TEXT[], llm_relevance_score, user_rating 1-5, ...)
ingest_jobs (job_id PK, slug, title, doc_type, status, error_message,
             result_summary, created_at, completed_at)
```

All child tables use `ON DELETE CASCADE` from documents.

### Qdrant

**Collection**: `doc-mate`
- Vectors: 384-dim (sentence-transformers BAAI/bge-small-en) or 1536-dim (OpenAI)
- Distance: Cosine similarity
- Payload: `{doc_slug, chunk_id, text, section_id, doc_type}`

## Key Flows

### 1. Query Flow (LangGraph RRG)

```
User query
  |
  v
Router Node
  |- Classify intent (explore/compare/chat)
  |- Extract target docs + entities
  |- Entity-based document discovery via graph
  |- Enrich with doc_types from DB
  |
  v (parallel fan-out)
  ├─ Summary Retriever (if summary/hybrid strategy)
  ├─ Content Retriever (type-aware hybrid search + context expansion)
  └─ Graph Retriever (relationship traversal, skip tech_doc/report)
  |
  v (fan-in)
Context Aggregator
  |- Merge + deduplicate
  |- Type-specific formatting (conversation timestamps, section refs)
  |- Episode context injection for conversations
  |- Confidence assessment (multi-signal scoring)
  |
  v
Generator
  |- Confidence-adaptive system prompt
  |- Query-type specific guidance
  |- Provider-specific LLM invocation
  |
  v
Response + source_refs
```

### 2. Document Ingestion Flow

```
User uploads document
  |
  v
Validate (slug, file, duplicates)
  |
  v
Parse (doc_type-aware parser selection)
  |- book: Gutenberg/PDF reader -> chapter chunking
  |- script: Scene-based chunking
  |- conversation: Speaker turn chunking with overlap
  |- tech_doc/report/markdown: Section-based chunking
  |
  v
Store metadata -> PostgreSQL (documents table)
  |
  v (parallel)
  ├─ Build BM25 index -> PostgreSQL (bm25_index, bm25_doc_lens)
  ├─ Generate embeddings -> Qdrant
  └─ Extract entities/relationships -> PostgreSQL (graph_*)
  |
  v
Generate summaries (LLM) -> PostgreSQL (summaries)
```

### 3. Note Flow

```
User creates/saves note (from chat or editor)
  |
  v
Store in documents (doc_type='note') + notes table
  |
  v
Async index_note:
  |- Chunk markdown content
  |- Build BM25 + vector indexes
  |- Extract graph entities (if content > 100 tokens)
```

## Privacy Modes

| Mode | ephemeral | internal_mode | LLM Provider | Metrics | Tracing |
|------|-----------|---------------|--------------|---------|---------|
| Normal | false | false | Any | Full | Full |
| Ephemeral | true | false | Any | None | None |
| Internal | false | true | Local only | Full | Full |
| Private | true | true | Local only | None | None |

See `docs/PRIVACY_MODES.md` for detailed specification.

## Document Type System

| Type | Parser | Structure | Graph Schema |
|------|--------|-----------|--------------|
| book | book_parser | Chapters, parts | Person, Location, Event, Concept |
| script | script_parser | Scenes, dialogue | Character, Scene, Prop, Action |
| conversation | conversation_parser | Speaker turns (formal or casual) | Speaker/Person, Topic, Decision + Episodes; casual chats auto-detected and use social extraction schema (Person, Event, SharedContent) |
| tech_doc | tech_doc_parser/markdown_parser | Sections, code blocks | Component, Function, Class, API |
| report | report_parser | Sections | Metric, Company, Trend, Risk |
| note | (markdown chunking) | Paragraphs | Person, Concept, Idea, Question |

## Configuration

### Environment Variables (.env)

```bash
LLM_PROVIDER=openai          # or 'local'
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini     # generator model

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=granite3.2:8b

PG_USER=bookuser
PG_PASS=bookpass
PG_HOST=localhost
PG_PORT=5433
PG_DB=booksdb

QDRANT_HOST=localhost
QDRANT_PORT=6333

PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

### Docker Services (docker-compose.yml)

| Service | Port | Profile |
|---------|------|---------|
| PostgreSQL 16 | 5432 | core |
| Qdrant 1.15.1 | 6333 | core |
| App (Gradio) | 7860 | core |
| Phoenix 12.30+ | 6006 | observability |
| Ollama | 11434 | local-ai |

## Performance

| Operation | Typical Latency |
|-----------|----------------|
| BM25 search | < 50ms |
| Vector search (Qdrant) | 50-200ms |
| Hybrid search (full) | 100-500ms |
| Graph traversal (2 hop) | < 200ms |
| LLM response (OpenAI) | 1-5s |
| LLM response (Ollama) | 5-15s |
| Entity extraction (per batch) | 2-5s |

## Observability

- **Phoenix**: http://localhost:6006 -- LLM traces, tool calls, token costs
- **Metrics**: PostgreSQL query_metrics -- latency, LLM assessment, user feedback
- **Dashboard**: Gradio monitoring tab -- query performance visualization
- Disabled in Ephemeral/Private modes

## References

- [Harness Engineering Guidelines](harness/INDEX.md)
- [Privacy Modes](PRIVACY_MODES.md)
- [Development Phases](DEVELOPMENT_PHASES.md)
- [Local LLM Setup](LOCAL_LLM_REFERENCE.md)
- [Search Quality](../EVALUATION.md)
