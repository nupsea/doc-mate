# Doc-Mate Development Phases

**Last Updated**: February 2026

## Overview

Doc-Mate development is organized into phases, with Phases 1-3 complete, Phase 5 partially started, and Phases 4, 6-7 planned.

**Current Status**: Phase 3 Complete, Phase 5 (Notes) In Progress

---

## Phase 1: Multi-Document Type Support -- COMPLETE

**Goal**: Extend beyond books to support scripts, technical docs, reports, and conversations

### Delivered
- Document-type aware parsers (book, script, conversation, tech_doc, report, markdown)
- Database schema with doc_type constraint and JSONB metadata
- Type-aware search with doc_type filtering
- Speaker attribution for conversations, scene-based search for scripts
- Code block extraction for tech docs
- UI support for all document types

---

## Phase 2: Local LLM & Privacy Modes -- COMPLETE

**Goal**: Add local LLM support and flexible privacy controls

### Delivered
- Ollama integration with provider abstraction layer (OpenAI + Local)
- 4 privacy modes: Normal, Ephemeral, Internal, Private
- Modular prompt system (YAML config + Python builder)
- NoOpQueryTimer for ephemeral modes
- Docker Compose profiles (core, observability, local-ai)
- Makefile commands for native Ollama (recommended for Mac)

---

## Phase 3: Knowledge Graph & Conversation Intelligence -- COMPLETE

**Goal**: Add knowledge graph for entity relationships, conversation episodes, and retrieval confidence

### Delivered

**Knowledge Graph (PostgreSQL-based, not Neo4j)**:
- Entity extraction via LLM (doc-type aware schemas for 7 doc types)
- Relationship storage with weight accumulation
- Multi-hop graph traversal (recursive CTE, 1-2 hops)
- Entity-based document discovery (content-based routing)
- Entity resolution and linking

**Conversation Intelligence**:
- Episode management (speaker, stance, topic, summary)
- Adaptive chunking with overlap for conversations
- Social chat detection heuristics (emojis, short turns)
- Context expansion (adjacent chunk retrieval for conversations)
- Episode context injection in aggregator

**LangGraph RRG Pipeline**:
- Router-Retriever-Generator architecture with parallel fan-out
- Summary, Content, and Graph retrievers running concurrently
- Context aggregator with deduplication and type-specific formatting
- Confidence assessment (multi-signal: density, retrieval score, entity coverage, lexical overlap)
- Confidence-adaptive generator prompting (high/medium/low)
- Query type classification (broad, entity, inference, factual)

**Adaptive Search**:
- Triple hybrid fusion: BM25 + Vector + Graph (with RRF)
- Query preprocessing (stop word removal, +8% Hit@5)
- ConversationGraphRetriever with episode matching
- Semantic entity matching via SentenceTransformer

**Infrastructure**:
- BM25 indexes moved to PostgreSQL (bm25_index, bm25_doc_lens tables)
- Cascade deletes on all child tables
- Ingest job tracking (survives browser disconnects)
- Ingestion profiles

**Test Coverage**:
- Agent basic, conversation intelligence, document types, edge cases
- Graph knowledge, Homer comprehensive (Iliad/Odyssey relationships)
- Local grounding (basic + graph), ephemeral mode, search quality
- 14 integration test files

---

## Phase 5: Notes (In Progress)

**Goal**: Note-taking with document provenance and search integration

### Delivered So Far
- Notes table with tags, source_refs (JSONB provenance), pinning, versioning
- Notes UI with list/editor split view
- Save-as-note from chat responses (captures source_refs + query)
- Note CRUD operations in PgresStore
- Lightweight note indexing pipeline (chunk, BM25, vector, optional graph)
- DOC_TYPE_SCHEMAS for notes (Person, Concept, Idea, Question, Insight)
- Migration script: `scripts/add_notes_tables.sql`

### Remaining
- [ ] Integration tests for notes
- [ ] Note search across all notes
- [ ] Cross-document note references
- [ ] Export functionality

---

## Phase 4: Images, Code & Architecture Analysis -- PLANNED

**Goal**: Extract and analyze images, diagrams, code snippets from technical documents

### Scope
- PDF image extraction and OCR
- Code block search by functionality
- Multi-modal embeddings (text + images)
- Architecture diagram parsing

---

## Phase 6: UI/UX Enhancements -- PLANNED

**Goal**: Modernize UI and improve user experience

### Scope
- Streaming responses
- Document preview panel
- Graph visualization (D3.js/Cytoscape)
- Search results highlighting
- Dark mode

---

## Phase 7: Optimization & Polish -- PLANNED

**Goal**: Performance optimization, test coverage, production readiness

### Scope
- Database query optimization and index tuning
- Search quality regression framework
- Unit test coverage improvement
- Monitoring dashboards (Prometheus/Grafana)
- Production deployment guide

---

## Timeline Summary

| Phase | Status | Key Features |
|-------|--------|--------------|
| Phase 1 | Complete | Multi-document types, parsers |
| Phase 2 | Complete | Local LLM, privacy modes |
| Phase 3 | Complete | Knowledge graph, conversation intelligence, RRG pipeline |
| Phase 5 | In Progress | Notes with provenance |
| Phase 4 | Planned | Images, code analysis |
| Phase 6 | Planned | UI/UX enhancements |
| Phase 7 | Planned | Optimization, testing, polish |

Note: Phase ordering was adjusted based on priority. Phase 5 (Notes) was pulled ahead of Phase 4 (Images) due to user demand.
