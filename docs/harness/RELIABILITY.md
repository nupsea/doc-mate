# Reliability Guidelines

Performance budgets, observability requirements, and reliability constraints for Doc-Mate.

## Performance Budgets

| Operation | Target | Acceptable | Needs Investigation |
|-----------|--------|------------|---------------------|
| BM25 search | < 50ms | < 100ms | > 200ms |
| Vector search (Qdrant) | < 200ms | < 500ms | > 1s |
| Hybrid search (full pipeline) | < 500ms | < 1s | > 2s |
| LLM response (OpenAI) | < 5s | < 10s | > 15s |
| LLM response (Local/Ollama) | < 15s | < 30s | > 60s |
| Document ingestion (per chunk) | < 200ms | < 500ms | > 1s |
| Graph entity extraction | < 2s | < 5s | > 10s |
| UI page load | < 2s | < 4s | > 6s |

These targets are measured at the application level, not including network latency to the user.

## Observability Stack

### Phoenix Tracing (OpenTelemetry)
- **Endpoint**: `http://localhost:6006`
- **What is traced**: LLM calls, tool usage, token costs, latency per step
- **Disabled in**: Ephemeral and Private privacy modes
- **Configuration**: `src/monitoring/tracer.py`

### Application Metrics
- **Collection**: `src/monitoring/metrics.py`
- **Storage**: PostgreSQL `query_metrics` table
- **Dashboard**: `src/monitoring/dashboard.py` via Gradio
- **Key metrics**:
  - Query latency breakdown (routing, retrieval, generation)
  - LLM self-assessment scores (EXCELLENT / ADEQUATE / POOR)
  - User feedback ratings (1-5)
  - Tool call frequency and success rates
  - Token usage per query

### Structured Logging
- Use Python's `logging` module with structured messages
- Include correlation IDs for tracing query flow across components
- Log levels:
  - ERROR: service failures, data corruption, unrecoverable states
  - WARNING: degraded performance, fallback behavior triggered
  - INFO: normal operations (ingestion started, query completed)
  - DEBUG: detailed internal state (disabled in production)

## Error Handling Strategy

### Retriable Errors
- LLM API timeouts: retry with exponential backoff (max 3 attempts)
- Database connection drops: reconnect via connection pool
- Qdrant search failures: fall back to BM25-only search

### Non-Retriable Errors
- Invalid document format: return clear error to user, do not retry
- Missing API key: surface configuration error, do not retry
- Schema validation failure: log and reject, do not retry

### Graceful Degradation
- If vector search is unavailable: fall back to BM25 keyword search
- If graph store is unavailable: skip graph-based retrieval, proceed with search
- If Phoenix is unavailable: continue without tracing (log a warning)
- If Ollama is unavailable in Internal/Private mode: inform user, do not fall back to external API

## Health Checks

Docker services define health checks in `docker-compose.yml`:
- PostgreSQL: `pg_isready`
- Qdrant: HTTP health endpoint
- App: HTTP endpoint on port 7860

Application-level health should verify:
- Database connectivity
- Qdrant connectivity
- At least one LLM provider available
- BM25 index directory accessible

## Capacity Guidelines

- **Document size**: tested with documents up to ~1MB text
- **Chunk count**: tested with up to ~5000 chunks per document
- **Concurrent queries**: single-user optimized (Gradio default)
- **Vector dimensions**: 384 (sentence-transformers bge-small-en) or 1536 (OpenAI)
- **BM25 index size**: proportional to document count, stored on disk
