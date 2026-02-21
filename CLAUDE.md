# Doc-Mate Agent Guide

Entry point for agents working in this repository. This is a map, not a manual.

## Project

Doc-Mate is a document intelligence platform: ingest documents, build knowledge graphs, and answer questions via hybrid search (BM25 + Vector + Graph) and LLM generation.

**Stack**: Python 3.12, uv, LangGraph, PostgreSQL, Qdrant, Gradio, Phoenix/OpenTelemetry

## Quick Reference

| Need | Look Here |
|------|-----------|
| Architecture & layers | `docs/ARCHITECTURE.md` |
| Dependency rules | `docs/harness/ARCHITECTURE_RULES.md` |
| Testing strategy | `docs/harness/TESTING.md` |
| Security boundaries | `docs/harness/SECURITY.md` |
| Quality grades | `docs/harness/QUALITY.md` |
| CI pipeline | `docs/harness/CI_CD.md` |
| Reliability budgets | `docs/harness/RELIABILITY.md` |
| Cleanup practices | `docs/harness/ENTROPY_MANAGEMENT.md` |
| Core beliefs | `docs/harness/CORE_BELIEFS.md` |
| All harness guidelines | `docs/harness/INDEX.md` |
| Database schema | `init.sql`, `scripts/` |
| Privacy modes | `docs/PRIVACY_MODES.md` |
| Development roadmap | `docs/DEVELOPMENT_PHASES.md` |
| Search benchmarks | `EVALUATION.md` |

## Key Conventions

- Lint before committing: `ruff check src/`
- Run unit tests: `.venv/bin/python -m pytest tests/unit/ -v`
- Imports: absolute from `src.` namespace
- File naming: snake_case, max ~500 lines
- Type hints: required on public function signatures
- Validate data at system boundaries, trust internal calls
- New parsers extend `BaseParser` in `src/content/parsers/`
- New LLM providers extend base in `src/llm/providers/`

## Source Layout

```
src/
  ui/          Gradio interface
  flows/       LangGraph orchestration
  search/      BM25 + Vector + Hybrid
  graph/       Knowledge graph
  llm/         LLM providers & routing
  content/     Document storage & parsers
  mcp_client/  Agent orchestration
  monitoring/  Metrics & tracing
  utils/       Shared utilities
```

## Dependency Direction (enforce strictly)

```
UI -> Flows -> {Search, Graph, LLM} -> Content -> Utils
Monitoring is cross-cutting (allowed from any layer)
```
