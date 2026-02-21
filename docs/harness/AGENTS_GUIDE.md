# Agents Guide

Entry point for agents working in the Doc-Mate repository. This file is a map, not a manual. Follow the links for detail.

## Repository Layout

```
doc-mate/
  src/
    ui/          -- Gradio interface (app, chat, ingest, monitoring, notes)
    content/     -- Document processing, storage, parsers
    flows/       -- LangGraph orchestration, ingestion, query pipelines
    search/      -- BM25, vector, hybrid search, adaptive chunking
    graph/       -- Knowledge graph (entities, relationships, retrieval)
    llm/         -- LLM abstraction, providers, routing
    mcp_client/  -- Agent orchestration (BookMateAgent)
    monitoring/  -- Metrics, tracing, dashboards
    utils/       -- Shared utilities
  tests/
    unit/        -- No external services required
    integration/ -- Requires Postgres, Qdrant, optionally Ollama
    debug/       -- Database inspection scripts
  docs/
    harness/     -- Engineering guidelines (you are here)
  scripts/       -- Database migrations
  DATA/          -- Sample documents for testing
  INDEXES/       -- BM25 search indexes
```

## Architecture Rules

See [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md) for dependency direction enforcement.

**Layer order** (dependencies flow downward only):
```
UI -> Flows -> Search / Graph / LLM -> Content -> Utils
         \-> MCP Client -> Flows
Monitoring sits alongside all layers (cross-cutting)
```

## Key Conventions

- **Python 3.12**, managed by `uv`
- **Linting**: `ruff check src/` -- see `ruff.toml` for rules
- **Tests**: `pytest` with `asyncio_mode = auto`
- **Naming**: snake_case for files and functions, PascalCase for classes
- **Imports**: absolute imports from `src.` namespace
- **Type hints**: required on public function signatures
- **Docstrings**: required on public classes and modules, not on private helpers

## Before Making Changes

1. Read the relevant source files first
2. Check `docs/harness/ARCHITECTURE_RULES.md` for dependency constraints
3. Check `docs/harness/QUALITY.md` for domain-specific quality expectations
4. Run `ruff check src/` before committing
5. Run relevant unit tests: `.venv/bin/python -m pytest tests/unit/ -v`

## Where to Find Things

| Need | Location |
|------|----------|
| Database schema | `init.sql`, `scripts/` |
| API/tool definitions | `src/flows/agent_tools.py` |
| Prompt templates | `src/mcp_client/prompts/config.yaml` |
| Docker services | `docker-compose.yml` |
| Environment config | `.env_template` |
| Search quality benchmarks | `EVALUATION.md` |
| Development roadmap | `docs/DEVELOPMENT_PHASES.md` |
| Privacy mode behavior | `docs/PRIVACY_MODES.md` |
