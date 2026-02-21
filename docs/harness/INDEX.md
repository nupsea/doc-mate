# Harness Engineering Guidelines

Adapted from OpenAI's agent-first development practices for the Doc-Mate project.

**Core philosophy**: Humans steer. Agents execute. The repository is the system of record.

## Document Index

| Document | Purpose |
|----------|---------|
| [AGENTS_GUIDE.md](AGENTS_GUIDE.md) | How agents should navigate and work in this repository |
| [ARCHITECTURE_RULES.md](ARCHITECTURE_RULES.md) | Layered architecture enforcement and dependency rules |
| [QUALITY.md](QUALITY.md) | Quality grading by domain and layer |
| [TESTING.md](TESTING.md) | Testing strategy, coverage expectations, structural tests |
| [SECURITY.md](SECURITY.md) | Security boundaries, validation rules, threat model |
| [RELIABILITY.md](RELIABILITY.md) | Reliability constraints, performance budgets, observability |
| [CI_CD.md](CI_CD.md) | CI pipeline, merge philosophy, automated enforcement |
| [KNOWLEDGE_MANAGEMENT.md](KNOWLEDGE_MANAGEMENT.md) | Documentation standards, progressive disclosure, freshness |
| [ENTROPY_MANAGEMENT.md](ENTROPY_MANAGEMENT.md) | Technical debt tracking, cleanup cadence, golden principles |
| [CORE_BELIEFS.md](CORE_BELIEFS.md) | Operating principles for agent-first development |

## How to Use These Guidelines

1. **Agents** should start with `AGENTS_GUIDE.md` as their entry point. It serves as a table of contents, not a manual.
2. **Human reviewers** should check PRs against `ARCHITECTURE_RULES.md` and `QUALITY.md`.
3. **CI enforcement** should be wired from `CI_CD.md` and `TESTING.md`.
4. **Periodic maintenance** follows `ENTROPY_MANAGEMENT.md` for cleanup cadence.

## Applicability to Doc-Mate

These guidelines are adapted to Doc-Mate's specific stack:
- Python 3.12 with uv package management
- LangGraph orchestration (Router-Retriever-Generator)
- PostgreSQL + Qdrant + BM25 hybrid search
- Gradio UI
- Phoenix/OpenTelemetry observability
- Docker Compose deployment
