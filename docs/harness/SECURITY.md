# Security Guidelines

Security boundaries, validation rules, and threat model for Doc-Mate.

## Threat Model

Doc-Mate processes user-uploaded documents and queries them via LLMs. Primary attack surfaces:

| Surface | Risk | Mitigation |
|---------|------|------------|
| Document upload | Malicious file content, path traversal | File type validation, sandboxed parsing |
| User queries | Prompt injection via chat input | Input sanitization, system prompt hardening |
| LLM responses | Hallucinated or harmful content | Confidence scoring, source attribution |
| Database | SQL injection, unauthorized access | Parameterized queries, connection pooling |
| API keys | Credential exposure | Environment variables, never in code or logs |
| Docker services | Container escape, network exposure | Internal networking, minimal port exposure |

## Validation Boundaries

Data must be validated at these system boundaries:

### 1. Document Ingestion (src/content/, src/flows/document_ingest.py)
- Validate file type before processing
- Sanitize file names (no path traversal characters)
- Enforce maximum file size limits
- Validate parsed content structure before database insertion

### 2. User Input (src/ui/chat.py)
- Sanitize query text before passing to LLM
- Validate provider/model selection against allowed values
- Validate privacy mode transitions

### 3. LLM Responses (src/llm/generator.py)
- Parse structured LLM output with error handling
- Do not execute or eval any LLM-generated code
- Validate confidence scores are within expected ranges

### 4. Database Operations (src/content/store.py, src/graph/store.py)
- Use parameterized queries exclusively (never string interpolation for SQL)
- Validate data types before insertion
- Handle connection failures gracefully

### 5. External API Calls (src/llm/providers/)
- Never log API keys or tokens
- Handle rate limits and API errors without exposing internal state
- Validate response schemas before processing

## Secrets Management

- **API keys**: stored in `.env`, never committed to git
- **Database credentials**: stored in `.env`, passed via Docker environment
- **`.env` is in `.gitignore`**: verified. `.env_template` provides the structure without values
- **No secrets in logs**: structured logging must exclude sensitive fields
- **No secrets in error messages**: catch and sanitize before surfacing to UI

## Privacy Mode Security

See `docs/PRIVACY_MODES.md` for full specification.

| Mode | Constraint |
|------|-----------|
| Ephemeral | No traces, no metrics, no persistent query data |
| Internal | Local LLM only, no external API calls |
| Private | Local LLM only AND no persistent data |

Violations of privacy mode constraints (e.g., calling OpenAI in Internal mode) must be treated as security bugs.

## Dependency Security

- Pin all dependencies in `uv.lock` (frozen)
- Review new dependency additions for:
  - Maintenance status (last commit, open issues)
  - Transitive dependency count
  - Known vulnerabilities
- Prefer well-established libraries over niche alternatives
- When a small utility function suffices, implement it locally rather than adding a dependency

## Docker Security

- Run application as non-root user in container
- Use specific image tags, not `latest`
- Limit container resources (memory, CPU) where applicable
- Internal services (Postgres, Qdrant) should not be exposed to host network in production
- Health checks on all services
