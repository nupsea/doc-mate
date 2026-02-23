# CI/CD Guidelines

Pipeline configuration, merge philosophy, and automated enforcement for Doc-Mate.

## Current CI Pipeline

Defined in `.github/workflows/ci.yml`. Triggers on push to `master` and pull requests.

### Jobs

1. **Lint** (`ruff check src/`): catches undefined names, unused imports, syntax errors
2. **Docker Build**: validates that the application container builds successfully

### Path Ignores
CI skips runs for changes to: `*.md`, `docs/`, `.gitignore`, `LICENSE`, `*.txt`

## Recommended CI Enhancements

The following should be added incrementally as the project matures:

### Phase 1: Essential Gates
- [ ] **Unit tests in CI**: run `pytest tests/unit/ -v` on every PR
- [ ] **Import validation**: structural test verifying dependency direction rules from ARCHITECTURE_RULES.md
- [ ] **File size check**: flag files exceeding 600 lines

### Phase 2: Quality Gates
- [ ] **Type checking**: add `mypy` or `pyright` for public interfaces
- [ ] **Schema consistency**: verify `init.sql` matches code expectations
- [ ] **Documentation freshness**: check that key docs reference existing files/modules

### Phase 3: Integration
- [ ] **Service-based integration tests**: run against containerized Postgres + Qdrant
- [ ] **Search quality regression**: compare Hit Rate and MRR against baseline in EVALUATION.md

## Merge Philosophy

Adapted from the harness engineering approach: corrections are cheap, waiting is expensive.

### Pull Request Guidelines
- PRs should be small and focused (one concern per PR)
- Unit tests must pass before merge
- Lint must pass before merge
- Integration test failures on infrastructure-dependent tests should not block merge if unit tests pass
- Follow-up PRs for minor issues are preferred over blocking progress

### What Blocks a Merge
- Lint failures
- Unit test failures
- Security violations (credentials in code, SQL injection patterns)
- Architecture rule violations (wrong dependency direction)
- Breaking changes to public interfaces without migration

### What Does NOT Block a Merge
- Style nits (unless they violate naming conventions)
- Missing documentation for internal helpers
- Integration test flakes (file a follow-up)
- Performance concerns (unless exceeding "Needs Investigation" thresholds from RELIABILITY.md)

## Local Development Checks

Before opening a PR, run:

```bash
# Lint
ruff check src/

# Unit tests
.venv/bin/python -m pytest tests/unit/ -v

# Format check (if ruff format is configured)
ruff format --check src/
```

## Automated Enforcement Targets

Rules that should be enforced mechanically (via linters, CI, or structural tests) rather than through code review:

| Rule | Enforcement |
|------|-------------|
| No wildcard imports | ruff (F403) |
| No undefined names | ruff (F821) |
| No unused imports | ruff (F401) |
| Snake_case file names | structural test / CI script |
| Dependency direction | structural test (import analysis) |
| File size limits | CI script |
| Parameterized SQL queries | grep-based CI check for string interpolation in SQL |
| No secrets in code | grep-based CI check for API key patterns |
