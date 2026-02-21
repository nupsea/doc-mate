# Entropy Management

Technical debt tracking, cleanup cadence, and golden principles for Doc-Mate.

## Golden Principles

These are opinionated, mechanical rules that keep the codebase consistent. They apply to every file and every change.

### 1. Shared utilities over hand-rolled helpers
If a utility function is used in more than one module, it belongs in `src/utils/`. Do not duplicate logic across modules.

### 2. Validate at boundaries, not everywhere
Data validation happens at system entry points (see SECURITY.md). Internal function calls trust their callers. Do not add redundant `isinstance` checks or null guards deep in the call stack.

### 3. Typed boundaries
Public function signatures must have type hints. Internal helpers may omit them if the types are obvious from context.

### 4. Structured over ad-hoc
Prefer dataclasses, TypedDict, or Pydantic models over raw dictionaries for data that crosses module boundaries. Within a single function, plain dicts are fine.

### 5. Explicit over implicit
No magic strings for configuration. Use constants, enums, or configuration files. If a string appears in more than one place, extract it.

### 6. Parser extensibility
New document types are added by creating a new parser in `src/content/parsers/` that extends `BaseParser`. Do not add format-specific logic to existing parsers.

### 7. Test mirroring
Every new module in `src/` should have a corresponding test file. The test file name matches: `src/foo/bar.py` -> `tests/unit/test_bar.py`.

## Technical Debt Tracker

| Item | Severity | Area | Notes |
|------|----------|------|-------|
| No UI automated tests | Medium | Testing | Gradio testing utilities available but unused |
| Graph entity resolution edge cases | Medium | Graph | Ambiguous entity names not handled well |
| Local LLM test coverage | Low | Testing | Integration tests exist but coverage is thin |
| BM25 index lifecycle | Low | Search | Index files accumulate without cleanup on document deletion |
| ui/ingest.py at 740 lines | Low | Architecture | Should be split into parsing and UI components |
| ~~Missing structural tests~~ | ~~Medium~~ | ~~CI~~ | RESOLVED: test_architecture.py enforces dependency direction |
| ~~store.py over 600 lines~~ | ~~Low~~ | ~~Architecture~~ | RESOLVED: note operations extracted to note_store.py |

## Cleanup Cadence

### Weekly
- Review any files that have grown beyond 500 lines
- Check for duplicated utility functions across modules
- Verify that new modules have corresponding tests

### After Each Feature
- Update QUALITY.md grades if a domain improved or degraded
- Update this debt tracker if new debt was introduced
- Ensure documentation references are current

### Monthly
- Run a full search quality evaluation and compare against EVALUATION.md baselines
- Review dependency list for unused or outdated packages
- Check documentation freshness (are cross-references still valid?)

## Pattern Replication Risk

Agents replicate patterns they find in the codebase, including suboptimal ones. When fixing a bad pattern:

1. Fix the original instance
2. Search for all copies of the pattern (`grep` for characteristic code snippets)
3. Fix all instances in the same PR
4. If the pattern is structural, add a lint rule or structural test to prevent recurrence
5. Document the preferred pattern in the relevant harness doc

## Refactoring Guidelines

- **Small, focused refactoring PRs** over large rewrites
- **One pattern change per PR**: don't mix "rename X" with "restructure Y"
- **Preserve behavior**: refactoring must not change external behavior. If behavior changes, it is a feature or bug fix, not a refactoring
- **Tests must pass**: if tests fail after refactoring, the refactoring introduced a bug
- **Update documentation**: if a refactoring changes file paths, module names, or public interfaces, update all references
