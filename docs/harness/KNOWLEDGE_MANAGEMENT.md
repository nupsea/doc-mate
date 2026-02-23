# Knowledge Management

Documentation standards, progressive disclosure, and freshness guidelines for Doc-Mate.

## Documentation Hierarchy

```
CLAUDE.md / AGENTS_GUIDE.md     -- Entry point (~100 lines, table of contents)
    |
    v
docs/harness/INDEX.md           -- Harness engineering guidelines index
docs/ARCHITECTURE.md            -- System architecture reference
docs/DEVELOPMENT_PHASES.md      -- Roadmap and phase tracking
    |
    v
docs/harness/*.md               -- Detailed guidelines by topic
docs/PRIVACY_MODES.md           -- Privacy mode specification
docs/LOCAL_LLM_REFERENCE.md     -- Local LLM setup guide
    |
    v
EVALUATION.md                   -- Search quality benchmarks
CONTRIBUTING.md                 -- Contribution process
```

## Progressive Disclosure

Agents and new contributors should be able to navigate from a small, stable entry point to any depth of detail:

1. **Level 0** (always in context): AGENTS_GUIDE.md -- repository layout, key conventions, where to find things
2. **Level 1** (on demand): Architecture, quality grades, core beliefs
3. **Level 2** (when working in a specific area): Domain-specific docs, API references, design decisions

No single file should exceed 300 lines. If it does, split into sub-documents and link from the parent.

## What Must Be Documented

| Category | Location | Trigger for Update |
|----------|----------|--------------------|
| Architecture decisions | docs/ARCHITECTURE.md | New component, changed dependency |
| Schema changes | init.sql + scripts/ | Any DDL change |
| New document type parser | docs/ARCHITECTURE.md, AGENTS_GUIDE.md | New parser added |
| Privacy mode behavior | docs/PRIVACY_MODES.md | Privacy logic change |
| Quality assessments | docs/harness/QUALITY.md | Quarterly or after major feature |
| Search quality baselines | EVALUATION.md | After search algorithm change |
| Development roadmap | docs/DEVELOPMENT_PHASES.md | Phase completion or new phase |

## What Should NOT Be Documented

- Implementation details that are clear from reading the code
- Temporary workarounds (use code comments with TODO instead)
- Meeting notes or discussion summaries (extract decisions into relevant docs)
- Third-party library API references (link to upstream docs instead)

## Freshness Rules

1. **Stale documentation is worse than no documentation**: if a doc contradicts the code, the code is the source of truth and the doc must be updated
2. **Cross-references must resolve**: any link to another file must point to a file that exists
3. **Version-sensitive content**: anything that references specific versions, counts, or metrics should be dated or marked as "as of [date]"
4. **Dead code references**: if documentation references a module or function that no longer exists, remove the reference

## Documentation Review Checklist

When modifying docs, verify:
- [ ] Links to other files resolve correctly
- [ ] File paths mentioned in docs match actual repository structure
- [ ] Code examples are syntactically valid
- [ ] No contradictions with other documentation
- [ ] Table of contents / index entries updated if new doc added
