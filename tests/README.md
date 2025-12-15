# Tests

## Structure

```
tests/
├── unit/              # Unit tests (pytest)
│   ├── test_pattern_builder.py
│   ├── test_pdf_reader.py
│   └── test_search.py
└── integration/       # Integration tests (requires services)
    └── test_agent_tool_calling.py
```

## Running Tests

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
# Requires: OPENAI_API_KEY, PostgreSQL, Qdrant, indexed books
source .env && python -m tests.integration.test_agent_tool_calling
```

## Coverage

**Unit**: Pattern builder, PDF reader, search components
**Integration**: MCP agent tool calling (22 tests, ~3-5 min)
