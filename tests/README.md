# Tests

## Structure

```
tests/
├── unit/              # Unit tests (pytest)
│   ├── test_pattern_builder.py
│   ├── test_pdf_reader.py
│   └── test_search.py
└── integration/       # Integration tests (requires services)
    ├── test_agent_tool_calling.py
    ├── test_local_llm.py
    └── test_local_llm_edge_cases.py
```

## Running Tests

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests

#### OpenAI Agent Tests
```bash
# Requires: OPENAI_API_KEY, PostgreSQL, Qdrant, indexed books
source .env && python -m tests.integration.test_agent_tool_calling
```

#### Local LLM Tests
```bash
# Requires: Ollama running, PostgreSQL, Qdrant, indexed books
# Set LLM_PROVIDER=local in .env
python -m tests.integration.test_local_llm

# Edge case tests (unit-style, no services needed)
python -m tests.integration.test_local_llm_edge_cases
```

## Coverage

**Unit**: Pattern builder, PDF reader, search components

**Integration**:
- OpenAI agent tool calling (comprehensive, ~5 min)
- Local LLM function calling (8 tests, ~2 min)
- Local LLM edge cases (parameter normalization, prompts, temperature)
