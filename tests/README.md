# Tests

## Structure

```
tests/
├── unit/              # Unit tests (logic, parsing, logic-only)
│   ├── test_chat_export_parsing.py
│   ├── test_conversation_history.py
│   ├── test_pattern_builder.py
│   └── ...
├── integration/       # Integration tests (requires services: Postgres, Qdrant, Ollama)
│   ├── test_agent_basic.py
│   ├── test_agent_conversation_intelligence.py (New: Modular RRG)
│   ├── test_agent_homer_comprehensive.py (New: Iliad/Odyssey Graph)
│   ├── test_local_grounding_graph.py
│   └── ...
└── debug/             # Debugging scripts for inspecting graph/DB
    ├── debug_gossip_entities.py
    ├── inspect_graph.py
    └── ...
```

## Running Tests

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests

Ensure your `.env` is loaded and services are running.

#### Agent Comprehensive Tests
```bash
# Modular RRG and Type-Aware Intelligence
PYTHONPATH=. uv run python tests/integration/test_agent_conversation_intelligence.py
PYTHONPATH=. uv run python tests/integration/test_agent_homer_comprehensive.py
```

#### Graph & Grounding Tests
```bash
# Graph relationship tests
PYTHONPATH=. uv run python tests/integration/test_agent_graph_knowledge.py

# Local model grounding tests
PYTHONPATH=. uv run python tests/integration/test_local_grounding_graph.py
```

## Debugging
Use the scripts in `tests/debug/` to inspect the Knowledge Graph content directly in the database.
