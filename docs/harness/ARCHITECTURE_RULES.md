# Architecture Rules

Enforced dependency directions and structural constraints for Doc-Mate.

## Layer Dependency Graph

```
                   ┌──────────┐
                   │    UI    │
                   └────┬─────┘
                        │
              ┌─────────▼─────────┐
              │    MCP Client     │
              └────┬──────────┬───┘
                   │          │
              ┌────▼─────┐    │
         ┌───►│  Flows   │◄───┘
         │    └────┬─────┘
         │         │
    ┌────┴───┐ ┌───▼───┐ ┌────────┐
    │ Search ├─► Graph ├─► LLM    │
    └────┬───┘ └───┬───┘ └────┬───┘
         │         │          │
         └────┬────┘──────────┘
              │
         ┌────▼─────┐
         │ Content  │
         └────┬─────┘
              │
         ┌────▼─────┐
         │  Utils   │
         └──────────┘

  Cross-cutting: Monitoring (imports from Content, LLM, Utils)
  Search -> Graph: triple hybrid fusion (BM25 + Vec + Graph)
  Graph -> LLM: entity extraction requires LLM
```

## Dependency Rules

### Allowed
- UI imports from MCP Client, Flows, Content, Monitoring, Utils
- Flows imports from Search, Graph, LLM, Content, Monitoring, Utils
- Search imports from Content, Graph, Utils  (Graph needed for triple hybrid fusion)
- Graph imports from Content, LLM, Utils  (LLM needed for entity extraction)
- LLM imports from Utils
- Content imports from Utils
- MCP Client imports from Flows, LLM, Content, Monitoring, Utils
- Monitoring imports from Content, LLM, Utils  (LLM needed for response judging)
- Any layer imports from standard library and third-party packages

### Forbidden
- Content MUST NOT import from Search, Graph, Flows, UI, MCP Client, Monitoring
- LLM MUST NOT import from Flows, UI, Search, Graph, Content, MCP Client, Monitoring
- Utils MUST NOT import from any src/ module
- UI MUST NOT import from Search, Graph, LLM directly (go through Flows or MCP Client)
- Search MUST NOT import from Flows, UI, LLM, MCP Client, Monitoring
- Graph MUST NOT import from Flows, UI, Search, MCP Client, Monitoring
- Monitoring MUST NOT import from Flows, UI, Search, Graph, MCP Client
- No circular dependencies between modules at the same layer

## File Organization Rules

1. **One module per concern**: a file should have a single, clear responsibility
2. **Max file size**: aim for under 500 lines per file. Files exceeding 600 lines should be split
3. **Parser pattern**: new document type parsers go in `src/content/parsers/` and extend `BaseParser`
4. **Provider pattern**: new LLM providers go in `src/llm/providers/` and extend the base provider
5. **Test mirroring**: each `src/` module should have a corresponding test file in `tests/unit/` or `tests/integration/`

## Data Validation Boundaries

Validate and parse data at these boundaries (see [SECURITY.md](SECURITY.md)):
- User input from Gradio UI
- LLM responses before processing
- Database reads before returning to callers
- External API responses (OpenAI, Ollama)
- File content during document ingestion

Internal function calls between trusted modules within the same layer do not need redundant validation.

## Schema Management

- Database schema lives in `init.sql` (initial) and `scripts/` (migrations)
- Schema changes require a migration script in `scripts/`
- Generated schema documentation goes in `docs/` and must be kept in sync
- Qdrant collection configuration is defined in `src/search/vec.py`
