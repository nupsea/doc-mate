# Doc-Mate Workflows

Core logic for document ingestion, querying, and agent orchestration.

## Key Modules

### 1. `document_ingest.py` (Ingestion Pipeline)
Handles the end-to-end process of adding a document to the system.

**Stages**:
1. **Validation**: Checks file existence, format, and slug uniqueness.
2. **Parsing**: Uses type-specific parsers (Book, PDF, Conversation, Script) to chunk text.
3. **Summarization**: Generates hierarchical summaries (Chapter -> Document) using LLM.
4. **Storage**: Persists metadata, summaries, and chunks to PostgreSQL.
5. **Graph Extraction**: Extracts entities and relationships into the Knowledge Graph (Postgres).
6. **Indexing**: Builds BM25 (Keyword) and Vector (Semantic) search indexes.

**Graph Integration**:
- Extracts entities ("Achilles", "Patroclus") and relationships ("companion_of") automatically.
- Resolves duplicates (e.g. merging "Achilles" and "Pelides").
- Stores graph data in `graph_entities` and `graph_relationships` tables.
- **Ephemeral Mode**: Skips Phoenix tracing but still persists graph for session privacy.

**Usage**:
```python
from src.flows.document_ingest import ingest_document

await ingest_document(
    slug="ili",
    file_path="DATA/the_iliad.txt",
    title="The Iliad",
    doc_type="book",
    ephemeral=False  # Set True for privacy mode
)
```

### 2. `agent_graph.py` (LangGraph Agent)
Defines the reactive agent state machine that powers the Chat UI.

**Nodes**:
- `agent`: The reasoning engine (LLM) that decides what to do.
- `tools`: The execution layer that runs selected tools.

**Logic**:
- **System Prompt**: Enforces tool usage and document inference rules.
- **Model Router**: Switches between OpenAI and Local (Ollama) models based on user selection.
- **Tool Loop**: Agent -> Tools -> Agent (Reasoning) -> Final Answer.

### 3. `agent_tools.py` (Tool Definitions)
Wraps core logic into LangChain-compatible tools.

- `search_document`: Hybrid search (BM25 + Vector + Graph).
- `explore_entity_graph`: specialized tool for relationship queries ("How is X related to Y?").
- `get_summary`: Retrieves high-level document summaries.
- `search_multiple_documents`: Cross-document comparison.

### 4. `document_query.py` (Retrieval Logic)
Low-level query interface used by the tools.

- **Triple Hybrid Fusion**: Combines scores from:
  1. **BM25** (35%): Exact keyword matches.
  2. **Vector** (35%): Semantic similarity.
  3. **Graph** (30%): Entity relationship proximity.

## Running Tests

Integration tests for these flows are available in `tests/integration/`.

```bash
# Test ingestion
uv run tests/integration/test_mini_ingest.py

# Test agent reasoning
uv run tests/integration/test_agent_graph.py
```