# Doc-Mate

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A universal document assistant powered by LLMs that lets you upload and chat with any document type - books, technical documentation, movie scripts, reports, conversations, and more. Features hybrid search (BM25 + vector embeddings), multi-model support (OpenAI, Anthropic, local), and optional privacy mode.

## Features

### Core Capabilities
- **Multi-Format Support**: Books, tech docs, PDFs, scripts, reports, conversations
- **Intelligent Parsing**: Document-type aware parsing (chapters, scenes, sections, turns)
- **Hybrid Search**: Combines BM25 keyword search with semantic vector search
- **Multi-Modal**: Text + images + tables (coming in Phase 3)
- **AI Chat Interface**: Ask questions with context-aware responses and citations
- **Multi-Document Search**: Compare themes across multiple documents simultaneously

### Model Flexibility
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo
- **Local Models**: Llama 3.1 8B via Ollama
- **Privacy Modes**: 4 flexible privacy levels (Normal, Ephemeral, Internal, Private)
  - **Normal**: Full observability with any model
  - **Ephemeral**: No tracking, still uses cloud APIs
  - **Internal**: Local LLM only, metrics collected
  - **Private**: No tracking + local LLM only (maximum privacy)

### Advanced Features
- **Automatic Query Retry**: Rephrases and retries when no results found
- **Smart Citations**: All responses include source references with page/section numbers
- **Monitoring Dashboard**: Track query performance, LLM assessments, user feedback
- **MCP Integration**: Uses Model Context Protocol for tool-based agent interactions

## Current Status

**Version**: 0.2.0
**Phase**: Phase 2 Complete ✅ (Multi-document types + Local LLM + Privacy modes)
**Next**: Phase 3 - Graph Database Integration

See [docs/DEVELOPMENT_PHASES.md](docs/DEVELOPMENT_PHASES.md) for detailed roadmap.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.12+
- OpenAI API Key (for cloud models)
- **Optional**: Ollama for local LLM support (privacy modes)

### Installation

Clone the repository:

```bash
git clone https://github.com/nupsea/doc-mate.git
cd doc-mate
```

Set up environment:

```bash
cp .env_template .env
```

Update `.env` with your API keys:

```bash
# Required for cloud models
OPENAI_API_KEY=sk-...

# Optional - Local LLM settings
LLM_PROVIDER=openai              # or 'local' for Ollama
LOCAL_MODEL=llama3.1:8b          # Ollama model name
OLLAMA_BASE_URL=http://ollama:11434  # Ollama endpoint (Docker service)
```

### Option 1: OpenAI Only (Quick Start)

```bash
make build
make start
```

### Option 2: With Local LLM (Ollama in Docker - Recommended)

**Start all services including Ollama:**

```bash
docker-compose -f docker-compose.ollama.yml up -d
```

**Pull model (first time only):**

```bash
docker exec -it doc-mate-ollama ollama pull llama3.1:8b
```

**Verify:**

```bash
docker exec -it doc-mate-ollama ollama list
```

### Option 3: Ollama Installed Separately (Alternative)

If running Ollama outside Docker:

```bash
# Install on host
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve
```

Update `.env`:
```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434  # macOS/Windows
# OR
OLLAMA_BASE_URL=http://172.17.0.1:11434            # Linux
```

Then start:
```bash
make start
```

---

The UI will be available at http://localhost:7860
Phoenix tracing UI will be available at http://localhost:6006

**Useful commands:**
```bash
make up      # Start all services
make down    # Stop all services
make logs    # View app logs
make build   # Rebuild the app container
```

## Using Doc-Mate

### 1. Add a Document

Go to the "Add Document" tab:

1. Upload your document file (.txt, .pdf, .md)
2. Select document type (book, tech_doc, script, etc.)
3. Enter title, author, and unique slug
4. Provide structure pattern (for books: chapter pattern)
5. Click "Test Pattern" to verify detection
6. Click "Ingest Document" to index the content

**Document Type Examples:**

| Type | Examples | Structure Pattern |
|------|----------|-------------------|
| **Book** | Novels, classics | `CHAPTER I`, `Part I` |
| **Tech Doc** | API docs, manuals | Markdown headings (`##`, `###`) |
| **Script** | Screenplays | `INT.`, `EXT.`, `FADE IN:` |
| **Report** | Business reports | `Executive Summary`, sections |
| **Conversation** | Chat logs, transcripts | `Speaker:`, `[00:00:00]` |

### 2. Chat with Documents

Go to the "Chat" tab:

1. **Select Document** (optional - leave empty for multi-doc queries)
2. **Choose Provider**: OpenAI or Local Ollama
3. **Select Model**:
   - OpenAI: GPT-4o Mini (recommended), GPT-4o, GPT-4 Turbo
   - Local: Llama 3.1 8B
4. **Choose Privacy Mode**:
   - **Normal**: Full metrics/tracing, any model (default)
   - **Ephemeral**: No tracking, still uses OpenAI
   - **Internal**: Local LLM only, metrics collected
   - **Private**: No tracking + local LLM only (maximum privacy)
5. Ask questions:
   - "What is this document about?"
   - "Find examples of authentication code"
   - "Compare Marcus Aurelius and Homer on persistence"
6. Rate responses (in Normal mode)

**Query Examples by Document Type:**

**Books:**
- "What are the main themes in this book?"
- "Compare how Marcus Aurelius and Homer view heroism"

**Tech Docs:**
- "Show me code examples for OAuth2 implementation"
- "What are the API rate limits?"

**Scripts:**
- "Summarize the opening scene"
- "Which characters have the most dialogue?"

**Reports:**
- "What are the Q4 revenue numbers?"
- "Extract all key metrics from this report"

### 3. Monitor Performance

Go to the "Monitoring" tab:
- Query performance metrics (latency, success rate)
- LLM self-assessments (EXCELLENT/ADEQUATE/POOR)
- User feedback ratings
- Model usage statistics
- Query retry statistics

### 4. View LLM Traces (Phoenix)

Go to http://localhost:6006 for detailed LLM observability:
- View all API calls with full prompts and responses
- Trace conversation flows and tool usage
- Analyze token usage and costs
- Debug and optimize LLM interactions

## Architecture

```
┌─────────────┐
│  Gradio UI  │
└──────┬──────┘
       │
┌──────▼────────────┐         ┌──────────────┐
│  DocMateAgent     │◄───────►│  LLM Provider│
│  (MCP Client)     │         │  Router      │
└──────┬────────────┘         └──────┬───────┘
       │                             │
       │                    ┌────────▼────────┐
       │                    │  OpenAI API     │
       │                    │  Anthropic API  │
       │                    │  Local (Ollama) │
       │                    └─────────────────┘
       │
┌──────▼────────────┐
│   MCP Server      │
│  (Document Tools) │
└──────┬────────────┘
       │
       ├──────────►  ╭─────────────────────╮
       │             │    PostgreSQL       │
       │             ├─────────────────────┤
       │             │ Metadata, Summaries │
       │             ╰─────────────────────╯
       │
       ├──────────►  [BM25 Index] (Keyword Search)
       │
       └──────────►  ╭─────────────────────╮
                     │      Qdrant         │
                     ├─────────────────────┤
                     │   Vector Search     │
                     ╰─────────────────────╯
```

### Key Components

- **PostgreSQL**: Stores document metadata, summaries, and query metrics
- **Qdrant**: Vector database for semantic search
- **BM25**: Keyword-based search index
- **MCP Server**: Exposes document tools (search, summaries) to agent
- **LLM Router**: Selects optimal model (OpenAI/Anthropic/Local) based on query
- **Phoenix**: LLM observability via OpenTelemetry traces

## Project Structure

```
doc-mate/
├── src/
│   ├── app/              # Main application entry points
│   ├── content/          # Document parsing and storage
│   │   ├── parsers/      # Document-type specific parsers (Phase 1)
│   │   └── store.py      # PostgreSQL storage
│   ├── flows/            # Ingestion and query workflows
│   ├── llm/              # LLM interfaces
│   │   └── providers/    # Multi-model support (Phase 2)
│   ├── mcp_client/       # MCP client agent implementation
│   ├── mcp_server/       # MCP server with document tools
│   ├── monitoring/       # Metrics collection and quality assessment
│   ├── search/           # Hybrid search (BM25 + vector)
│   │   └── multimodal.py # Image search (Phase 3)
│   └── ui/               # Gradio UI components
├── notebooks/            # Jupyter notebooks for exploration
├── DATA/                 # Document files
│   └── images/           # Extracted images (Phase 3)
├── INDEXES/              # BM25 search indexes
├── scripts/              # Database migrations
├── DEVELOPMENT_PHASES.md # Detailed roadmap
├── MIGRATION.md          # Migration guide from book-mate
└── docker-compose.yml    # PostgreSQL + Qdrant services
```

## Development Roadmap

**Current Status**: Phase 2 Complete ✅ (Multi-document types + Local LLM + Privacy modes)

See [docs/DEVELOPMENT_PHASES.md](docs/DEVELOPMENT_PHASES.md) for detailed roadmap and implementation plans.

## Privacy Modes

Doc-Mate offers 4 flexible privacy levels:

### Mode Comparison

| Mode | Tracking | LLM Options | Use Case |
|------|----------|-------------|----------|
| **Normal** | Full metrics | OpenAI or Local | General usage, observability |
| **Ephemeral** | No tracking | OpenAI or Local | Sensitive queries, powerful models |
| **Internal** | Metrics saved | Local only | Air-gapped, compliance |
| **Private** | No tracking | Local only | Maximum privacy |

### How to Use

**In UI:**
- Select Privacy Mode dropdown in Chat tab
- Choose from: Normal, Ephemeral, Internal, Private
- Provider/Model update automatically based on mode

**Programmatically:**

```python
from src.mcp_client.agent import BookMateAgent

# Ephemeral: No tracking, can use OpenAI
agent = BookMateAgent(provider="openai", ephemeral=True)

# Internal: Local LLM only, metrics saved
agent = BookMateAgent(internal_mode=True)

# Private: Maximum privacy (ephemeral + internal)
agent = BookMateAgent(ephemeral=True, internal_mode=True)
```

### What Gets Tracked

**Normal & Internal:**
- Query/response text
- Tool calls, latency
- LLM assessments
- Phoenix traces

**Ephemeral & Private:**
- Nothing saved
- No database writes
- No Phoenix traces
- Memory only

See [docs/PRIVACY_MODES.md](docs/PRIVACY_MODES.md) for complete details.

For local LLM setup and testing, see [docs/LOCAL_LLM_REFERENCE.md](docs/LOCAL_LLM_REFERENCE.md).

## Model Selection

| Provider | Model | Best For | Cost | Speed |
|----------|-------|----------|------|-------|
| **OpenAI** | GPT-4o Mini | General queries | $0.15/M tokens | Fast (1-3s) |
| **OpenAI** | GPT-4o | Complex analysis | $2.50/M tokens | Fast (2-4s) |
| **OpenAI** | GPT-4 Turbo | Advanced reasoning | $5/M tokens | Fast (2-5s) |
| **Local** | Llama 3.1 8B | Privacy, offline | Free | Slow (5-15s) |

**Recommendations:**
- **Default**: GPT-4o Mini (best speed/cost/quality balance)
- **Complex queries**: GPT-4o (comparative analysis, nuanced reasoning)
- **Privacy**: Llama 3.1 8B (offline, no external calls)
- **Air-gapped**: Llama 3.1 8B (compliance requirements)

## Database Management

```bash
# Check indexed documents
psql -h localhost -U bookuser -d booksdb -c "SELECT doc_slug, title, doc_type FROM documents;"

# Check metrics
psql -h localhost -U bookuser -d booksdb -c "SELECT COUNT(*), AVG(latency_ms) FROM query_metrics;"

# View recent queries
psql -h localhost -U bookuser -d booksdb -c "SELECT query, llm_relevance_score FROM query_metrics ORDER BY timestamp DESC LIMIT 10;"
```

## Migrating from Book-Mate

See [MIGRATION.md](MIGRATION.md) for detailed migration instructions.

**Quick migration:**
```bash
cd /path/to/book-mate
pg_dump -h localhost -U bookuser booksdb > backup.sql

cd /path/to/doc-mate
psql -h localhost -U bookuser -d booksdb -f scripts/migrate_to_docmate.sql
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority areas:**
- Document type parsers (scripts, reports, conversations)
- Local model optimization
- UI/UX improvements
- Test coverage

## Evaluation

Search quality benchmarks (from book-mate baseline):

```
Hybrid RRF (current):
{
  "hit_rate_at_5": 0.814,
  "mrr_at_5": 0.609,
  "total_queries": 495
}
```

See [EVALUATION.md](EVALUATION.md) for methodology and ground truth generation.

## References & Acknowledgements

**LLM Zoomcamp**: Thanks to DataTalks.Club for foundational learnings
https://datatalks.club/courses/llm-zoomcamp/

**Project Gutenberg**: Free public domain texts
https://www.gutenberg.org/

**Original Project**: Evolved from [book-mate](https://github.com/nupsea/book-mate)

## License

GNU General Public License v3.0 - see LICENSE file for details.

---

## What's Next?

1. **Start with Phase 1**: Implement enhanced document parsing
2. **Add multi-model support** (Phase 2): Support Anthropic and local models
3. **Enable images** (Phase 3): Extract and search diagrams
4. **Specialized features** (Phase 4): Script analysis, conversation stats

Join the journey: [Issues](https://github.com/nupsea/doc-mate/issues) | [Discussions](https://github.com/nupsea/doc-mate/discussions)
