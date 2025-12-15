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

### Model Flexibility (Phase 2)
- **OpenAI**: GPT-4o, GPT-4o-mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku
- **Local Models**: Llama, Mistral, Phi via Ollama/LM Studio
- **Privacy Mode**: Complete offline operation for sensitive documents

### Advanced Features
- **Automatic Query Retry**: Rephrases and retries when no results found
- **Smart Citations**: All responses include source references with page/section numbers
- **Monitoring Dashboard**: Track query performance, LLM assessments, user feedback
- **MCP Integration**: Uses Model Context Protocol for tool-based agent interactions

## Current Status

**Version**: 0.1.0 (based on book-mate)
**Phase**: 0 (Foundation) - Ready for Phase 1 development

See [DEVELOPMENT_PHASES.md](DEVELOPMENT_PHASES.md) for complete roadmap.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.12+
- OpenAI API Key (or Anthropic for Claude)
- Optional: Ollama for local models

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
# Required
OPENAI_API_KEY=sk-...

# Optional (for multi-model support)
ANTHROPIC_API_KEY=sk-ant-...

# Optional (for local/private mode)
LOCAL_MODEL=llama3.2:3b
PRIVACY_MODE=false
```

Build and start services:

```bash
make build
make start
```

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

1. Select document(s) from dropdown (optional - leave empty for multi-doc queries)
2. Choose LLM provider (Auto, OpenAI, Anthropic, Local)
3. Toggle Privacy Mode for local-only processing
4. Ask questions:
   - "What is this document about?"
   - "Find examples of authentication code"
   - "Which characters appear in Act 2?"
   - "Compare leadership styles in these documents"
5. Rate responses to help improve the system

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

## Development Phases

Doc-Mate is under active development. See [DEVELOPMENT_PHASES.md](DEVELOPMENT_PHASES.md) for complete details.

**Roadmap Summary:**

| Phase | Status | Key Features |
|-------|--------|--------------|
| 0: Foundation | ✅ Complete | Repository setup, taxonomy |
| 1: Enhanced Parsing | 🚧 Next | Script, conversation, report parsers |
| 2: Multi-Model | 📅 Planned | OpenAI + Anthropic + Local support |
| 3: Multimodal | 📅 Planned | Image extraction & search |
| 4: Doc Features | 📅 Planned | Type-specific analysis tools |
| 5: UI/UX | 📅 Planned | Enhanced interface |
| 6: Privacy | 📅 Planned | Complete offline operation |
| 7: Testing | 📅 Planned | Quality assurance |

**Timeline**: ~12 weeks to full feature set

## Privacy Mode

Doc-Mate supports complete offline operation for sensitive documents:

### Setup Local Models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2:3b        # Fast text generation
ollama pull llava:7b           # Image understanding
ollama pull nomic-embed-text   # Embeddings
```

### Enable Privacy Mode

```bash
# .env
PRIVACY_MODE=true
LOCAL_MODEL=llama3.2:3b
```

When enabled:
- All queries processed locally (no cloud APIs)
- Embeddings generated locally
- Image captions via local BLIP-2
- Zero data leaves your machine

**Trade-offs:**
- Slower than cloud APIs
- Requires GPU for best performance
- Lower quality for complex reasoning

## Multi-Model Support

Choose the best model for your needs:

| Provider | Models | Best For | Cost |
|----------|--------|----------|------|
| **OpenAI** | GPT-4o, GPT-4o-mini | General purpose, code | $0.15-5/M tokens |
| **Anthropic** | Claude 3.5 Sonnet/Haiku | Complex reasoning, analysis | $0.80-3/M tokens |
| **Local** | Llama 3.2, Mistral | Privacy, cost savings | Free |

**Auto-routing**: Doc-Mate automatically selects the best model based on:
- Query complexity (simple → local, complex → Claude)
- Privacy mode settings
- Cost preferences
- Document type (code → GPT-4, analysis → Claude)

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
