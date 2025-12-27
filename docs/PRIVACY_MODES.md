# Privacy Modes

BookMate offers flexible privacy controls to match your data sensitivity and infrastructure requirements.

## Available Modes

### 1. Normal (Default)
**What it does:**
- Full metrics collection and tracing
- Can use OpenAI API or local LLM
- Complete observability via Phoenix UI
- User feedback and ratings saved

**Use when:**
- General usage
- Want best model quality (OpenAI)
- Need metrics for monitoring/debugging
- Building analytics

### 2. Ephemeral
**What it does:**
- NO metrics/tracing saved
- Can still use OpenAI API
- Conversation only in memory
- Data cleared on session end

**Use when:**
- Privacy-sensitive queries
- Don't want tracking
- Still need powerful models (GPT-4)
- Testing without polluting metrics

**Example:** Querying proprietary documents with OpenAI but no logging

### 3. Internal
**What it does:**
- Forces local LLM only (Ollama)
- NO external API calls
- Metrics still collected locally
- All processing on-premise

**Use when:**
- Air-gapped environments
- Can't use external APIs
- Compliance requires on-premise
- Cost optimization (no API costs)

**Example:** Enterprise deployment with strict data residency requirements

### 4. Private (Ephemeral + Internal)
**What it does:**
- Forces local LLM only
- NO metrics/tracing saved
- NO external API calls
- Maximum privacy

**Use when:**
- Highly sensitive data
- Zero external dependencies
- Complete privacy required
- Compliance + security

**Example:** Medical/legal documents that cannot leave infrastructure

## Comparison Matrix

| Feature | Normal | Ephemeral | Internal | Private |
|---------|--------|-----------|----------|---------|
| Metrics Saved | ✓ | ✗ | ✓ | ✗ |
| Phoenix Tracing | ✓ | ✗ | ✓ | ✗ |
| OpenAI API | ✓ | ✓ | ✗ | ✗ |
| Local LLM | ✓ | ✓ | ✓ (forced) | ✓ (forced) |
| External Calls | Yes | Yes | No | No |
| Data Privacy | Moderate | High | Moderate | Maximum |
| Model Quality | Best | Best | Good | Good |
| Cost | API costs | API costs | Free | Free |

## Usage

### UI (Gradio)
Select privacy mode from radio buttons in chat interface:
- Normal
- Ephemeral (no tracking)
- Internal (local LLM only)
- Private (ephemeral + internal)

### Programmatic (Python)
```python
from src.mcp_client.agent import BookMateAgent

# Normal mode (default)
agent = BookMateAgent(provider="openai")

# Ephemeral mode (OpenAI but no tracking)
agent = BookMateAgent(provider="openai", ephemeral=True)

# Internal mode (local LLM, with metrics)
agent = BookMateAgent(internal_mode=True)

# Private mode (maximum privacy)
agent = BookMateAgent(ephemeral=True, internal_mode=True)
```

## Technical Details

### What Gets Tracked

**Normal & Internal modes:**
- Query text
- Response text
- Tool calls (search_book, etc.)
- Latency metrics
- Result counts
- LLM assessment scores
- User feedback ratings

**Ephemeral & Private modes:**
- Nothing persisted
- Query ID prefixed with `ephemeral_`
- Conversation only in memory
- Cleared on session end

### Model Availability

**Normal & Ephemeral:**
- GPT-4o Mini (recommended)
- GPT-4o
- GPT-4 Turbo
- GPT-3.5 Turbo
- Llama 3.1 8B (local)

**Internal & Private:**
- Llama 3.1 8B (local) only
- Provider dropdown disabled

## FAQ

**Q: Can I switch modes mid-conversation?**
A: Yes, but it reinitializes the agent and clears conversation history.

**Q: Does ephemeral mode work with OpenAI?**
A: Yes! Ephemeral only affects local tracking, not LLM choice.

**Q: What's the difference between ephemeral and private?**
A: Ephemeral = no tracking, any LLM. Private = no tracking + local LLM only.

**Q: Which mode is fastest?**
A: Normal/Ephemeral with GPT-4o Mini (API is faster than local LLM).

**Q: Which mode is most private?**
A: Private mode (ephemeral + internal) - everything local, nothing saved.

**Q: Can admin see ephemeral queries?**
A: No, ephemeral queries are never logged or stored anywhere.

**Q: Does internal mode affect search quality?**
A: No, search (BM25 + vector) works the same across all modes.
