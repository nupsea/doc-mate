# Local LLM Reference

Quick reference for local LLM setup, testing, and troubleshooting.

**Last Updated**: February 2026

## Quick Start

### Option A: Native Ollama (Recommended for Mac)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull granite3.2:8b

# Start serving (bind to all interfaces for Docker access)
OLLAMA_HOST=0.0.0.0 ollama serve

# Start Doc-Mate with local AI
make local-ai
```

### Option B: Docker Ollama

```bash
# Start all services including Ollama
make full

# Pull model inside container
docker exec -it doc-mate-ollama ollama pull granite3.2:8b
```

### Configure .env

```bash
LLM_PROVIDER=local
OLLAMA_MODEL=granite3.2:8b
OLLAMA_BASE_URL=http://localhost:11434/v1    # Native Ollama
# OR
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1  # Docker -> host Ollama
```

## Model Selection

| Model | RAM Required | Speed | Quality | Notes |
|-------|-------------|-------|---------|-------|
| granite3.2:8b | 8GB | Medium | Good | Default, good function calling |
| llama3.2:3b | 4GB | Fast | Fair | Lightweight alternative |
| llama3.1:8b | 8GB | Medium | Good | Alternative to Granite |
| llama3.3:70b | 64GB+ | Slow | Excellent | Requires significant RAM |

**Current default**: `granite3.2:8b` (set via `OLLAMA_MODEL` in `.env`)

## Docker Compose Profiles

```bash
make dev          # Core only (Postgres + Qdrant + App) - no Ollama
make up           # Core + Phoenix observability
make full         # All services including Docker Ollama
make local-ai     # Core + detects native Ollama on port 11434
```

### Docker Resource Limits (docker-compose.yml)

```yaml
# Ollama service (local-ai profile)
deploy:
  resources:
    limits:
      memory: 10G
      cpus: '4.0'
    reservations:
      memory: 8G
```

## Testing

### Verify Setup

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list
```

### Run Integration Tests

```bash
# Local LLM tests (requires Ollama + Postgres + Qdrant)
.venv/bin/python -m pytest tests/integration/test_local_llm.py -v

# Local grounding with graph
.venv/bin/python -m pytest tests/integration/test_local_grounding_basic.py -v
.venv/bin/python -m pytest tests/integration/test_local_grounding_graph.py -v
```

### Test Provider Programmatically

```python
from src.llm.providers import ModelRouter

router = ModelRouter()
provider = router.get_provider('local')
print(f'Available: {provider.is_available()}')
print(f'Model: {provider.model}')
```

## Privacy Modes with Local LLM

| Mode | LLM | Metrics | Use Case |
|------|-----|---------|----------|
| Internal | Local only | Collected | Compliance, air-gapped |
| Private | Local only | None | Maximum privacy |

```python
from src.mcp_client.agent import DocMateAgent

# Internal mode (local LLM, with metrics)
agent = DocMateAgent(internal_mode=True)

# Private mode (local LLM, no tracking)
agent = DocMateAgent(ephemeral=True, internal_mode=True)
```

In the UI, selecting "Internal" or "Private" mode automatically forces the local LLM provider and disables the provider dropdown.

## Troubleshooting

### Ollama not responding
```bash
# Check process
pgrep -f ollama

# Check port
curl -s http://localhost:11434/api/tags

# Restart native Ollama
OLLAMA_HOST=0.0.0.0 ollama serve
```

### Out of memory
- Switch to a smaller model: `ollama pull llama3.2:3b`
- Update `OLLAMA_MODEL=llama3.2:3b` in `.env`
- Check usage: `docker stats doc-mate-ollama` (if Docker)

### Slow first response
- First request after model load takes 10-30s (model loading into memory)
- Subsequent requests: 2-10s
- GPU acceleration (NVIDIA only) gives 5-10x speedup

### Connection refused from Docker
```bash
# Native Ollama must bind to 0.0.0.0
OLLAMA_HOST=0.0.0.0 ollama serve

# .env should use:
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1  # macOS/Windows
OLLAMA_BASE_URL=http://172.17.0.1:11434/v1            # Linux
```

## Code Configuration

- Request timeout: 60 seconds (`src/llm/providers/local_provider.py`)
- Default max_tokens: 2048
- Provider selection: `src/llm/providers/model_router.py`
- Fallback disabled in internal_mode (strict local enforcement)

## Resources

- Ollama: https://github.com/ollama/ollama
- Model Library: https://ollama.com/library
