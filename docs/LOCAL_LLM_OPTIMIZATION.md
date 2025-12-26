# Local LLM Optimization Guide

## Overview

This guide documents the optimizations implemented to prevent system crashes when running local LLMs with Ollama.

## Problem Summary

The local LLM was crashing macOS due to:
1. Code defaulting to llama3.3:70b (64GB+ RAM requirement)
2. No Docker resource limits (could consume all system memory)
3. No request timeouts (hanging requests could lock resources)
4. No concurrency controls (multiple requests could overwhelm system)

## Solutions Implemented

### 1. Safe Model Defaults

**Changed in:**
- `src/llm/config.py`
- `src/llm/providers/local_provider.py`

**What changed:**
- Default model changed from `llama3.3:70b` to `llama3.2:3b`
- Prevents accidental loading of large models

### 2. Docker Resource Limits

**Changed in:**
- `docker-compose.yml`
- `docker-compose.ollama.yml`

**Limits applied:**
```yaml
deploy:
  resources:
    limits:
      memory: 6G      # Safe for llama3.2:3b (4GB model + 2GB overhead)
      cpus: '4.0'     # Limit CPU usage
    reservations:
      memory: 4G      # Minimum memory guarantee
```

**Environment variables:**
```yaml
OLLAMA_NUM_PARALLEL: 1          # Only one concurrent request
OLLAMA_MAX_LOADED_MODELS: 1     # Only one model in memory
```

### 3. Request Timeouts

**Changed in:**
- `src/llm/providers/local_provider.py`

**What changed:**
- Added 60-second timeout to all requests
- Prevents hanging requests from locking resources

### 4. Concurrency Controls

**Changed in:**
- `src/llm/providers/local_provider.py`

**What changed:**
- Added semaphore to limit concurrent requests (max 2)
- Prevents system overload from multiple simultaneous requests
- Applied to async requests only (most common use case)

### 5. Token Generation Limits

**Changed in:**
- `src/llm/providers/local_provider.py`

**What changed:**
- Default `max_tokens=2048` if not specified
- Prevents runaway generation consuming excessive resources

## Recommended Model Settings

### For Most Macs (8-16GB RAM)
```bash
OLLAMA_MODEL=llama3.2:3b
```
- Model size: 2GB download
- RAM required: 4GB
- Docker limit: 6GB (default)
- Performance: 40-80 tokens/sec

### For High-End Macs (32GB+ RAM)
```bash
OLLAMA_MODEL=llama3.2:8b
```
- Model size: 4.7GB download
- RAM required: 8GB
- Docker limit: **Update to 10GB in docker-compose.yml**
- Performance: 20-40 tokens/sec

**To use 8B model, update docker-compose.yml:**
```yaml
memory: 10G  # Changed from 6G
```

### NOT Recommended for Most Macs
```bash
OLLAMA_MODEL=llama3.3:70b
```
- Model size: 40GB download
- RAM required: 64GB+
- Docker limit: **Would need 70GB+ in docker-compose.yml**
- Only use on high-end workstations with 128GB+ RAM

## Safety Checklist

Before running local LLM:

- [ ] Verify `.env` has `OLLAMA_MODEL=llama3.2:3b` (or 8b)
- [ ] Check Docker memory limit matches model (6GB for 3b, 10GB for 8b)
- [ ] Ensure Mac has sufficient available RAM (8GB+ free recommended)
- [ ] Close other resource-intensive applications
- [ ] Test with small queries first before production use

## Monitoring Resource Usage

### Check Ollama Container Resources
```bash
./scripts/ollama.sh status
```

### Real-time Docker Stats
```bash
docker stats doc-mate-ollama
```

### Check Mac System Resources
- Activity Monitor → Memory tab
- Watch for memory pressure (should stay green/yellow)

## Troubleshooting

### System Still Running Slow
1. Reduce concurrent requests:
   - Edit `src/llm/providers/local_provider.py`
   - Change `max_concurrent_requests: int = 2` to `= 1`

2. Lower Docker memory limit (forces Ollama to be more conservative):
   ```yaml
   memory: 4G  # Minimum for llama3.2:3b
   ```

### Out of Memory Errors
1. Check which model is actually loaded:
   ```bash
   ./scripts/ollama.sh list
   ```

2. If wrong model loaded:
   ```bash
   ./scripts/ollama.sh rm llama3.3:70b  # Remove large model
   ./scripts/ollama.sh pull llama3.2:3b  # Pull small model
   ```

3. Restart Ollama:
   ```bash
   ./scripts/ollama.sh restart
   ```

### Request Timeouts
- Default timeout is 60 seconds
- For longer documents, increase timeout in code:
  ```python
  LocalProvider(timeout=120)  # 2 minutes
  ```

## Performance Tuning

### Faster Responses (Lower Quality)
```python
provider.chat_completion(
    messages=messages,
    temperature=0.3,      # Lower = more deterministic
    max_tokens=512        # Shorter responses
)
```

### Better Quality (Slower)
```python
provider.chat_completion(
    messages=messages,
    temperature=0.7,      # Higher = more creative
    max_tokens=2048       # Longer responses
)
```

## Configuration Reference

### Environment Variables
```bash
# Provider selection
LLM_PROVIDER=local              # Use local Ollama
PRIVACY_MODE=true               # Force local-only, no fallback

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2:3b        # Model name
OLLAMA_NUM_PARALLEL=1           # Concurrent requests (Docker env)
OLLAMA_MAX_LOADED_MODELS=1      # Models in memory (Docker env)
```

### Docker Resource Limits
```yaml
memory: 6G          # Max memory (adjust for model)
cpus: '4.0'         # Max CPU cores
reservations:
  memory: 4G        # Minimum guaranteed memory
```

### Code Settings
```python
LocalProvider(
    base_url="http://localhost:11434/v1",
    model="llama3.2:3b",
    timeout=60,                    # Request timeout (seconds)
    max_concurrent_requests=2      # Max parallel requests
)
```

## Testing After Changes

1. Restart Docker services:
   ```bash
   docker compose down
   docker compose up -d
   ```

2. Check Ollama is running:
   ```bash
   ./scripts/ollama.sh status
   ```

3. Test with simple query:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "llama3.2:3b",
     "prompt": "Hello!",
     "stream": false
   }'
   ```

4. Monitor resources during test:
   ```bash
   docker stats doc-mate-ollama
   ```

## Additional Resources

- Ollama Documentation: https://ollama.com/library
- Model Comparison: See `docs/OLLAMA_SETUP.md`
- Docker Resource Limits: https://docs.docker.com/config/containers/resource_constraints/

## Version History

- 2024-01-XX: Initial optimizations implemented
  - Safe model defaults (3b instead of 70b)
  - Docker resource limits
  - Request timeouts
  - Concurrency controls
  - Token generation limits
