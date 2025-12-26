# Ollama Setup for Doc-Mate

Run local LLMs (Llama 3.3 70B) with Doc-Mate using Docker for easy installation and management.

## Why Docker for Ollama?

✅ **Easy Setup** - No manual installation, just `docker compose up`
✅ **Consistent** - Works the same on macOS, Linux, Windows
✅ **Isolated** - Doesn't interfere with system packages
✅ **Persistent** - Models stored in Docker volume (no re-downloads)
✅ **GPU Support** - Easy NVIDIA GPU configuration

---

## Prerequisites

**Required:**
- Docker Desktop (or Docker Engine + Docker Compose)
- 8GB+ RAM for small models (64GB+ recommended for Llama 3.3 70B)
- 50GB+ free disk space for models

**Optional:**
- NVIDIA GPU with drivers installed (for faster inference)

### Install Docker

**macOS:**
```bash
# Download Docker Desktop from https://www.docker.com/products/docker-desktop/
# Or install via Homebrew:
brew install --cask docker
```

**Linux:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

**Windows:**
```
Download Docker Desktop from https://www.docker.com/products/docker-desktop/
```

---

## Quick Start

### 1. Start Ollama

```bash
cd /Users/sethurama/DEV/LM/doc-mate
./scripts/ollama.sh start
```

**Output:**
```
✓ Ollama is running!

API endpoint: http://localhost:11434
OpenAI-compatible endpoint: http://localhost:11434/v1

Next steps:
  1. Pull a model: ./scripts/ollama.sh pull llama3.3:70b
  2. Check status: ./scripts/ollama.sh status
```

### 2. Pull a Model

Choose based on your hardware:

**Option A: Llama 3.3 70B** (Best quality, requires 64GB+ RAM)
```bash
./scripts/ollama.sh pull llama3.3:70b
# Download size: ~40GB
# RAM needed: 64GB minimum (80GB recommended)
# Speed: Slower but highest quality
```

**Option B: Llama 3.2 8B** (Good balance, requires 8GB RAM)
```bash
./scripts/ollama.sh pull llama3.2:8b
# Download size: ~4.7GB
# RAM needed: 8GB
# Speed: Fast with good quality
```

**Option C: Llama 3.2 3B** (Fastest, requires 4GB RAM)
```bash
./scripts/ollama.sh pull llama3.2:3b
# Download size: ~2GB
# RAM needed: 4GB
# Speed: Very fast, acceptable quality
```

### 3. Configure Doc-Mate

Update your `.env` file:

```bash
# Add these lines to .env
LLM_PROVIDER=local
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.3:70b  # or llama3.2:8b or llama3.2:3b
```

### 4. Test It Works

```bash
# Test Ollama is responding
curl http://localhost:11434/api/tags

# Test with doc-mate
python -c "
from src.llm.providers import ModelRouter
router = ModelRouter()
provider = router.get_provider('local')
print(f'Provider: {provider.provider_name}')
print(f'Available: {provider.is_available()}')
"
```

### 5. Run Doc-Mate

```bash
python -m src.mcp_client.chat
# Now using local Llama instead of OpenAI!
```

---

## Management Commands

```bash
# Status and monitoring
./scripts/ollama.sh status       # Check if running + show models
./scripts/ollama.sh logs         # View logs (Ctrl+C to exit)

# Model management
./scripts/ollama.sh list         # List downloaded models
./scripts/ollama.sh pull <model> # Download a new model
./scripts/ollama.sh rm <model>   # Remove a model

# Container management
./scripts/ollama.sh start        # Start Ollama
./scripts/ollama.sh stop         # Stop Ollama
./scripts/ollama.sh restart      # Restart Ollama

# Advanced
./scripts/ollama.sh shell        # Open shell in container
./scripts/ollama.sh clean        # Remove container (keep models)
./scripts/ollama.sh purge        # Remove everything (WARNING!)
```

---

## GPU Support (NVIDIA Only)

If you have an NVIDIA GPU, enable GPU acceleration for **much faster** inference:

### 1. Install NVIDIA Container Toolkit

**Linux:**
```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**macOS:**
```
# GPU support on macOS requires Metal (not NVIDIA)
# Use CPU mode for now
```

### 2. Enable GPU in Docker Compose

Edit `docker-compose.ollama.yml`:

```yaml
services:
  ollama:
    # ... existing config ...

    # Uncomment these lines:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. Restart Ollama

```bash
./scripts/ollama.sh restart
```

### 4. Verify GPU is Used

```bash
./scripts/ollama.sh shell
# Inside container:
nvidia-smi  # Should show GPU usage when running inference
```

---

## Troubleshooting

### Ollama won't start

```bash
# Check Docker is running
docker ps

# Check logs
./scripts/ollama.sh logs

# Restart Docker Desktop
# (macOS: Docker Desktop → Restart)
# (Linux: sudo systemctl restart docker)
```

### Model download fails

```bash
# Check disk space (models are large!)
df -h

# Check internet connection
curl -I https://ollama.com

# Try again (downloads resume automatically)
./scripts/ollama.sh pull llama3.3:70b
```

### "Out of memory" errors

```bash
# Check RAM usage
./scripts/ollama.sh status

# Switch to smaller model
./scripts/ollama.sh pull llama3.2:3b

# Update .env
OLLAMA_MODEL=llama3.2:3b
```

### Doc-Mate not connecting to Ollama

```bash
# 1. Check Ollama is running
./scripts/ollama.sh status

# 2. Test API endpoint
curl http://localhost:11434/api/tags

# 3. Check .env configuration
cat .env | grep LLM_PROVIDER
cat .env | grep OLLAMA

# 4. Test provider
python -c "
from src.llm.providers import LocalProvider
provider = LocalProvider()
print(f'Available: {provider.is_available()}')
"
```

### Port 11434 already in use

```bash
# Find what's using the port
lsof -i :11434

# Stop the other process, or change Ollama port:
# Edit docker-compose.ollama.yml:
#   ports:
#     - "11435:11434"  # Use different port
# Update .env:
#   OLLAMA_BASE_URL=http://localhost:11435/v1
```

---

## Performance Comparison

| Model | Download | RAM | GPU | Speed (tokens/sec) | Quality |
|-------|----------|-----|-----|-------------------|---------|
| llama3.3:70b | 40GB | 64GB+ | 24GB VRAM | 5-15 t/s | ⭐⭐⭐⭐⭐ Best |
| llama3.2:8b  | 4.7GB | 8GB | 8GB VRAM | 20-40 t/s | ⭐⭐⭐⭐ Good |
| llama3.2:3b  | 2GB | 4GB | 4GB VRAM | 40-80 t/s | ⭐⭐⭐ OK |

**Note:** CPU-only is 5-10x slower than GPU.

---

## Privacy Mode

When using Ollama, enable privacy mode to ensure **no data** leaves your machine:

```bash
# In .env
PRIVACY_MODE=true
LLM_PROVIDER=local
```

With privacy mode enabled:
- ✅ All queries processed locally
- ✅ No external API calls
- ✅ No data sent to OpenAI/Anthropic
- ✅ Complete privacy for sensitive documents

---

## Multiple Models

You can have multiple models installed and switch between them:

```bash
# Install multiple models
./scripts/ollama.sh pull llama3.3:70b
./scripts/ollama.sh pull llama3.2:8b
./scripts/ollama.sh pull llama3.2:3b

# List all
./scripts/ollama.sh list

# Switch models by updating .env
OLLAMA_MODEL=llama3.2:8b  # Use 8B for faster responses
```

---

## Disk Space Management

Models are stored in Docker volume `doc-mate-ollama-models`. To free up space:

```bash
# Check disk usage
docker system df -v | grep ollama

# Remove unused models
./scripts/ollama.sh rm llama3.2:3b

# Remove all models (WARNING!)
./scripts/ollama.sh purge
```

---

## Advanced: Running on Remote Server

If you have a powerful server, run Ollama there and connect from your laptop:

**On server:**
```bash
# Edit docker-compose.ollama.yml to expose port:
ports:
  - "0.0.0.0:11434:11434"  # Allow external connections

./scripts/ollama.sh start
./scripts/ollama.sh pull llama3.3:70b
```

**On laptop:**
```bash
# In .env
OLLAMA_BASE_URL=http://YOUR_SERVER_IP:11434/v1
LLM_PROVIDER=local
```

**Security note:** Use VPN or SSH tunnel for production:
```bash
# SSH tunnel (secure)
ssh -L 11434:localhost:11434 user@server

# In .env (use localhost via tunnel)
OLLAMA_BASE_URL=http://localhost:11434/v1
```

---

## Next Steps

1. **Test with OpenAI first** to ensure everything works
2. **Start Ollama** with `./scripts/ollama.sh start`
3. **Pull a small model first** (llama3.2:3b) to test
4. **Upgrade to larger model** if quality is insufficient
5. **Enable privacy mode** for sensitive documents

---

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.com/library)
- [Docker Documentation](https://docs.docker.com/)
- [Doc-Mate Phase 2 Plan](/.claude/plans/smooth-sleeping-dream.md)

---

**Questions?** Create an issue: https://github.com/yourusername/doc-mate/issues
