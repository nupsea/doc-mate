.PHONY: up down start build logs dev observability full local-ai

# Include .env file
include .env
export

# --- Core Commands ---

# Default: Start core services (lightweight)
up:
	docker compose --profile core up -d
	@$(MAKE) wait-ui

# Alias for up
dev: up

# Start with Phoenix tracing enabled
observability:
	docker compose --profile core --profile observability up -d
	@$(MAKE) wait-ui

# Start with Docker-based Ollama (Backup option)
full:
	docker compose --profile full --profile local-ai up -d
	@$(MAKE) wait-ui

# Check/Start Native Local AI (Recommended for Mac)
local-ai:
	@echo "Checking for Native Ollama..."
	@if curl -s http://localhost:11434 > /dev/null; then \
		echo "Native Ollama detected! Starting Doc-Mate..."; \
		docker compose --profile core up -d; \
		$(MAKE) wait-ui; \
	else \
		echo "Native Ollama NOT found on port 11434."; \
		echo ""; \
		echo "To use Private/Internal mode with hardware acceleration:"; \
		echo "1. Open a new terminal"; \
		echo "2. Run this command:"; \
		echo "   OLLAMA_HOST=0.0.0.0 ollama serve"; \
		echo ""; \
		echo "Once running, try 'make local-ai' again."; \
		exit 1; \
	fi

down:
	docker compose --profile '*' down

build:
	docker compose build

logs:
	docker compose logs -f app

# --- Helpers ---

wait-ui:
	@echo "Waiting for Gradio UI to be ready..."
	@timeout=60; \
	while [ $$timeout -gt 0 ]; do \
		if curl -s http://localhost:7860 > /dev/null 2>&1; then \
			echo "All services started! Access UI at http://localhost:7860"; \
			exit 0; \
		fi; \
		sleep 1; \
		timeout=$$((timeout - 1)); \
	done; \
	echo "Warning: UI did not start within 60 seconds. Check logs with: make logs"

start: up
	@echo "Book Mate is running at http://localhost:7860"