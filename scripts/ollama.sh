#!/bin/bash
# Ollama Docker Management Script for Doc-Mate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.ollama.yml"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function print_usage() {
    cat << EOF
Ollama Docker Management for Doc-Mate

Usage: $0 <command> [options]

Commands:
    start           Start Ollama container
    stop            Stop Ollama container
    restart         Restart Ollama container
    status          Check Ollama status
    logs            Show Ollama logs (Ctrl+C to exit)

    pull <model>    Pull a model (e.g., llama3.3:70b, llama3.2:3b)
    list            List downloaded models
    rm <model>      Remove a model

    shell           Open shell in Ollama container
    clean           Stop and remove container (keeps models)
    purge           Remove everything including models (WARNING!)

Examples:
    $0 start
    $0 pull llama3.3:70b
    $0 pull llama3.2:3b
    $0 status
    $0 logs

Recommended Models:
    llama3.3:70b    - Best quality (requires 64GB+ RAM)
    llama3.2:8b     - Good balance (requires 8GB RAM)
    llama3.2:3b     - Fastest (requires 4GB RAM)
EOF
}

function check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        echo "Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker daemon is not running${NC}"
        echo "Please start Docker Desktop or docker service"
        exit 1
    fi
}

function start_ollama() {
    echo -e "${GREEN}Starting Ollama container...${NC}"
    docker compose -f "$COMPOSE_FILE" up -d

    echo -e "${YELLOW}Waiting for Ollama to be ready...${NC}"
    sleep 5

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        echo -e "${GREEN}✓ Ollama is running!${NC}"
        echo ""
        echo "API endpoint: http://localhost:11434"
        echo "OpenAI-compatible endpoint: http://localhost:11434/v1"
        echo ""
        echo "Next steps:"
        echo "  1. Pull a model: $0 pull llama3.3:70b"
        echo "  2. Check status: $0 status"
        echo "  3. Test: curl http://localhost:11434/api/tags"
    else
        echo -e "${RED}✗ Failed to start Ollama${NC}"
        echo "Check logs with: $0 logs"
        exit 1
    fi
}

function stop_ollama() {
    echo -e "${YELLOW}Stopping Ollama container...${NC}"
    docker compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✓ Ollama stopped${NC}"
}

function restart_ollama() {
    stop_ollama
    sleep 2
    start_ollama
}

function show_status() {
    echo -e "${GREEN}Ollama Status:${NC}"
    echo ""

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        echo -e "Status: ${GREEN}Running${NC}"

        # Check if API is responding
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "API: ${GREEN}Responding${NC}"

            # Show downloaded models
            echo ""
            echo "Downloaded models:"
            docker exec doc-mate-ollama ollama list 2>/dev/null || echo "  (none)"
        else
            echo -e "API: ${RED}Not responding${NC}"
        fi

        # Show resource usage
        echo ""
        echo "Resource usage:"
        docker stats doc-mate-ollama --no-stream --format "  CPU: {{.CPUPerc}}\n  Memory: {{.MemUsage}}"
    else
        echo -e "Status: ${RED}Not running${NC}"
        echo ""
        echo "Start with: $0 start"
    fi
}

function show_logs() {
    echo -e "${GREEN}Ollama logs (Ctrl+C to exit):${NC}"
    docker compose -f "$COMPOSE_FILE" logs -f
}

function pull_model() {
    local model="$1"

    if [ -z "$model" ]; then
        echo -e "${RED}Error: Model name required${NC}"
        echo "Usage: $0 pull <model>"
        echo ""
        echo "Popular models:"
        echo "  llama3.3:70b   - Best quality (40GB download, 64GB+ RAM)"
        echo "  llama3.2:8b    - Good balance (4.7GB download, 8GB RAM)"
        echo "  llama3.2:3b    - Fastest (2GB download, 4GB RAM)"
        exit 1
    fi

    echo -e "${GREEN}Pulling model: $model${NC}"
    echo -e "${YELLOW}This may take a while (models are large)...${NC}"
    echo ""

    docker exec -it doc-mate-ollama ollama pull "$model"

    echo ""
    echo -e "${GREEN}✓ Model pulled successfully!${NC}"
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:11434/api/generate -d '{\"model\": \"$model\", \"prompt\": \"Hello!\"}'"
}

function list_models() {
    echo -e "${GREEN}Downloaded models:${NC}"
    echo ""
    docker exec doc-mate-ollama ollama list 2>/dev/null || {
        echo "  (none)"
        echo ""
        echo "Pull a model with: $0 pull llama3.3:70b"
    }
}

function remove_model() {
    local model="$1"

    if [ -z "$model" ]; then
        echo -e "${RED}Error: Model name required${NC}"
        echo "Usage: $0 rm <model>"
        exit 1
    fi

    echo -e "${YELLOW}Removing model: $model${NC}"
    docker exec doc-mate-ollama ollama rm "$model"
    echo -e "${GREEN}✓ Model removed${NC}"
}

function open_shell() {
    echo -e "${GREEN}Opening shell in Ollama container...${NC}"
    docker exec -it doc-mate-ollama bash
}

function clean() {
    echo -e "${YELLOW}This will stop and remove the container (models will be preserved)${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✓ Container removed (models preserved in volume)${NC}"
    else
        echo "Cancelled"
    fi
}

function purge() {
    echo -e "${RED}WARNING: This will remove EVERYTHING including downloaded models!${NC}"
    echo "You will need to re-download models (40GB+ for llama3.3:70b)"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose -f "$COMPOSE_FILE" down -v
        echo -e "${GREEN}✓ Everything removed${NC}"
    else
        echo "Cancelled"
    fi
}

# Main script
check_docker

case "${1:-}" in
    start)
        start_ollama
        ;;
    stop)
        stop_ollama
        ;;
    restart)
        restart_ollama
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    pull)
        pull_model "$2"
        ;;
    list|ls)
        list_models
        ;;
    rm|remove)
        remove_model "$2"
        ;;
    shell|sh)
        open_shell
        ;;
    clean)
        clean
        ;;
    purge)
        purge
        ;;
    help|--help|-h|"")
        print_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
