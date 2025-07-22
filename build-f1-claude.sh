#!/bin/bash

# F1 Claude Code - Easy Manager Script for Linux/Mac
# Usage: ./build-f1-claude.sh [command]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions
print_success() { echo -e "${GREEN}$1${NC}"; }
print_info() { echo -e "${CYAN}$1${NC}"; }
print_warning() { echo -e "${YELLOW}$1${NC}"; }
print_error() { echo -e "${RED}$1${NC}"; }

# Check if Docker is running
check_docker() {
    if ! docker ps >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker."
        exit 1
    fi
}

# Check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Please edit .env and add your ANTHROPIC_API_KEY"
            echo "Press Enter after adding your API key..."
            read -r
        else
            print_error ".env.example not found!"
            exit 1
        fi
    fi
}

# Main script
COMMAND=${1:-help}

echo ""
print_info "🏎️  F1 Claude Code Manager"
print_info "========================="
echo ""

check_docker

case "$COMMAND" in
    help)
        print_info "Available commands:"
        echo "  ./build-f1-claude.sh setup      - First time setup (build images)"
        echo "  ./build-f1-claude.sh jupyter    - Start Jupyter Lab for F1 notebooks"
        echo "  ./build-f1-claude.sh claude     - Start Claude Code CLI"
        echo "  ./build-f1-claude.sh all        - Start all services"
        echo "  ./build-f1-claude.sh stop       - Stop all services"
        echo "  ./build-f1-claude.sh clean      - Stop and remove all containers/volumes"
        echo "  ./build-f1-claude.sh status     - Show running services"
        echo "  ./build-f1-claude.sh logs       - Show logs"
        echo ""
        ;;
    
    setup)
        print_info "🔧 Setting up F1 Claude Code..."
        check_env_file
        
        print_info "Building Docker images..."
        docker-compose -f docker/docker-compose.yml build
        
        if [ $? -eq 0 ]; then
            print_success "\n✅ Setup complete! Run './build-f1-claude.sh jupyter' to start."
        else
            print_error "Build failed!"
            exit 1
        fi
        ;;
    
    jupyter)
        print_info "🚀 Starting Jupyter Lab..."
        check_env_file
        
        docker-compose -f docker/docker-compose.yml up -d f1-jupyter
        
        if [ $? -eq 0 ]; then
            sleep 3
            print_success "\n✅ Jupyter Lab is running!"
            print_info "📊 Open http://localhost:8888 in your browser"
            print_info "📝 Available notebooks:"
            echo "   - F1_Improved_Models.ipynb"
            echo "   - F1_Constructor_Driver_Evaluation.ipynb"
            echo "   - F1_Betting_Market_Models.ipynb"
            echo ""
            print_info "To stop: ./build-f1-claude.sh stop"
            
            # Optionally open browser
            echo -n "Open browser now? (y/n): "
            read -r OPEN_BROWSER
            if [ "$OPEN_BROWSER" = "y" ]; then
                if command -v xdg-open > /dev/null; then
                    xdg-open http://localhost:8888
                elif command -v open > /dev/null; then
                    open http://localhost:8888
                fi
            fi
        fi
        ;;
    
    claude)
        print_info "🤖 Starting Claude Code CLI..."
        check_env_file
        
        print_info "Entering interactive Claude session (type 'exit' to quit)..."
        docker-compose -f docker/docker-compose.yml run --rm claude-code claude
        ;;
    
    all)
        print_info "🚀 Starting all services..."
        check_env_file
        
        docker-compose -f docker/docker-compose.yml up -d
        
        if [ $? -eq 0 ]; then
            sleep 3
            print_success "\n✅ All services started!"
            print_info "📊 Jupyter Lab: http://localhost:8888"
            print_info "🤖 Claude CLI: docker exec -it claude-code-cli bash"
            echo ""
            print_info "Quick commands:"
            echo "  Claude: docker exec -it claude-code-cli claude"
            echo "  Bash:   docker exec -it claude-code-cli bash"
        fi
        ;;
    
    stop)
        print_info "🛑 Stopping all services..."
        docker-compose -f docker/docker-compose.yml down
        print_success "✅ All services stopped."
        ;;
    
    clean)
        print_warning "⚠️  This will remove all containers, images, and data!"
        echo -n "Are you sure? (yes/no): "
        read -r CONFIRM
        if [ "$CONFIRM" = "yes" ]; then
            print_info "🧹 Cleaning up..."
            docker-compose -f docker/docker-compose.yml down -v --rmi all
            print_success "✅ Cleanup complete."
        else
            print_info "Cancelled."
        fi
        ;;
    
    status)
        print_info "📊 Service Status:"
        docker-compose -f docker/docker-compose.yml ps
        ;;
    
    logs)
        print_info "📜 Showing logs (Ctrl+C to exit)..."
        docker-compose -f docker/docker-compose.yml logs -f
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        echo "Run './build-f1-claude.sh help' for available commands."
        exit 1
        ;;
esac
