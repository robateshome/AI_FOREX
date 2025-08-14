#!/bin/bash

# AI Forex Signal Generator Startup Script
# This script provides easy startup options for the AI Forex signal generator

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
DOCKER_COMPOSE_FILE="docker-compose.ai.yml"
BACKEND_DIR="$PROJECT_DIR/backend"
LOG_DIR="$PROJECT_DIR/logs"
MODELS_DIR="$PROJECT_DIR/models"

# Create necessary directories
mkdir -p "$LOG_DIR" "$MODELS_DIR"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  AI Forex Signal Generator${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.11+"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.11"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_error "Python $REQUIRED_VERSION+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    # Check if Docker is installed (for Docker deployment)
    if command -v docker &> /dev/null; then
        DOCKER_AVAILABLE=true
        print_status "Docker is available"
    else
        DOCKER_AVAILABLE=false
        print_warning "Docker is not available. Will use local deployment only."
    fi
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        print_warning ".env file not found. Please create one from .env.example"
        if [ -f "$PROJECT_DIR/.env.example" ]; then
            print_status "Copying .env.example to .env"
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
            print_warning "Please edit .env file with your API keys before starting"
        fi
    fi
    
    print_status "Prerequisites check completed"
}

# Function to start with Docker
start_docker() {
    print_status "Starting AI Forex Signal Generator with Docker..."
    
    if [ "$DOCKER_AVAILABLE" = false ]; then
        print_error "Docker is not available. Cannot start with Docker."
        return 1
    fi
    
    cd "$PROJECT_DIR"
    
    # Check if docker-compose file exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        print_error "Docker compose file not found: $DOCKER_COMPOSE_FILE"
        return 1
    fi
    
    # Build and start services
    print_status "Building Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build
    
    print_status "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    print_status "Services started successfully!"
    print_status "API available at: http://localhost:8000"
    print_status "Health check: http://localhost:8000/health"
    
    # Show logs
    print_status "Showing logs (Ctrl+C to stop viewing logs, services will continue running)..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
}

# Function to start locally
start_local() {
    print_status "Starting AI Forex Signal Generator locally..."
    
    cd "$BACKEND_DIR"
    
    # Check if requirements are installed
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        print_warning "Virtual environment not found. Installing dependencies globally..."
        pip3 install -r "$PROJECT_DIR/requirements.txt"
    else
        print_status "Using virtual environment"
        if [ -d "venv" ]; then
            source venv/bin/activate
        elif [ -d ".venv" ]; then
            source .venv/bin/activate
        fi
    fi
    
    # Start the API server
    print_status "Starting API server..."
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    
    print_status "API server started with PID: $API_PID"
    print_status "API available at: http://localhost:8000"
    print_status "Health check: http://localhost:8000/health"
    
    # Wait a moment for the server to start
    sleep 3
    
    # Start CLI dashboard in another terminal if available
    if command -v gnome-terminal &> /dev/null; then
        print_status "Starting CLI dashboard in new terminal..."
        gnome-terminal -- bash -c "cd '$PROJECT_DIR' && python3 cli_dashboard.py; exec bash"
    elif command -v xterm &> /dev/null; then
        print_status "Starting CLI dashboard in new terminal..."
        xterm -e "cd '$PROJECT_DIR' && python3 cli_dashboard.py; exec bash" &
    else
        print_status "CLI dashboard available at: $PROJECT_DIR/cli_dashboard.py"
        print_status "Run in another terminal: python3 cli_dashboard.py"
    fi
    
    print_status "Local deployment started successfully!"
    print_status "Press Ctrl+C to stop the API server"
    
    # Wait for the API server
    wait $API_PID
}

# Function to stop Docker services
stop_docker() {
    print_status "Stopping Docker services..."
    
    if [ "$DOCKER_AVAILABLE" = false ]; then
        print_error "Docker is not available."
        return 1
    fi
    
    cd "$PROJECT_DIR"
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        print_status "Docker services stopped"
    else
        print_error "Docker compose file not found"
    fi
}

# Function to show status
show_status() {
    print_status "Checking system status..."
    
    # Check if services are running
    if [ "$DOCKER_AVAILABLE" = true ]; then
        cd "$PROJECT_DIR"
        if [ -f "$DOCKER_COMPOSE_FILE" ]; then
            print_status "Docker services status:"
            docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        fi
    fi
    
    # Check API health
    if command -v curl &> /dev/null; then
        print_status "API health check:"
        if curl -s http://localhost:8000/health > /dev/null; then
            print_status "API is running and healthy"
        else
            print_warning "API is not responding"
        fi
    fi
}

# Function to show logs
show_logs() {
    print_status "Showing logs..."
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        cd "$PROJECT_DIR"
        if [ -f "$DOCKER_COMPOSE_FILE" ]; then
            docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
        fi
    else
        print_warning "Docker not available. Check logs in $LOG_DIR"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start         Start the AI Forex Signal Generator"
    echo "  start-docker  Start with Docker (recommended)"
    echo "  start-local   Start locally with Python"
    echo "  stop          Stop Docker services"
    echo "  status        Show system status"
    echo "  logs          Show Docker logs"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start-docker    # Start with Docker"
    echo "  $0 start-local     # Start locally"
    echo "  $0 status          # Check status"
    echo "  $0 logs            # View logs"
}

# Main script logic
main() {
    print_header
    
    # Check prerequisites
    check_prerequisites
    
    # Parse command line arguments
    case "${1:-start}" in
        "start")
            if [ "$DOCKER_AVAILABLE" = true ]; then
                start_docker
            else
                start_local
            fi
            ;;
        "start-docker")
            start_docker
            ;;
        "start-local")
            start_local
            ;;
        "stop")
            stop_docker
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"