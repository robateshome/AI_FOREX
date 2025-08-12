#!/bin/bash

# Forex Trading Bot Startup Script
# Module 0x06: Docker Deployment Startup Script
# CRC32: PLACEHOLDER_CRC32_06_STARTUP

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="forex-trading-bot"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "All prerequisites are satisfied."
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            print_warning ".env file not found. Creating from .env.example..."
            cp .env.example .env
            print_warning "Please edit .env file with your configuration before starting."
            read -p "Press Enter to continue after editing .env file..."
        else
            print_error ".env.example file not found. Cannot create .env file."
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p data logs config monitoring/prometheus monitoring/grafana/provisioning
    
    print_success "Environment setup completed."
}

# Function to build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Build and start all services
    docker-compose -f "$COMPOSE_FILE" up --build -d
    
    print_success "Services started successfully."
}

# Function to check service health
check_health() {
    print_status "Checking service health..."
    
    # Wait for services to be ready
    sleep 10
    
    # Check backend health
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "Backend is healthy"
    else
        print_warning "Backend health check failed"
    fi
    
    # Check frontend health
    if curl -f http://localhost/health &> /dev/null; then
        print_success "Frontend is healthy"
    else
        print_warning "Frontend health check failed"
    fi
    
    # Check database health
    if docker-compose -f "$COMPOSE_FILE" exec -T database pg_isready -U forex_user -d forex_bot &> /dev/null; then
        print_success "Database is healthy"
    else
        print_warning "Database health check failed"
    fi
}

# Function to show service status
show_status() {
    print_status "Service status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    print_status "Service logs (last 10 lines):"
    docker-compose -f "$COMPOSE_FILE" logs --tail=10
}

# Function to show access information
show_access_info() {
    echo ""
    print_success "Forex Trading Bot is now running!"
    echo ""
    echo "Access URLs:"
    echo "  Frontend Dashboard: http://localhost"
    echo "  Backend API:        http://localhost:8000"
    echo "  API Documentation:  http://localhost:8000/docs"
    echo "  Grafana:            http://localhost:3000 (admin/grafana_password_secure)"
    echo "  Prometheus:         http://localhost:9090"
    echo ""
    echo "Default credentials:"
    echo "  Database:           forex_user / forex_password_secure"
    echo "  Redis:              (no auth) / redis_password_secure"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop:      docker-compose down"
    echo "To restart:   docker-compose restart"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose -f "$COMPOSE_FILE" down
    print_success "Services stopped successfully."
}

# Function to restart services
restart_services() {
    print_status "Restarting services..."
    docker-compose -f "$COMPOSE_FILE" restart
    print_success "Services restarted successfully."
}

# Function to show logs
show_logs() {
    print_status "Showing service logs..."
    docker-compose -f "$COMPOSE_FILE" logs -f
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down -v
    
    # Remove images
    docker-compose -f "$COMPOSE_FILE" down --rmi all
    
    # Remove volumes
    docker volume prune -f
    
    print_success "Cleanup completed."
}

# Function to show help
show_help() {
    echo "Forex Trading Bot Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Build and start all services"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show service status"
    echo "  logs      Show service logs"
    echo "  cleanup   Stop services and clean up resources"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start      # Start the system"
    echo "  $0 status     # Check service status"
    echo "  $0 logs       # View logs"
    echo "  $0 stop       # Stop the system"
}

# Main script logic
main() {
    case "${1:-start}" in
        "start")
            check_prerequisites
            setup_environment
            start_services
            check_health
            show_access_info
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            check_health
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Trap to handle script interruption
trap 'print_warning "Script interrupted. Stopping services..."; stop_services; exit 1' INT TERM

# Run main function with all arguments
main "$@"