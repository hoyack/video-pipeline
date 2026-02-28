#!/usr/bin/env bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }

# ---- Pre-flight checks ----
preflight() {
    local failed=0

    if ! command -v docker &>/dev/null; then
        error "Docker is not installed or not in PATH."
        failed=1
    fi

    if ! docker compose version &>/dev/null; then
        error "Docker Compose (v2 plugin) is not available."
        failed=1
    fi

    if [[ ! -f .env ]]; then
        warn ".env file not found. Copy .env.example and configure it:"
        warn "  cp .env.example .env"
        failed=1
    fi

    if [[ -f .env ]]; then
        local creds
        creds=$(grep -oP '(?<=GOOGLE_APPLICATION_CREDENTIALS=).*' .env 2>/dev/null || true)
        if [[ -n "$creds" && ! -f "$creds" ]]; then
            warn "GOOGLE_APPLICATION_CREDENTIALS points to '$creds' but file does not exist."
            warn "GCP credential mount will fail at runtime."
        fi
    fi

    if [[ $failed -ne 0 ]]; then
        error "Pre-flight checks failed. Fix the issues above and try again."
        exit 1
    fi

    ok "Pre-flight checks passed."
}

# ---- Actions ----
start_production() {
    info "Starting production environment..."
    docker compose -f docker-compose.yml up --build -d
    ok "Production is running."
    info "Frontend: http://localhost:${VIDPIPE_FRONTEND_PORT:-80}"
    info "Backend:  http://localhost:${VIDPIPE_PORT:-8000}"
}

start_development() {
    info "Starting development environment..."
    docker compose -f docker-compose.dev.yml up --build -d
    ok "Development is running."
    info "Frontend: http://localhost:${VIDPIPE_FRONTEND_PORT:-5173}"
    info "Backend:  http://localhost:${VIDPIPE_PORT:-8000}"
}

stop_services() {
    info "Stopping all services..."
    docker compose -f docker-compose.yml down 2>/dev/null || true
    docker compose -f docker-compose.dev.yml down 2>/dev/null || true
    ok "All services stopped."
}

stop_and_clean() {
    warn "This will stop all services AND delete volumes (database + artifacts)."
    read -rp "Are you sure? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        docker compose -f docker-compose.yml down -v 2>/dev/null || true
        docker compose -f docker-compose.dev.yml down -v 2>/dev/null || true
        ok "Services stopped and volumes removed."
    else
        info "Cancelled."
    fi
}

# ---- Menu ----
show_menu() {
    echo ""
    echo -e "${CYAN}vidpipe Docker Deployment${NC}"
    echo "========================="
    echo "1) Production   - Build optimized images, serve via nginx"
    echo "2) Development  - Hot-reload, source-mounted volumes"
    echo "3) Stop         - Stop all running services"
    echo "4) Stop + Clean - Stop services and remove volumes"
    echo "5) Exit"
    echo ""
}

main() {
    preflight

    while true; do
        show_menu
        read -rp "Select an option [1-5]: " choice
        case "$choice" in
            1) start_production ;;
            2) start_development ;;
            3) stop_services ;;
            4) stop_and_clean ;;
            5) info "Bye!"; exit 0 ;;
            *) warn "Invalid option. Please select 1-5." ;;
        esac
    done
}

main "$@"
