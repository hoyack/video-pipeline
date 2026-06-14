#!/usr/bin/env bash
set -euo pipefail

# start-stack.sh — bring the full vidpipe stack up safely (Supabase + app).
#
# Why this exists: on Docker Desktop + WSL2 the bind-mount cache under
# /run/desktop/mnt/host/wsl/docker-desktop-bind-mounts/ goes stale after a
# Docker Desktop or WSL restart. That causes two failure modes this script
# detects and repairs automatically:
#
#   1. Containers Exit(127) with "OCI runtime create failed ... not a directory:
#      Are you trying to mount a directory onto a file?" — a stale single-file
#      bind mount. A plain `docker restart` reuses the broken spec and loops;
#      the fix is to RECREATE (down + up) so Docker re-resolves the mount.
#
#   2. MinIO sees an EMPTY view of its shared ./volumes/storage drive, logs
#      "Formatting 1st pool", and serves a blank bucket (media 400/500) — even
#      though the real objects are still on disk (visible to supabase-storage,
#      which mounts the same host dir). The fix is to force-recreate just MinIO
#      so its /data mount re-resolves to the canonical host path.
#
# Supabase must always run with BOTH compose files. The base docker-compose.yml
# sets storage to STORAGE_BACKEND=file (which chokes on MinIO's directory-format
# objects → EISDIR/500) and does not define minio; docker-compose.s3.yml switches
# storage to s3 → http://minio:9000 and defines the minio service.
#
# Usage:
#   scripts/start-stack.sh            # bring everything up safely
#   scripts/start-stack.sh --build    # also rebuild the app images
#   SUPABASE_DIR=/path scripts/start-stack.sh
#
# Safe to run repeatedly — it is idempotent and never passes `down -v`, so
# database (./volumes/db/data) and object storage (./volumes/storage) are never
# wiped.

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPABASE_DIR="${SUPABASE_DIR:-$(cd "$APP_DIR/.." && pwd)/supabase-project}"
SUPABASE_FILES=(-f docker-compose.yml -f docker-compose.s3.yml)
BACKEND_PORT="${VIDPIPE_PORT:-8100}"
FRONTEND_PORT="${VIDPIPE_FRONTEND_PORT:-80}"
BUILD=""
[[ "${1:-}" == "--build" ]] && BUILD="--build"

# Load app .env so VIDPIPE_PORT / VIDPIPE_FRONTEND_PORT overrides apply.
if [[ -f "$APP_DIR/.env" ]]; then
    BACKEND_PORT="$(grep -oP '(?<=^VIDPIPE_PORT=).*' "$APP_DIR/.env" 2>/dev/null || echo "$BACKEND_PORT")"
    FRONTEND_PORT="$(grep -oP '(?<=^VIDPIPE_FRONTEND_PORT=).*' "$APP_DIR/.env" 2>/dev/null || echo "$FRONTEND_PORT")"
fi

sb() { (cd "$SUPABASE_DIR" && docker compose "${SUPABASE_FILES[@]}" "$@"); }

preflight() {
    command -v docker &>/dev/null || { error "Docker is not installed or not in PATH."; exit 1; }
    docker compose version &>/dev/null || { error "Docker Compose v2 plugin is not available."; exit 1; }
    [[ -d "$SUPABASE_DIR" ]] || { error "Supabase dir not found: $SUPABASE_DIR (set SUPABASE_DIR=...)."; exit 1; }
    [[ -f "$SUPABASE_DIR/docker-compose.yml" && -f "$SUPABASE_DIR/docker-compose.s3.yml" ]] || {
        error "Expected docker-compose.yml + docker-compose.s3.yml in $SUPABASE_DIR."; exit 1; }
    [[ -f "$APP_DIR/.env" ]] || warn ".env not found in $APP_DIR — backend may not start. Copy .env.example."
    ok "Pre-flight checks passed. Supabase: $SUPABASE_DIR"
}

# Repair failure mode #1: stale single-file bind mounts → Exit(127) "not a directory".
repair_stale_mounts() {
    local broken=""
    local id state
    while read -r id; do
        [[ -z "$id" ]] && continue
        state="$(docker inspect "$id" --format '{{.State.ExitCode}} {{.State.Error}}' 2>/dev/null || true)"
        if grep -qiE '^127 |not a directory|OCI runtime create' <<<"$state"; then
            broken+="$id "
        fi
    done < <(sb ps -aq 2>/dev/null || true)
    if [[ -n "$broken" ]]; then
        warn "Detected stale bind-mount failure (Exit 127). Recreating the Supabase stack..."
        sb down                       # never -v — bind-mounted data is preserved
        sb up -d
        ok "Supabase stack recreated."
    fi
}

# Repair failure mode #2: MinIO mounted a stale/empty view and formatted an empty pool.
repair_minio_divergence() {
    local minio_id storage_id minio_mb storage_mb
    minio_id="$(sb ps -q minio 2>/dev/null || true)"
    storage_id="$(sb ps -q storage 2>/dev/null || true)"
    [[ -z "$minio_id" || -z "$storage_id" ]] && return 0

    minio_mb="$(docker exec "$minio_id" du -sm /data 2>/dev/null | awk '{print $1}')"
    storage_mb="$(docker exec "$storage_id" du -sm /var/lib/storage 2>/dev/null | awk '{print $1}')"
    [[ -z "$minio_mb" || -z "$storage_mb" ]] && return 0

    # Both containers bind-mount the SAME host dir. In a healthy state the sizes
    # match. If supabase-storage sees materially more data than MinIO, MinIO's
    # mount is stale (it formatted an empty pool) while the real objects survive.
    if (( storage_mb > minio_mb + 5 )); then
        warn "MinIO sees ${minio_mb}MB but the real volume holds ${storage_mb}MB — stale mount / formatted pool."
        warn "Force-recreating MinIO so its /data mount re-resolves to the real objects..."
        sb up -d --force-recreate --no-deps minio
        sleep 8
        minio_mb="$(docker exec "$minio_id" du -sm /data 2>/dev/null | awk '{print $1}')" || true
        # Re-resolve id (force-recreate replaces the container).
        minio_id="$(sb ps -q minio 2>/dev/null || true)"
        minio_mb="$(docker exec "$minio_id" du -sm /data 2>/dev/null | awk '{print $1}')"
        if (( storage_mb > minio_mb + 5 )); then
            error "MinIO still does not see the data (${minio_mb}MB vs ${storage_mb}MB)."
            error "The Docker Desktop WSL bind-mount cache may need a full restart of Docker Desktop."
            error "Real objects are intact on the host at $SUPABASE_DIR/volumes/storage — they were NOT lost."
            return 1
        fi
        ok "MinIO re-attached to the real volume (${minio_mb}MB)."
    fi
}

verify_storage_backend() {
    local backend
    backend="$(docker exec "$(sb ps -q storage)" printenv STORAGE_BACKEND 2>/dev/null || true)"
    if [[ "$backend" != "s3" ]]; then
        error "supabase-storage is STORAGE_BACKEND=$backend (expected s3). The S3 overlay did not apply."
        exit 1
    fi
    ok "Storage backend is s3 → MinIO."
}

wait_healthy() {  # wait_healthy <container-id> <label> <timeout-s>
    local id="$1" label="$2" timeout="${3:-90}" waited=0 status
    [[ -z "$id" ]] && { warn "$label: container not found, skipping health wait."; return 0; }
    while (( waited < timeout )); do
        status="$(docker inspect "$id" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' 2>/dev/null || echo unknown)"
        [[ "$status" == "healthy" || "$status" == "running" ]] && { ok "$label is $status."; return 0; }
        sleep 3; waited=$((waited+3))
    done
    warn "$label did not become healthy within ${timeout}s (status: ${status:-unknown})."
}

main() {
    preflight

    info "Starting Supabase (with S3 overlay)..."
    sb up -d
    repair_stale_mounts
    wait_healthy "$(sb ps -q db)" "supabase-db" 120
    repair_minio_divergence
    verify_storage_backend

    info "Starting vidpipe app..."
    (cd "$APP_DIR" && docker compose up -d $BUILD)

    info "Waiting for backend to come up..."
    local waited=0
    until curl -fsS -o /dev/null "http://localhost:${BACKEND_PORT}/api/health" 2>/dev/null; do
        sleep 3; waited=$((waited+3))
        if (( waited >= 90 )); then
            error "Backend health check failed after 90s. Recent logs:"
            (cd "$APP_DIR" && docker compose logs --tail 30 backend) || true
            exit 1
        fi
    done
    ok "Backend healthy: http://localhost:${BACKEND_PORT}/api/health"

    if curl -fsS -o /dev/null "http://localhost:${FRONTEND_PORT}/" 2>/dev/null; then
        ok "Frontend reachable: http://localhost:${FRONTEND_PORT}/"
    else
        warn "Frontend not reachable on port ${FRONTEND_PORT} yet."
    fi

    echo ""
    ok "Stack is up."
    info "Frontend: http://localhost:${FRONTEND_PORT}/"
    info "Backend:  http://localhost:${BACKEND_PORT}/  (API under /api)"
    info "Note: supabase-storage may report (unhealthy) — its healthcheck probes IPv6 localhost"
    info "      while it serves on IPv4. It is cosmetic; real traffic via storage:5000 works."
}

main "$@"
