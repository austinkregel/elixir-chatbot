#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ChatBot — Production setup with model download from S3
#
# Sets up a production node from scratch: compiles the app, runs database
# migrations, downloads the latest trained models from S3-compatible storage,
# builds Phoenix assets, and starts the server.
#
# Unlike setup_and_deploy.sh, this script does NOT train models, download
# corpora, augment data, evaluate, or seed. It assumes models have already
# been trained and published to S3 via `mix train --publish`.
#
# Prerequisites:
#   - Elixir/Erlang installed
#   - PostgreSQL accessible (DATABASE_URL or POSTGRES_* in .env)
#   - S3 credentials configured in .env (MODEL_STORE_ENABLED=true)
#
# Usage:
#   ./scripts/setup_production.sh                # Full setup + start server
#   ./scripts/setup_production.sh --no-start     # Setup only, don't start
#   ./scripts/setup_production.sh --skip-models  # Skip model download
#   ./scripts/setup_production.sh --skip-assets  # Skip asset compilation
#   ./scripts/setup_production.sh --skip-migrate # Skip database migrations
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/lib/ui.sh"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/xla.sh"
source "$SCRIPT_DIR/lib/preflight.sh"

install_interrupt_trap

SKIP_MODELS=false
SKIP_ASSETS=false
SKIP_MIGRATE=false
NO_START=false

for arg in "$@"; do
  case "$arg" in
    --skip-models)  SKIP_MODELS=true ;;
    --skip-assets)  SKIP_ASSETS=true ;;
    --skip-migrate) SKIP_MIGRATE=true ;;
    --no-start)     NO_START=true ;;
    --help|-h)
      sed -n '3,27p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

resolve_root_dir
load_dotenv
detect_xla_target
configure_xla_cache

START_TIME=$SECONDS

banner "ChatBot — Production Setup"
echo "  XLA_TARGET:           ${XLA_TARGET}"
echo "  XLA_JIT_CACHE:        ${XLA_CACHE_DIR}"
echo "  MODEL_STORE_ENABLED:  ${MODEL_STORE_ENABLED:-false}"
echo "  S3_HOST:              ${S3_HOST:-not set}"
echo -e "${CYAN}============================================================${NC}"

# ---------------------------------------------------------------------------
step "Dependencies"
# ---------------------------------------------------------------------------
mix deps.get --only prod || fail "deps.get failed"
ok "Dependencies fetched"

# ---------------------------------------------------------------------------
step "Compile"
# ---------------------------------------------------------------------------
MIX_ENV=prod mix compile || fail "Compilation failed"
ok "Compiled (prod)"

# ---------------------------------------------------------------------------
if [ "$SKIP_MIGRATE" = false ]; then
  step "Database migrations"
  MIX_ENV=prod mix atlas.migrate && ok "Migrations applied" || {
    warn "atlas.migrate failed, trying ecto.migrate"
    MIX_ENV=prod mix ecto.migrate && ok "Migrations applied" || fail "Migrations failed"
  }
else
  step "Database migrations (skipped)"
  warn "Skipped via --skip-migrate"
fi

# ---------------------------------------------------------------------------
if [ "$SKIP_MODELS" = false ]; then
  step "Download models from S3"

  if [ "${MODEL_STORE_ENABLED:-false}" != "true" ]; then
    fail "MODEL_STORE_ENABLED is not set to 'true' in .env. Cannot download models."
  fi

  MIX_ENV=prod mix models.download --force && ok "Models downloaded" || fail "Model download failed"
else
  step "Model download (skipped)"
  warn "Skipped via --skip-models"
fi

# ---------------------------------------------------------------------------
if [ "$SKIP_ASSETS" = false ]; then
  step "Compile assets for production"
  (cd apps/chat_web && MIX_ENV=prod mix assets.deploy) && ok "Assets compiled" || fail "Asset build failed"
else
  step "Asset compilation (skipped)"
  warn "Skipped via --skip-assets"
fi

# ---------------------------------------------------------------------------
step "EXLA library preflight"
# ---------------------------------------------------------------------------
source_exla_preflight "$ROOT_DIR"

# ---------------------------------------------------------------------------
print_elapsed "$START_TIME" "Production setup complete"

if [ "$NO_START" = true ]; then
  ok "Server not started (--no-start)"
  exit 0
fi

# ---------------------------------------------------------------------------
step "Starting Phoenix server"
# ---------------------------------------------------------------------------
echo "  PHX_HOST: ${PHX_HOST:-localhost}"
echo "  PHX_PORT: ${PHX_PORT:-4000}"
echo ""
exec MIX_ENV=prod mix phx.server
