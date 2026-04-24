#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ChatBot — Smart Development Setup
# =============================================================================
# Auto-detects hardware, scans for missing artifacts, and runs only the
# setup steps needed. Works for both local development and Docker-based setups.
#
# Prerequisites:
#   - Elixir >= 1.18, Erlang/OTP >= 27 (or Docker)
#   - PostgreSQL accessible (local or via docker-compose db service)
#
# Usage:
#   ./scripts/setup.sh                   # Standard local dev setup
#   ./scripts/setup.sh --check           # Dry-run: detect hardware + report missing artifacts
#   ./scripts/setup.sh --docker          # Setup via Docker Compose (auto-detects GPU)
#   ./scripts/setup.sh --skip-training   # Skip model training
#   ./scripts/setup.sh --full-training   # Full training (no --quick)
#   ./scripts/setup.sh --all             # Enable all optional steps
#   ./scripts/setup.sh --help            # Show all options
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/lib/ui.sh"
source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/xla.sh"
source "$SCRIPT_DIR/lib/preflight.sh"
source "$SCRIPT_DIR/lib/artifacts.sh"
source "$SCRIPT_DIR/lib/docker.sh"

# ---------------------------------------------------------------------------
# CLI flag defaults
# ---------------------------------------------------------------------------
MODE="local"
SKIP_TRAINING=false
FULL_TRAINING=false
SKIP_DB=false
WITH_PYTHON=false
WITH_GRAMMARS=false
WITH_OURO=false
WITH_WORDNET=false
CHECK_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --docker)         MODE="docker" ;;
    --skip-training)  SKIP_TRAINING=true ;;
    --full-training)  FULL_TRAINING=true ;;
    --skip-db)        SKIP_DB=true ;;
    --with-python)    WITH_PYTHON=true ;;
    --with-grammars)  WITH_GRAMMARS=true ;;
    --with-ouro)      WITH_OURO=true ;;
    --with-wordnet)   WITH_WORDNET=true ;;
    --all)
      WITH_PYTHON=true
      WITH_GRAMMARS=true
      WITH_OURO=true
      WITH_WORDNET=true
      ;;
    --check)          CHECK_ONLY=true ;;
    --help|-h)
      sed -n '3,16p' "$0"
      echo ""
      echo "Options:"
      echo "  --docker          Setup via Docker Compose (auto-detects GPU config)"
      echo "  --skip-training   Skip model training"
      echo "  --full-training   Full training pipeline (no --quick shortcut)"
      echo "  --skip-db         Skip database create/migrate/seed"
      echo "  --with-python     Set up Python venv and generate data files"
      echo "  --with-grammars   Compile tree-sitter grammars"
      echo "  --with-ouro       Download Ouro model files"
      echo "  --with-wordnet    Download WordNet Prolog files"
      echo "  --all             Enable all optional steps"
      echo "  --check           Dry-run: detect hardware and report artifacts only"
      echo "  --help, -h        Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
resolve_root_dir
install_interrupt_trap
load_dotenv

START_TIME=$SECONDS

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
detect_xla_target
detect_docker_compose_files "$ROOT_DIR"
suggest_gpu_dockerfile "$ROOT_DIR"

banner "ChatBot — Setup"
echo "  XLA_TARGET:           ${XLA_TARGET}"
echo "  Hardware:             ${_XLA_HW_NOTE:-n/a}"
echo "  Docker GPU:           ${_DOCKER_GPU:-cpu}"
echo "  Docker Compose:       ${COMPOSE_FILES}"
echo "  Suggested Dockerfile: ${SUGGESTED_DOCKERFILE}"
echo -e "${CYAN}============================================================${NC}"

# ---------------------------------------------------------------------------
# Artifact scan
# ---------------------------------------------------------------------------
step "Scanning for artifacts"
print_artifact_report "$ROOT_DIR"

if [ "$CHECK_ONLY" = true ]; then
  echo ""
  info "Dry-run complete (--check). No changes were made."
  exit 0
fi

# ---------------------------------------------------------------------------
# Docker path
# ---------------------------------------------------------------------------
if [ "$MODE" = "docker" ]; then
  step "Docker Compose setup"

  if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    ok "Created .env from .env.example"
    sed -i.bak "s/^XLA_TARGET=.*/XLA_TARGET=${XLA_TARGET}/" "$ROOT_DIR/.env" && rm -f "$ROOT_DIR/.env.bak"
    sed -i.bak "s/^GPU_DOCKERFILE=.*/GPU_DOCKERFILE=${SUGGESTED_DOCKERFILE}/" "$ROOT_DIR/.env" && rm -f "$ROOT_DIR/.env.bak"
    ok "Patched .env with detected hardware (XLA_TARGET=${XLA_TARGET}, GPU_DOCKERFILE=${SUGGESTED_DOCKERFILE})"
  fi

  info "Starting database..."
  docker compose ${COMPOSE_FILES} up -d db
  ok "Database container started"

  info "Waiting for database to become healthy..."
  local_timeout=60
  elapsed_wait=0
  while ! docker compose ${COMPOSE_FILES} exec db pg_isready -q 2>/dev/null; do
    sleep 2
    elapsed_wait=$((elapsed_wait + 2))
    if [ "$elapsed_wait" -ge "$local_timeout" ]; then
      fail "Database did not become ready within ${local_timeout}s"
    fi
  done
  ok "Database is healthy"

  info "Building and running app container..."
  docker compose ${COMPOSE_FILES} build app
  docker compose ${COMPOSE_FILES} run --rm app mix setup
  ok "Docker setup complete"

  print_elapsed "$START_TIME" "Docker setup complete"
  echo "  Start the app with:"
  echo "    docker compose ${COMPOSE_FILES} up"
  exit 0
fi

# ===========================================================================
# Local setup path
# ===========================================================================

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
step "Checking prerequisites"

_prereq_ok=true

if command -v elixir &>/dev/null; then
  elixir_version=$(elixir --version 2>/dev/null | grep "Elixir" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  elixir_major=$(echo "$elixir_version" | cut -d. -f1)
  elixir_minor=$(echo "$elixir_version" | cut -d. -f2)
  if [ "${elixir_major:-0}" -lt 1 ] || ([ "${elixir_major:-0}" -eq 1 ] && [ "${elixir_minor:-0}" -lt 18 ]); then
    warn "Elixir ${elixir_version} found, but >= 1.18 is required"
    _prereq_ok=false
  else
    ok "Elixir ${elixir_version}"
  fi
else
  warn "Elixir not found"
  _prereq_ok=false
fi

if command -v erl &>/dev/null; then
  otp_version=$(erl -eval 'io:format("~s", [erlang:system_info(otp_release)]), halt().' -noshell 2>/dev/null || echo "0")
  if [ "${otp_version:-0}" -lt 27 ]; then
    warn "Erlang/OTP ${otp_version} found, but >= 27 is required"
    _prereq_ok=false
  else
    ok "Erlang/OTP ${otp_version}"
  fi
else
  warn "Erlang not found"
  _prereq_ok=false
fi

if command -v psql &>/dev/null; then
  ok "PostgreSQL client (psql)"
else
  warn "psql not found — database setup may rely on Docker (docker compose up -d db)"
fi

if command -v asdf &>/dev/null; then
  ok "asdf version manager"
else
  info "asdf not found — install it to use .tool-versions for version management"
fi

if command -v python3 &>/dev/null; then
  py_version=$(python3 --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
  ok "Python ${py_version} (optional, for data scripts)"
else
  if [ "$WITH_PYTHON" = true ]; then
    warn "Python 3 not found but --with-python was requested"
    _prereq_ok=false
  else
    info "Python 3 not found (optional, use --with-python to enable data scripts)"
  fi
fi

if command -v gcc &>/dev/null; then
  ok "gcc (optional, for tree-sitter grammars)"
else
  if [ "$WITH_GRAMMARS" = true ]; then
    warn "gcc not found but --with-grammars was requested"
    _prereq_ok=false
  else
    info "gcc not found (optional, use --with-grammars for tree-sitter)"
  fi
fi

if [ "$_prereq_ok" = false ]; then
  echo ""
  warn "Some prerequisites are missing. The setup may fail at later steps."
  echo "  Install missing tools or use --docker for a containerized setup."
  echo ""
fi

# ---------------------------------------------------------------------------
# Environment file
# ---------------------------------------------------------------------------
step "Environment file"

if [ ! -f "$ROOT_DIR/.env" ]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  ok "Created .env from .env.example"

  if [ "$(uname -s)" = "Darwin" ]; then
    sed -i '' "s/^XLA_TARGET=.*/XLA_TARGET=${XLA_TARGET}/" "$ROOT_DIR/.env"
  else
    sed -i "s/^XLA_TARGET=.*/XLA_TARGET=${XLA_TARGET}/" "$ROOT_DIR/.env"
  fi
  ok "Patched XLA_TARGET=${XLA_TARGET} in .env"
  info "Review .env and adjust POSTGRES_*, S3, and other settings as needed"

  load_dotenv
else
  ok ".env already exists"
fi

# ---------------------------------------------------------------------------
# Hex and Rebar
# ---------------------------------------------------------------------------
step "Hex and Rebar"
mix local.hex --force --if-missing
mix local.rebar --force --if-missing
ok "Hex and Rebar installed"

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
step "Fetching dependencies"
mix deps.get || fail "mix deps.get failed"
ok "Dependencies fetched"

# ---------------------------------------------------------------------------
# EXLA preflight
# ---------------------------------------------------------------------------
step "EXLA library preflight"
source_exla_preflight "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
step "Compiling application"
mix compile || fail "Compilation failed"
ok "Application compiled"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
if [ "$SKIP_DB" = false ]; then
  step "Database setup"
  mix atlas.setup 2>&1 && ok "Database created and migrated" || {
    warn "atlas.setup failed (database may already exist), running migrations"
    mix atlas.migrate && ok "Migrations applied" || fail "Database migrations failed"
  }
  mix atlas.seed 2>&1 && ok "Seeds applied" || warn "Seeding had issues (non-fatal)"
else
  step "Database setup (skipped)"
  warn "Skipped via --skip-db"
fi

# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
check_corpora "$ROOT_DIR"
if [ ${#MISSING_CORPORA[@]} -gt 0 ]; then
  step "Downloading missing training corpora"

  for corpus in "${MISSING_CORPORA[@]}"; do
    case "$corpus" in
      speech_act/*)
        mix download_speech_act_corpus 2>&1 && ok "Speech act corpus downloaded" || fail "Speech act corpus download failed"
        ;;
      sentiment/*)
        mix download_sentiment_corpus 2>&1 && ok "Sentiment corpus downloaded" || fail "Sentiment corpus download failed"
        ;;
    esac
  done
else
  step "Training corpora"
  ok "All corpora already present"
fi

# ---------------------------------------------------------------------------
# Framing corpus (GVFC)
# ---------------------------------------------------------------------------
step "Framing corpus (GVFC)"
if [ ! -f "$ROOT_DIR/data/framing/GVFC_extension_multimodal.csv" ]; then
  if [ -f "$ROOT_DIR/GVFC.zip" ]; then
    mix ingest_framing_corpus && ok "GVFC corpus extracted" || fail "GVFC corpus extraction failed"
  else
    fail "GVFC corpus missing. Download GVFC.zip from https://github.com/ganggit/GVFC-raw-corpus (Google Drive link) and place it in the project root, then rerun setup."
  fi
else
  ok "GVFC corpus already present"
fi

# ---------------------------------------------------------------------------
# Lexicon setup
# ---------------------------------------------------------------------------
step "Lexicon ingestion"
mix setup_lexicon && ok "Lexicon ingested" || fail "Lexicon ingestion failed"

# ---------------------------------------------------------------------------
# Data generation (micro-classifiers + framing)
# ---------------------------------------------------------------------------
step "Generating classifier training data"
mix gen_micro_data && ok "Micro-classifier data generated" || fail "Micro-classifier data generation failed"
mix gen_framing_data --corpus gvfc && ok "Framing data generated" || fail "Framing data generation failed"

# ---------------------------------------------------------------------------
# WordNet
# ---------------------------------------------------------------------------
if [ "$WITH_WORDNET" = true ]; then
  check_wordnet "$ROOT_DIR"
  if [ ${#MISSING_WORDNET[@]} -gt 0 ]; then
    step "Downloading WordNet"
    mix download_wordnet 2>&1 && ok "WordNet downloaded" || warn "WordNet download failed (non-fatal)"
  else
    step "WordNet"
    ok "WordNet already present"
  fi
fi

# ---------------------------------------------------------------------------
# Python data generation
# ---------------------------------------------------------------------------
if [ "$WITH_PYTHON" = true ]; then
  step "Python data generation"

  if [ ! -d "$ROOT_DIR/.venv" ]; then
    python3 -m venv "$ROOT_DIR/.venv"
    ok "Created Python venv at .venv"
  else
    ok "Python venv already exists"
  fi

  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"

  if [ -f "$ROOT_DIR/scripts/requirements.txt" ]; then
    pip install -q -r "$ROOT_DIR/scripts/requirements.txt"
    ok "Python dependencies installed"
  fi

  check_python_generated "$ROOT_DIR"
  if [ ${#MISSING_PYTHON_DATA[@]} -gt 0 ]; then
    for item in "${MISSING_PYTHON_DATA[@]}"; do
      case "$item" in
        *informal_expansions*)
          python "$ROOT_DIR/scripts/generate_informal_expansions.py" && ok "Generated informal_expansions.json" || warn "generate_informal_expansions.py failed (non-fatal)"
          ;;
        *seeded_heuristics*)
          python "$ROOT_DIR/scripts/generate_heuristics.py" && ok "Generated seeded_heuristics.json" || warn "generate_heuristics.py failed (non-fatal)"
          ;;
      esac
    done
  else
    ok "Python-generated data already present"
  fi

  deactivate 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAINING" = false ]; then
  check_term_models "$ROOT_DIR"
  total_missing=$(( ${#MISSING_MODELS[@]} + ${#MISSING_MICRO_MODELS[@]} ))

  if [ "$total_missing" -gt 0 ] || [ "$FULL_TRAINING" = true ]; then
    step "Training models (all 6 stages)"

    if [ "$FULL_TRAINING" = true ]; then
      info "Full training pipeline (this may take a while)..."
      mix train && ok "Training complete" || fail "Training failed"
    else
      info "Quick training (use --full-training for the complete pipeline)..."
      mix train --quick && ok "Quick training complete" || fail "Training failed"
    fi
  else
    step "Model training"
    ok "All models already present (skipping training)"
  fi
else
  step "Model training (skipped)"
  warn "Skipped via --skip-training"
fi

# ---------------------------------------------------------------------------
# Ouro
# ---------------------------------------------------------------------------
if [ "$WITH_OURO" = true ]; then
  check_ouro "$ROOT_DIR"
  if [ ${#MISSING_OURO[@]} -gt 0 ]; then
    step "Downloading Ouro model"
    mix ouro.download && ok "Ouro model downloaded" || warn "Ouro download failed (non-fatal)"
  else
    step "Ouro model"
    ok "Ouro model already present"
  fi
fi

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
step "Phoenix assets"
(cd apps/chat_web && mix assets.setup) && ok "Assets set up" || fail "Asset setup failed"

# ---------------------------------------------------------------------------
# Tree-sitter grammars
# ---------------------------------------------------------------------------
if [ "$WITH_GRAMMARS" = true ]; then
  check_grammars "$ROOT_DIR"
  if [ ${#MISSING_GRAMMARS[@]} -gt 0 ]; then
    step "Compiling tree-sitter grammars"
    make grammars && ok "Grammars compiled" || warn "Grammar compilation failed (non-fatal)"
  else
    step "Tree-sitter grammars"
    ok "Grammars already present"
  fi
fi

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
step "Final artifact report"
print_artifact_report "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_elapsed "$START_TIME" "Setup complete"
echo "  Start the app with:"
echo "    mix phx.server"
echo ""
echo "  Or with an interactive shell:"
echo "    iex -S mix phx.server"
echo ""
echo "  Open http://localhost:${PHX_PORT:-4000}/chat"
echo ""
