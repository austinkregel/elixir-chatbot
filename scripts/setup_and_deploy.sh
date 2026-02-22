#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ChatBot — Full setup, train, evaluate, and build pipeline
#
# Brings the system from a fresh clone (or dirty state) to a fully trained,
# evaluated, and asset-compiled deployment-ready state.
#
# Prerequisites:
#   - Elixir/Erlang, Node (for assets), gcc/g++ (for tree-sitter)
#   - PostgreSQL running (e.g. docker-compose up -d db) for atlas.setup
#   - Optional: .env in project root (POSTGRES_*, DATABASE_URL, XLA_TARGET, etc.)
#
# Usage (from repo root or any directory):
#   ./scripts/setup_and_deploy.sh                  # Run everything
#   ./scripts/setup_and_deploy.sh --skip-training  # Skip model training
#   ./scripts/setup_and_deploy.sh --skip-eval      # Skip evaluation
#   ./scripts/setup_and_deploy.sh --quick          # Quick LSTM training
# =============================================================================

SKIP_TRAINING=false
SKIP_EVAL=false
TRAIN_FLAGS=""

for arg in "$@"; do
  case "$arg" in
    --skip-training) SKIP_TRAINING=true ;;
    --skip-eval)     SKIP_EVAL=true ;;
    --quick)         TRAIN_FLAGS="--quick" ;;
    --help|-h)
      sed -n '3,21p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# Project root and .env
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR" || exit 1

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.env"
  set +a
fi

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step=0
step() {
  step=$((step + 1))
  echo ""
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Step ${step}: $1${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; exit 1; }

START_TIME=$SECONDS

# ---------------------------------------------------------------------------
step "Dependencies"
# ---------------------------------------------------------------------------
mix deps.get || fail "deps.get failed"
ok "Dependencies fetched"

# ---------------------------------------------------------------------------
step "Compile"
# ---------------------------------------------------------------------------
MIX_ENV=prod mix compile --force --warnings-as-errors 2>&1 | tail -5 && \
  ok "Compiled (prod)" || fail "Compilation failed"

# ---------------------------------------------------------------------------
step "Database — Atlas setup & migrations"
# ---------------------------------------------------------------------------
MIX_ENV=prod mix atlas.setup 2>&1 && ok "Atlas database created & migrated" || {
  warn "atlas.setup failed (DB may already exist), running migrations only"
  MIX_ENV=prod mix atlas.migrate && ok "Migrations applied" || fail "Migrations failed"
}

# ---------------------------------------------------------------------------
step "Seed baseline data (facts, beliefs, worlds)"
# ---------------------------------------------------------------------------
MIX_ENV=prod mix atlas.seed 2>&1 && ok "Seeds applied" || warn "Seeding had issues (non-fatal)"

# ---------------------------------------------------------------------------
step "Download training corpora"
# ---------------------------------------------------------------------------
MIX_ENV=prod mix download_speech_act_corpus 2>&1 && ok "Speech act corpus ready"
MIX_ENV=prod mix download_sentiment_corpus  2>&1 && ok "Sentiment corpus ready"

# --- Training ---
if [ "$SKIP_TRAINING" = false ]; then
  step "Train all models"
  MIX_ENV=prod mix train $TRAIN_FLAGS
  train_exit=$?
  if [ "$train_exit" -eq 0 ]; then
    ok "Training complete"
  else
    fail "Training failed"
  fi
else
  step "Training (skipped)"
  warn "Skipped via --skip-training"
fi

# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = false ]; then
  step "Evaluate models against gold standard"

  echo ""
  echo "  Intent..."
  MIX_ENV=prod mix evaluate.intent --save 2>&1 && ok "Intent evaluation saved" || fail "Intent evaluation failed"

  echo ""
  echo "  Sentiment..."
  MIX_ENV=prod mix evaluate.sentiment --save 2>&1 && ok "Sentiment evaluation saved" || fail "Sentiment evaluation failed"

  echo ""
  echo "  Speech act..."
  MIX_ENV=prod mix evaluate.speech_act --save 2>&1 && ok "Speech act evaluation saved" || fail "Speech act evaluation failed"

  echo ""
  echo "  NER..."
  MIX_ENV=prod mix evaluate.ner --save 2>&1 && ok "NER evaluation saved" || fail "NER evaluation failed"
else
  step "Evaluation (skipped)"
  warn "Skipped via --skip-eval"
fi

# ---------------------------------------------------------------------------
step "Compile assets for production"
# ---------------------------------------------------------------------------
(cd apps/chat_web && MIX_ENV=prod mix assets.deploy) && ok "Assets compiled and digested" || fail "Asset build failed"

# ---------------------------------------------------------------------------
step "Tree-sitter grammars (make)"
# ---------------------------------------------------------------------------
make && ok "Grammars compiled" || warn "Grammar compilation had issues (non-fatal)"

# ---------------------------------------------------------------------------
ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  All done in ${MINS}m ${SECS}s${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
