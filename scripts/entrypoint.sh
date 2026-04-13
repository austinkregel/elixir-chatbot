#!/usr/bin/env bash
# =============================================================================
# Unified Container Entrypoint
# =============================================================================
# Handles dependency fetching, compilation, and startup for both training
# containers (ROCm/CUDA) and the app container (CPU).
#
# Deps and _build are expected to live on persistent Docker volumes so that
# compilation only happens once per platform. Subsequent starts recompile
# only changed application code.
#
# Environment variables:
#   MIX_ENV       - Elixir environment (default: prod)
#   XLA_BUILD     - Set to "true" for ROCm source builds of XLA
#   XLA_TARGET    - EXLA backend target (rocm, cuda, cpu)
#   BAZEL_JOBS    - Max parallel Bazel actions (default: all CPUs)
#   TRAIN_TASKS   - Space-separated mix tasks (training containers only)
#   TRAIN_FLAGS   - Extra flags for training tasks
# =============================================================================
set -euo pipefail

cd /app

# Source shared preflight wrapper if available (container path)
if [ -f /app/scripts/lib/preflight.sh ]; then
  source /app/scripts/lib/ui.sh
  source /app/scripts/lib/preflight.sh
  _HAS_LIB=true
else
  _HAS_LIB=false
fi

# Raise BEAM VM process limit (default 262K is too low for concurrent EXLA/NLP workloads)
export ELIXIR_ERL_OPTIONS="${ELIXIR_ERL_OPTIONS:-} +P 1048576"

# Auto-detect available CPUs and set Bazel parallelism.
CPUS=$(nproc 2>/dev/null || echo 2)
JOBS="${BAZEL_JOBS:-$CPUS}"
export BUILD_FLAGS="${BUILD_FLAGS:-} --jobs=${JOBS}"

# XLA persistent compilation cache -- JIT-compiled GPU kernels are cached
# to disk so subsequent runs skip recompilation entirely.
# Only set GPU-specific flags when a GPU target is active.
if [ "${XLA_TARGET:-cpu}" != "cpu" ]; then
  XLA_CACHE_DIR="/root/.cache/xla_jit"
  mkdir -p "$XLA_CACHE_DIR"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=2"
  export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
fi

echo "============================================================"
echo "  Container Startup"
echo "============================================================"
echo "  MIX_ENV:              ${MIX_ENV:-prod}"
echo "  XLA_TARGET:           ${XLA_TARGET:-cpu}"
echo "  XLA_BUILD:            ${XLA_BUILD:-false}"
echo "  BAZEL_JOBS:           ${JOBS} (of ${CPUS} CPUs)"
echo "  XLA_JIT_CACHE:        ${XLA_CACHE_DIR:-n/a}"
echo "  MODEL_STORE_ENABLED:  ${MODEL_STORE_ENABLED:-false}"
echo "  SKIP_MODEL_DOWNLOAD:  ${SKIP_MODEL_DOWNLOAD:-false}"
echo "  S3_HOST:              ${S3_HOST:-localhost}"
echo "============================================================"
echo ""

# -- Hex + Rebar --
mix local.hex --force --if-missing
mix local.rebar --force --if-missing

# -- Fetch dependencies --
echo ">>> mix deps.get"
mix deps.get

# -- Verify EXLA shared libraries --
echo ">>> Checking EXLA library dependencies..."
if [ "$_HAS_LIB" = true ]; then
  ROOT_DIR=/app
  source_exla_preflight /app
else
  PREFLIGHT="/app/scripts/exla_preflight.sh"
  if [ -f "$PREFLIGHT" ]; then
    source "$PREFLIGHT" /app/_build
  else
    echo "[preflight] exla_preflight.sh not found, skipping"
  fi
fi

# -- Compile dependencies --
# For ROCm XLA source builds, the first run needs --force on xla and exla
# to trigger a from-source build rather than attempting a precompiled download.
# A marker file in _build/ tracks whether this has already been done.
if [ "${XLA_BUILD:-}" = "true" ] && [ ! -f /app/_build/.xla_built ]; then
  echo ">>> First-time XLA source build (this will take a while)..."
  mix deps.compile
  mkdir -p /app/_build && touch /app/_build/.xla_built
else
  echo ">>> mix deps.compile"
  mix deps.compile
fi

# -- Compile application --
# Force recompilation of app code to prevent stale .beam files when
# source is bind-mounted but _build lives on a persistent Docker volume.
echo ">>> mix compile --force (app code only)"
mix compile --force


# -- Database setup --
if [ "${MIX_ENV:-prod}" = "test" ]; then
  echo ">>> Test env: create database (migrations handled by test suite)"
  mix ecto.drop 2>/dev/null || true
  mix ecto.create
else
  echo ">>> mix ecto.setup (create + migrate)"
  mix ecto.setup 2>/dev/null
  mix ecto.migrate
fi

# -- Model download from S3 --
# Detect if the CMD will produce its own models (training scripts)
_CMD_STR="$*"
_IS_TRAINING=false
if [ -n "${TRAIN_TASKS:-}" ]; then
  _IS_TRAINING=true
elif echo "$_CMD_STR" | grep -q "setup_and_deploy"; then
  _IS_TRAINING=true
fi

if [ "${MODEL_STORE_ENABLED:-false}" = "true" ]; then
  if [ "${SKIP_MODEL_DOWNLOAD:-false}" = "true" ]; then
    echo ">>> Model download skipped (SKIP_MODEL_DOWNLOAD=true)"
  elif [ "$_IS_TRAINING" = true ]; then
    echo ">>> Model download skipped (training session will produce its own models)"
  else
    echo ">>> Downloading models from S3..."
    mix models.download || echo "[warning] Model download failed (non-fatal)"
  fi
fi

echo ""
echo "============================================================"
echo "  Startup complete — launching command"
echo "============================================================"
echo ""

# If TRAIN_TASKS is set, run each task individually then exit
if [ -n "${TRAIN_TASKS:-}" ]; then
  for task in $TRAIN_TASKS; do
    echo ">>> Running: mix ${task} --publish ${TRAIN_FLAGS:-}"
    mix "${task}" --publish ${TRAIN_FLAGS:-}
    echo ""
  done
  echo "============================================================"
  echo "  Training complete"
  echo "============================================================"
  exit 0
fi


# Delegate to CMD (e.g. "mix phx.server", "mix test", etc.)
exec "$@"
