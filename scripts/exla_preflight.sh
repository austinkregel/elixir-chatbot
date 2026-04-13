#!/usr/bin/env bash
# =============================================================================
# EXLA Library Preflight Check
# =============================================================================
# Discovers and exports LD_LIBRARY_PATH for EXLA's hermetic libraries.
# Works for both CUDA and ROCm builds -- uses ldd to find whatever is
# missing regardless of library names, then searches platform-appropriate paths.
#
# Usage:
#   source scripts/exla_preflight.sh [build_dir]
#
# Can be sourced from entrypoint.sh (containers) or shell profile (local dev).
# =============================================================================

BUILD_DIR="${1:-_build}"
LIBEXLA=$(find "$BUILD_DIR" -name "libexla.so" 2>/dev/null | head -1)

# Also check deps/exla/cache if not found in _build
if [ -z "$LIBEXLA" ]; then
  LIBEXLA=$(find "$(dirname "$BUILD_DIR")/deps/exla/cache" -name "libexla.so" 2>/dev/null | head -1)
fi

if [ -z "$LIBEXLA" ]; then
  echo "[preflight] libexla.so not found, skipping library check"
  return 0 2>/dev/null || exit 0
fi

echo "[preflight] Checking libraries for: $LIBEXLA"

MISSING=$(ldd "$LIBEXLA" 2>/dev/null | grep "not found" | awk '{print $1}' || true)

if [ -z "$MISSING" ]; then
  echo "[preflight] All shared libraries resolved"
  return 0 2>/dev/null || exit 0
fi

echo "[preflight] Missing libraries detected, scanning for them..."

# Collect candidate library directories. We search targeted paths rather than
# a blanket "external/*/lib" glob which produces thousands of entries and
# overflows the environment size limit.

EXTRA_DIRS=""

add_dir() {
  if [ -d "$1" ]; then
    EXTRA_DIRS="$EXTRA_DIRS:$1"
  fi
}

# --- Bazel hermetic libraries (targeted) ---
# Only add directories for the specific packages XLA actually links against.
# This keeps LD_LIBRARY_PATH small enough to avoid "Argument list too long".
BAZEL_PKGS="cuda_cudnn cuda_cupti cuda_nccl nvidia_nvshmem rocm local_config_rocm"
for cache_dir in "$HOME/.cache/bazel" "/root/.cache/bazel"; do
  if [ -d "$cache_dir" ]; then
    for pkg in $BAZEL_PKGS; do
      while IFS= read -r dir; do
        add_dir "$dir"
      done < <(find "$cache_dir" -path "*/external/${pkg}/lib" -type d 2>/dev/null)
    done
    # Also add the _solib directories (where Bazel stages runtime .so files)
    while IFS= read -r dir; do
      add_dir "$dir"
    done < <(find "$cache_dir" -path "*/_solib_*" -type d -maxdepth 8 2>/dev/null)
  fi
done

# --- TheRock / pip-installed ROCm ---
# rocm-sdk installs to a Python site-packages path; find it dynamically.
if command -v rocm-sdk &>/dev/null; then
  ROCM_ROOT=$(rocm-sdk path --root 2>/dev/null || true)
  if [ -n "$ROCM_ROOT" ]; then
    add_dir "$ROCM_ROOT/lib"
    add_dir "$ROCM_ROOT/lib/rocm"
  fi
fi

# --- System GPU library paths ---
for sys_dir in \
  "/opt/rocm/lib" \
  "/opt/rocm/hip/lib" \
  "/opt/rocm/lib/rocm" \
  "/usr/local/cuda/lib64" \
  "/usr/lib/x86_64-linux-gnu" \
  "/usr/lib/x86_64-linux-gnu/rocm/gfx11" \
; do
  add_dir "$sys_dir"
done

# --- Python site-packages ROCm libs (TheRock pip packages) ---
# TheRock nests libs in subdirectories like _rocm_sdk_core/lib/rocm_sysdeps/lib/
for sdk_dir in /usr/local/lib/python3.*/dist-packages/_rocm_sdk_*; do
  if [ -d "$sdk_dir" ]; then
    while IFS= read -r lib_dir; do
      add_dir "$lib_dir"
    done < <(find "$sdk_dir" -type d -name lib 2>/dev/null)
  fi
done

if [ -n "$EXTRA_DIRS" ]; then
  export LD_LIBRARY_PATH="${EXTRA_DIRS#:}:${LD_LIBRARY_PATH:-}"
  echo "[preflight] Added library paths to LD_LIBRARY_PATH"
fi

# Verify -- use a temp file to avoid "Argument list too long" with large env
_LDD_TMP=$(mktemp)
{ ldd "$LIBEXLA" 2>/dev/null | grep "not found" | awk '{print $1}' || true; } > "$_LDD_TMP"
STILL_MISSING=$(cat "$_LDD_TMP")
rm -f "$_LDD_TMP"

if [ -n "$STILL_MISSING" ]; then
  echo "[preflight] WARNING: Libraries still unresolved:"
  echo "$STILL_MISSING" | sed 's/^/  - /'
  echo "[preflight] The application may still work if these are optional runtime libs."
  echo "[preflight] Set LD_LIBRARY_PATH manually if needed."
  # Don't hard-fail -- let the app try to start
  return 0 2>/dev/null || exit 0
fi

echo "[preflight] All shared libraries resolved"
