#!/usr/bin/env bash
# =============================================================================
# scripts/lib/xla.sh — XLA/EXLA hardware detection and configuration
# =============================================================================
# Detects GPU hardware and sets XLA_TARGET accordingly. Also configures the
# XLA JIT compilation cache and GPU-specific environment flags.
#
# Usage:
#   source "$(dirname "$0")/lib/xla.sh"
#   detect_xla_target       # sets XLA_TARGET if unset
#   configure_xla_cache     # sets XLA_FLAGS and cache directory
# =============================================================================

detect_xla_target() {
  if [ -n "${XLA_TARGET:-}" ]; then
    return 0
  fi

  if [ "$(uname -s)" = "Darwin" ]; then
    export XLA_TARGET=cpu
    if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "apple"; then
      _XLA_HW_NOTE="Apple Silicon detected — EXLA uses CPU backend (no Metal support)"
    else
      _XLA_HW_NOTE="macOS Intel detected — EXLA uses CPU backend"
    fi
    return 0
  fi

  if [ -d "/opt/rocm" ]; then
    export XLA_TARGET=rocm
    _XLA_HW_NOTE="ROCm detected at /opt/rocm"
    return 0
  fi

  if [ -d "/usr/local/cuda" ]; then
    local cuda_major
    cuda_major=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "")
    case "$cuda_major" in
      12) export XLA_TARGET=cuda12 ;;
      13) export XLA_TARGET=cuda13 ;;
      *)  export XLA_TARGET=cuda ;;
    esac
    _XLA_HW_NOTE="CUDA detected (major version: ${cuda_major:-unknown})"
    return 0
  fi

  export XLA_TARGET=cpu
  _XLA_HW_NOTE="No GPU detected — using CPU backend"
}

configure_xla_cache() {
  XLA_CACHE_DIR="${XLA_CACHE_DIR:-${HOME}/.cache/xla_jit}"
  mkdir -p "$XLA_CACHE_DIR"

  if [ "${XLA_TARGET:-cpu}" != "cpu" ]; then
    local extra_flags="--xla_gpu_enable_xla_runtime_executable=true"
    extra_flags="${extra_flags} --xla_persistent_cache_directory=${XLA_CACHE_DIR}"
    extra_flags="${extra_flags} --xla_gpu_autotune_level=2"
    export XLA_FLAGS="${XLA_FLAGS:-} ${extra_flags}"
    export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
  fi
}
