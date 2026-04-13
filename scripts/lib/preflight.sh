#!/usr/bin/env bash
# =============================================================================
# scripts/lib/preflight.sh — EXLA library preflight wrapper
# =============================================================================
# Wraps the exla_preflight.sh sourcing pattern used by multiple scripts.
#
# Usage:
#   source "$(dirname "$0")/lib/preflight.sh"
#   source_exla_preflight [root_dir]
# =============================================================================

source_exla_preflight() {
  local root="${1:-$ROOT_DIR}"
  local preflight="$root/scripts/exla_preflight.sh"
  local build_dir="$root/_build"

  if [ -f "$preflight" ]; then
    # shellcheck source=/dev/null
    source "$preflight" "$build_dir"
    ok "EXLA libraries checked"
  else
    warn "exla_preflight.sh not found, skipping"
  fi
}
