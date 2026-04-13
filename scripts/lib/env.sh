#!/usr/bin/env bash
# =============================================================================
# scripts/lib/env.sh — Environment and dotenv helpers
# =============================================================================
# Provides ROOT_DIR resolution, .env loading, and interrupt trap installation.
#
# Usage:
#   source "$(dirname "$0")/lib/env.sh"
#   resolve_root_dir            # sets ROOT_DIR and cd's into it
#   load_dotenv                 # sources .env if present
#   install_interrupt_trap      # installs INT/TERM cleanup handler
# =============================================================================

resolve_root_dir() {
  local script_path="${BASH_SOURCE[1]:-$0}"
  ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$script_path")/.." && pwd)}"
  cd "$ROOT_DIR" || exit 1
}

load_dotenv() {
  local dir="${1:-$ROOT_DIR}"
  if [ -f "$dir/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$dir/.env"
    set +a
  fi
}

install_interrupt_trap() {
  _cleanup() {
    trap - INT TERM
    echo ""
    echo -e "\033[0;31m  Interrupted — killing child processes and exiting.\033[0m"
    kill -- -$$ 2>/dev/null
    exit 130
  }
  trap _cleanup INT TERM
}
