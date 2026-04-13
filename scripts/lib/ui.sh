#!/usr/bin/env bash
# =============================================================================
# scripts/lib/ui.sh — Shared terminal UI helpers
# =============================================================================
# Provides colored output, step counters, and status functions used across
# setup, deploy, and entrypoint scripts.
#
# Usage:
#   source "$(dirname "$0")/lib/ui.sh"
# =============================================================================

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

_ui_step=0

step() {
  _ui_step=$((_ui_step + 1))
  echo ""
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  Step ${_ui_step}: $1${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; exit 1; }
info() { echo -e "  ${CYAN}→ $1${NC}"; }

banner() {
  local title="$1"
  echo ""
  echo -e "${CYAN}============================================================${NC}"
  echo -e "${CYAN}  ${title}${NC}"
  echo -e "${CYAN}============================================================${NC}"
}

print_elapsed() {
  local start_time="$1"
  local label="${2:-Done}"
  local elapsed=$(( SECONDS - start_time ))
  local mins=$(( elapsed / 60 ))
  local secs=$(( elapsed % 60 ))
  echo ""
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}  ${label} in ${mins}m ${secs}s${NC}"
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
}
