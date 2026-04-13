#!/bin/bash
# ============================================================================
# verify_no_string_matching.sh
# ============================================================================
# Verifies that event extraction code uses tensor operations, not string matching.
# 
# This script is designed to run in CI to ensure the data-driven approach
# is maintained in the event extraction system.
#
# Usage:
#   ./scripts/verify_no_string_matching.sh
#
# Exit codes:
#   0 - No string matching detected (success)
#   1 - String matching detected in event code (failure)
# ============================================================================

set -e

echo "=========================================="
echo "Verifying no string matching in event code"
echo "=========================================="

# Files to check for string matching patterns
EVENT_FILES=(
  "apps/brain/lib/brain/analysis/event_extractor.ex"
  "apps/brain/lib/brain/analysis/event_patterns.ex"
  "apps/brain/lib/brain/analysis/types/event.ex"
)

# Patterns that indicate string matching (forbidden)
FORBIDDEN_PATTERNS=(
  "~r/"              # Regex sigil
  "String\.contains\?"
  "String\.match\?"
  "String\.starts_with\?"
  "String\.ends_with\?"
  "Regex\."           # Regex module functions
)

# Build grep pattern from forbidden patterns
GREP_PATTERN=$(IFS='|'; echo "${FORBIDDEN_PATTERNS[*]}")

echo ""
echo "Checking files:"
for file in "${EVENT_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "  - $file"
  else
    echo "  - $file (not found, skipping)"
  fi
done

echo ""
echo "Forbidden patterns: ${FORBIDDEN_PATTERNS[*]}"
echo ""

# Check for violations
VIOLATIONS=0

for file in "${EVENT_FILES[@]}"; do
  if [ -f "$file" ]; then
    # Use grep with extended regex, show line numbers
    if grep -nE "$GREP_PATTERN" "$file" 2>/dev/null; then
      echo ""
      echo "ERROR: String matching detected in $file"
      VIOLATIONS=$((VIOLATIONS + 1))
    fi
  fi
done

echo ""
echo "=========================================="

if [ $VIOLATIONS -gt 0 ]; then
  echo "FAILED: Found $VIOLATIONS file(s) with string matching"
  echo ""
  echo "Event extraction must use tensor operations only."
  echo "Replace regex/string matching with:"
  echo "  - Nx tensor comparisons (Nx.equal, Nx.logical_or)"
  echo "  - POS tag indices from EventPatterns"
  echo "  - Character-level operations (String.graphemes)"
  echo ""
  exit 1
else
  echo "PASSED: No string matching in event extraction code"
  echo ""
  echo "Verified files use tensor-based operations."
  exit 0
fi
