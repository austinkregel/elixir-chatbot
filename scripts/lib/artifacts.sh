#!/usr/bin/env bash
# =============================================================================
# scripts/lib/artifacts.sh — Artifact presence detection
# =============================================================================
# Functions to check whether generated/downloaded artifacts exist. Each
# check_* function populates a global array with missing items and returns
# 0 if everything is present, 1 if something is missing.
#
# Usage:
#   source "$(dirname "$0")/lib/artifacts.sh"
#   check_term_models "$ROOT_DIR"
#   check_corpora "$ROOT_DIR"
#   print_artifact_report "$ROOT_DIR"
# =============================================================================

MISSING_MODELS=()
MISSING_LSTM_MODELS=()
MISSING_MICRO_MODELS=()
MISSING_CORPORA=()
MISSING_OURO=()
MISSING_GRAMMARS=()
MISSING_PYTHON_DATA=()
MISSING_WORDNET=()

check_term_models() {
  local root="${1:-$ROOT_DIR}"
  local ml_dir="$root/apps/brain/priv/ml_models"
  MISSING_MODELS=()
  MISSING_LSTM_MODELS=()
  MISSING_MICRO_MODELS=()

  local core_models=(
    "classifier.term"
    "embedder.term"
    "entity_model.term"
    "gazetteer.term"
    "sentiment_classifier.term"
    "speech_act_classifier.term"
    "pos_model.term"
  )

  for model in "${core_models[@]}"; do
    if [ ! -f "$ml_dir/$model" ]; then
      MISSING_MODELS+=("$model")
    fi
  done

  local lstm_models=(
    "lstm/unified_model.term"
    "lstm/response_scorer.term"
    "lstm/axon_intent.term"
  )

  for model in "${lstm_models[@]}"; do
    if [ ! -f "$ml_dir/$model" ]; then
      MISSING_LSTM_MODELS+=("$model")
    fi
  done

  local micro_models=(
    "micro/personal_question.term"
    "micro/clarification_response.term"
    "micro/modal_directive.term"
    "micro/fallback_response.term"
    "micro/goal_type.term"
    "micro/entity_type.term"
    "micro/user_fact_type.term"
    "micro/directed_at_bot.term"
    "micro/event_argument_role.term"
  )

  for model in "${micro_models[@]}"; do
    if [ ! -f "$ml_dir/$model" ]; then
      MISSING_MICRO_MODELS+=("$model")
    fi
  done

  local total_missing=$(( ${#MISSING_MODELS[@]} + ${#MISSING_LSTM_MODELS[@]} + ${#MISSING_MICRO_MODELS[@]} ))
  return $(( total_missing > 0 ? 1 : 0 ))
}

check_corpora() {
  local root="${1:-$ROOT_DIR}"
  local eval_dir="$root/apps/brain/priv/evaluation"
  MISSING_CORPORA=()

  local corpora=(
    "speech_act/gold_standard.json"
    "sentiment/gold_standard.json"
  )

  for corpus in "${corpora[@]}"; do
    if [ ! -f "$eval_dir/$corpus" ]; then
      MISSING_CORPORA+=("$corpus")
    fi
  done

  return $(( ${#MISSING_CORPORA[@]} > 0 ? 1 : 0 ))
}

check_wordnet() {
  local root="${1:-$ROOT_DIR}"
  local wn_dir="$root/apps/brain/priv/wordnet"
  MISSING_WORDNET=()

  if [ ! -d "$wn_dir" ] || [ -z "$(ls -A "$wn_dir"/*.pl 2>/dev/null)" ]; then
    MISSING_WORDNET+=("wordnet/*.pl")
  fi

  return $(( ${#MISSING_WORDNET[@]} > 0 ? 1 : 0 ))
}

check_ouro() {
  local root="${1:-$ROOT_DIR}"
  local ouro_dir="$root/apps/brain/priv/ml_models/ouro"
  MISSING_OURO=()

  local required_files=(
    "model.safetensors"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "config.json"
  )

  for f in "${required_files[@]}"; do
    if [ ! -f "$ouro_dir/$f" ]; then
      MISSING_OURO+=("$f")
    fi
  done

  return $(( ${#MISSING_OURO[@]} > 0 ? 1 : 0 ))
}

check_grammars() {
  local root="${1:-$ROOT_DIR}"
  local gram_dir="$root/apps/brain/priv/code/grammars"
  MISSING_GRAMMARS=()

  if [ ! -d "$gram_dir" ] || [ -z "$(ls -A "$gram_dir"/*.so 2>/dev/null)" ]; then
    MISSING_GRAMMARS+=("tree-sitter grammars (*.so)")
  fi

  return $(( ${#MISSING_GRAMMARS[@]} > 0 ? 1 : 0 ))
}

check_python_generated() {
  local root="${1:-$ROOT_DIR}"
  MISSING_PYTHON_DATA=()

  if [ ! -f "$root/data/informal_expansions.json" ]; then
    MISSING_PYTHON_DATA+=("data/informal_expansions.json")
  fi
  if [ ! -f "$root/data/heuristics/seeded_heuristics.json" ]; then
    MISSING_PYTHON_DATA+=("data/heuristics/seeded_heuristics.json")
  fi

  return $(( ${#MISSING_PYTHON_DATA[@]} > 0 ? 1 : 0 ))
}

check_data_classifiers() {
  local root="${1:-$ROOT_DIR}"
  local cls_dir="$root/data/classifiers"

  if [ ! -d "$cls_dir" ] || [ -z "$(ls -A "$cls_dir"/*.json 2>/dev/null)" ]; then
    return 1
  fi
  return 0
}

_report_section() {
  local label="$1"
  shift
  local missing=("$@")

  if [ ${#missing[@]} -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} ${label}"
  else
    echo -e "  ${RED}✗${NC} ${label}"
    for item in "${missing[@]}"; do
      echo -e "      ${YELLOW}missing:${NC} $item"
    done
  fi
}

print_artifact_report() {
  local root="${1:-$ROOT_DIR}"

  check_term_models "$root"
  check_corpora "$root"
  check_wordnet "$root"
  check_ouro "$root"
  check_grammars "$root"
  check_python_generated "$root"

  echo ""
  echo -e "  ${BOLD}Artifact Status${NC}"
  echo -e "  ─────────────────────────────────────────"
  _report_section "Core TF-IDF models (${#MISSING_MODELS[@]} missing)" "${MISSING_MODELS[@]}"
  _report_section "LSTM models (${#MISSING_LSTM_MODELS[@]} missing)" "${MISSING_LSTM_MODELS[@]}"
  _report_section "Micro classifiers (${#MISSING_MICRO_MODELS[@]} missing)" "${MISSING_MICRO_MODELS[@]}"
  _report_section "Training corpora (${#MISSING_CORPORA[@]} missing)" "${MISSING_CORPORA[@]}"
  _report_section "WordNet data (${#MISSING_WORDNET[@]} missing)" "${MISSING_WORDNET[@]}"
  _report_section "Ouro model files (${#MISSING_OURO[@]} missing)" "${MISSING_OURO[@]}"
  _report_section "Tree-sitter grammars (${#MISSING_GRAMMARS[@]} missing)" "${MISSING_GRAMMARS[@]}"
  _report_section "Python-generated data (${#MISSING_PYTHON_DATA[@]} missing)" "${MISSING_PYTHON_DATA[@]}"
  echo ""
}
