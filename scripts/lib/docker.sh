#!/usr/bin/env bash
# =============================================================================
# scripts/lib/docker.sh — Docker Compose hardware detection
# =============================================================================
# Detects available GPU hardware and returns the appropriate Docker Compose
# file flags and Dockerfile selection.
#
# Usage:
#   source "$(dirname "$0")/lib/docker.sh"
#   detect_docker_compose_files   # sets COMPOSE_FILES
#   suggest_gpu_dockerfile        # sets SUGGESTED_DOCKERFILE
# =============================================================================

detect_docker_compose_files() {
  local root="${1:-$ROOT_DIR}"

  COMPOSE_FILES="-f $root/docker-compose.yml"

  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || [ -d "/opt/rocm" ]; then
    COMPOSE_FILES="$COMPOSE_FILES -f $root/docker-compose.rocm.yml"
    _DOCKER_GPU="rocm"
    return 0
  fi

  if command -v nvidia-smi &>/dev/null || [ -d "/usr/local/cuda" ]; then
    COMPOSE_FILES="$COMPOSE_FILES -f $root/docker-compose.cuda.yml"
    _DOCKER_GPU="cuda"
    return 0
  fi

  _DOCKER_GPU="cpu"
}

suggest_gpu_dockerfile() {
  detect_docker_compose_files "${1:-$ROOT_DIR}"

  case "$_DOCKER_GPU" in
    rocm) SUGGESTED_DOCKERFILE="Dockerfile.train.rocm" ;;
    cuda) SUGGESTED_DOCKERFILE="Dockerfile.train.cuda" ;;
    *)    SUGGESTED_DOCKERFILE="Dockerfile" ;;
  esac
}
