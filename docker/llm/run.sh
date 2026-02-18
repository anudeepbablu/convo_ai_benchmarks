#!/usr/bin/env bash
# Launch meta/llama-3.1-8b-instruct NIM container
# Requires NGC_API_KEY in environment or .env file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

: "${NGC_API_KEY:?NGC_API_KEY must be set in environment or .env file}"

IMAGE="nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"
CONTAINER_NAME="nim-llm"
HTTP_PORT="${LLM_HTTP_PORT:-8000}"
MODEL_CACHE="${MODEL_CACHE_DIR:-$HOME/.cache/nim}/llm"

mkdir -p "$MODEL_CACHE"

if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
  echo "Stopping existing $CONTAINER_NAME container..."
  docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
fi

echo "Pulling $IMAGE ..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

docker pull "$IMAGE"

echo "Starting $CONTAINER_NAME on HTTP port $HTTP_PORT ..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus '"device=0"' \
  --shm-size=16g \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$MODEL_CACHE:/opt/nim/.cache" \
  -p "${HTTP_PORT}:8000" \
  "$IMAGE"

echo "Container $CONTAINER_NAME started."
echo "Waiting for health check..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${HTTP_PORT}/v1/health/ready" &>/dev/null; then
    echo "LLM NIM is ready on http://localhost:${HTTP_PORT}"
    exit 0
  fi
  sleep 5
  echo "  ... still waiting ($((i*5))s)"
done
echo "WARNING: Health check timed out. Container may still be initializing."
