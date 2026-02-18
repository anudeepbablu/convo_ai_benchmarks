#!/usr/bin/env bash
# Launch Nvidia Parakeet 1.1B ASR NIM container
# Requires NGC_API_KEY in environment or .env file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env if present
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

: "${NGC_API_KEY:?NGC_API_KEY must be set in environment or .env file}"

IMAGE="nvcr.io/nim/nvidia/parakeet-ctc-1.1b-asr:latest"
CONTAINER_NAME="riva-asr"
GRPC_PORT="${ASR_GRPC_PORT:-50051}"
MODEL_CACHE="${MODEL_CACHE_DIR:-$HOME/.cache/nim}/asr"

mkdir -p "$MODEL_CACHE"

# Stop existing container if running
if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
  echo "Stopping existing $CONTAINER_NAME container..."
  docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
fi

echo "Pulling $IMAGE ..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

docker pull "$IMAGE"

echo "Starting $CONTAINER_NAME on gRPC port $GRPC_PORT ..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus '"device=0"' \
  --shm-size=8g \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$MODEL_CACHE:/opt/nim/.cache" \
  -p "${GRPC_PORT}:50051" \
  "$IMAGE"

echo "Container $CONTAINER_NAME started."
echo "Waiting for health check..."
for i in $(seq 1 60); do
  if docker exec "$CONTAINER_NAME" curl -sf http://localhost:8000/v1/health/ready &>/dev/null 2>&1; then
    echo "ASR NIM is ready on gRPC port $GRPC_PORT"
    exit 0
  fi
  sleep 5
  echo "  ... still waiting ($((i*5))s)"
done
echo "WARNING: Health check timed out. Container may still be initializing."
