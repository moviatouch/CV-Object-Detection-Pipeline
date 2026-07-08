#!/bin/bash

current_dir=$(pwd)
current_dir=${current_dir%/scripts}

# Helper function to stop and remove a container if it exists
recreate_container() {
  local name=$1
  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    echo "🔄 Stopping and removing existing container: $name"
    docker stop "$name" 2>/dev/null
    docker rm "$name"
  fi
}


cd "$current_dir"
docker build -t viatouch-cv-pipeline:latest .


recreate_container viatouch_cv
docker create  \
  --restart always \
  --name viatouch_cv \
  --runtime nvidia \
  -v ${current_dir}/models:/app/models \
  -v ${current_dir}/outputs:/app/outputs \
  -v ${current_dir}/videos:/app/videos \
  -v ${current_dir}/../camera-preview-pipeline/post_archive:/app/post_archive \
  -e POST_ARCHIVE_PATH=/app/post_archive \
  -e MOCK_API=false \
  viatouch-cv-pipeline:latest

docker start viatouch_cv