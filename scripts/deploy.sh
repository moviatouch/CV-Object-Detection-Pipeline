#!/bin/bash

cd "$(dirname "$0")"

echo "📡 Starting Viatouch CV Pipeline Monitor"
echo "----------------------------------------"
echo "Watching: /home/cv-nx/cv_pipeline/camera-preview-pipeline/post_archive"
echo "Press Ctrl+C to stop."
echo "----------------------------------------"




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
  -v /home/cv-nx/cv_pipeline/camera-preview-pipeline/post_archive:/app/post_archive \
  -e POST_ARCHIVE_PATH=/app/post_archive \
  -e MOCK_API=false \
  viatouch-cv-pipeline:latest

docker start viatouch_cv