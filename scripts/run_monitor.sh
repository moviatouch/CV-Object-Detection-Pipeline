#!/bin/bash

cd "$(dirname "$0")/.."

echo "📡 Starting Viatouch CV Pipeline Monitor"
echo "----------------------------------------"
echo "Watching: /home/cv-nx/cv_pipeline/camera-preview-pipeline/post_archive"
echo "Press Ctrl+C to stop."
echo "----------------------------------------"

# Run the container in interactive mode with the monitor command
# docker run --rm -it \
#   --runtime nvidia \
#   -v $(pwd)/models:/app/models \
#   -v $(pwd)/outputs:/app/outputs \
#   -v $(pwd)/videos:/app/videos \
#   -v /home/cv-nx/cv_pipeline/camera-preview-pipeline/post_archive:/app/post_archive \
#   -e POST_ARCHIVE_PATH=/app/post_archive \
#   -e MOCK_API=false \
#   viatouch-cv-pipeline:latest \
#   python3 scripts/monitor.py

docker build -t viatouch-cv-pipeline:latest .

docker create  \
  --restart always \
  --name viatouch_cv \
  --runtime nvidia \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/videos:/app/videos \
  -v /home/cv-nx/cv_pipeline/camera-preview-pipeline/post_archive:/app/post_archive \
  -e POST_ARCHIVE_PATH=/app/post_archive \
  -e MOCK_API=false \
  viatouch-cv-pipeline:latest \
  python3 scripts/monitor.py

docker start viatouch_cv