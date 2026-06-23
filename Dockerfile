# Use Python 3.10.12 slim image as base
FROM ultralytics-jetson:latest

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for package compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    wget \
    curl \
    build-essential \
    python3-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Create necessary directories
RUN mkdir -p uploads flagged_images processed_images dataset outputs

# Create models directory as mount point (will be empty in container)
RUN mkdir -p models && \
    echo "Mount your model files here using: docker run -v /path/to/your/models:/app/models" > models/README.txt

# Create volume mount points for models and outputs
VOLUME ["/app/models", "/app/outputs"]

# Set the default entrypoint to run main.py
CMD ["python3", "monitor.py"]