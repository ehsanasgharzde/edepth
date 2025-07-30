# Multi-stage build for flexibility and optimization
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=11.8
ARG UBUNTU_VERSION=20.04

# Stage 1: Base image with CUDA support (will fallback to CPU if GPU unavailable)
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as cuda-base

# Stage 2: CPU-only base image
FROM ubuntu:${UBUNTU_VERSION} as cpu-base

# Stage 3: Python base setup
FROM ${BASE_IMAGE:-cuda-base} as python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    # OpenEXR dependencies
    libopenexr-dev \
    libilmbase-dev \
    # Additional ML/CV dependencies
    libopencv-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with GPU/CPU detection
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "GPU installation failed, trying CPU-only..." && \
     pip install torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
     pip install --no-cache-dir -r requirements.txt)

# Create necessary directories
RUN mkdir -p /app/outputs /app/logs /app/data /app/checkpoints

# Copy application code
COPY . .

# Set permissions
RUN chmod +x /app/run.py

# Create non-root user for security
RUN groupadd -r edepth && useradd -r -g edepth -d /app -s /bin/bash edepth && \
    chown -R edepth:edepth /app

# Switch to non-root user
USER edepth

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Default command
CMD ["python", "run.py", "--help"]

# Expose port for monitoring dashboard
EXPOSE 8501