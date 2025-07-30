#!/bin/bash

# Build script for edepth Docker containers
# Supports cross-platform builds and GPU/CPU variants

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PLATFORM="auto"
BUILD_TYPE="both"
PUSH=false
CACHE=true
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build edepth Docker containers with GPU and CPU support

OPTIONS:
    -p, --platform PLATFORM    Target platform (auto|linux/amd64|linux/arm64|darwin/amd64|darwin/arm64)
    -t, --type TYPE            Build type (gpu|cpu|both) [default: both]
    -c, --no-cache             Disable build cache
    -v, --verbose              Verbose output
    --push                     Push images to registry
    --help                     Show this help message

EXAMPLES:
    $0                                    # Build both GPU and CPU variants for current platform
    $0 --type gpu                        # Build only GPU variant
    $0 --type cpu --platform linux/amd64 # Build CPU variant for linux/amd64
    $0 --push                            # Build and push to registry
    $0 --no-cache --verbose              # Build without cache with verbose output

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY           Docker registry URL (default: none)
    IMAGE_TAG                Image tag (default: latest)
    PYTHON_VERSION           Python version (default: 3.10)
    CUDA_VERSION             CUDA version (default: 11.8)
    UBUNTU_VERSION           Ubuntu version (default: 20.04)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -c|--no-cache)
            CACHE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Environment variables with defaults
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}
CUDA_VERSION=${CUDA_VERSION:-"11.8"}
UBUNTU_VERSION=${UBUNTU_VERSION:-"20.04"}

# Detect platform if auto
if [[ "$PLATFORM" == "auto" ]]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            PLATFORM="linux/amd64"
        elif [[ "$ARCH" == "aarch64" ]]; then
            PLATFORM="linux/arm64"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        ARCH=$(uname -m)
        if [[ "$ARCH" == "x86_64" ]]; then
            PLATFORM="darwin/amd64"
        elif [[ "$ARCH" == "arm64" ]]; then
            PLATFORM="darwin/arm64"
        fi
    else
        PLATFORM="linux/amd64"  # Default fallback
    fi
fi

print_status "Detected platform: $PLATFORM"

# Set image names
if [[ -n "$DOCKER_REGISTRY" ]]; then
    GPU_IMAGE="${DOCKER_REGISTRY}/edepth:gpu-${IMAGE_TAG}"
    CPU_IMAGE="${DOCKER_REGISTRY}/edepth:cpu-${IMAGE_TAG}"
else
    GPU_IMAGE="edepth:gpu-${IMAGE_TAG}"
    CPU_IMAGE="edepth:cpu-${IMAGE_TAG}"
fi

# Build arguments
BUILD_ARGS="--build-arg PYTHON_VERSION=${PYTHON_VERSION}"
BUILD_ARGS="${BUILD_ARGS} --build-arg CUDA_VERSION=${CUDA_VERSION}"
BUILD_ARGS="${BUILD_ARGS} --build-arg UBUNTU_VERSION=${UBUNTU_VERSION}"

if [[ "$CACHE" == "false" ]]; then
    BUILD_ARGS="${BUILD_ARGS} --no-cache"
fi

if [[ "$VERBOSE" == "true" ]]; then
    BUILD_ARGS="${BUILD_ARGS} --progress=plain"
fi

if [[ "$PLATFORM" != "auto" ]]; then
    BUILD_ARGS="${BUILD_ARGS} --platform ${PLATFORM}"
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    exit 1
fi

# Function to build image
build_image() {
    local image_type=$1
    local image_name=$2
    local base_image=$3
    
    print_status "Building $image_type image: $image_name"
    print_status "Platform: $PLATFORM"
    print_status "Base image: $base_image"
    
    # Check if buildx is available for multi-platform builds
    if [[ "$PLATFORM" != "auto" ]] && docker buildx version &> /dev/null; then
        docker buildx build \
            ${BUILD_ARGS} \
            --build-arg BASE_IMAGE=${base_image} \
            --tag ${image_name} \
            --load \
            .
    else
        docker build \
            ${BUILD_ARGS} \
            --build-arg BASE_IMAGE=${base_image} \
            --tag ${image_name} \
            .
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "Successfully built $image_type image: $image_name"
    else
        print_error "Failed to build $image_type image"
        exit 1
    fi
}

# Build GPU image
if [[ "$BUILD_TYPE" == "gpu" || "$BUILD_TYPE" == "both" ]]; then
    # Check if NVIDIA runtime is available
    if docker info 2>/dev/null | grep -q nvidia; then
        print_status "NVIDIA Docker runtime detected"
        build_image "GPU" "$GPU_IMAGE" "cuda-base"
    else
        print_warning "NVIDIA Docker runtime not detected, GPU features may not work"
        print_warning "Building GPU image anyway (will fallback to CPU at runtime)"
        build_image "GPU" "$GPU_IMAGE" "cuda-base"
    fi
fi

# Build CPU image
if [[ "$BUILD_TYPE" == "cpu" || "$BUILD_TYPE" == "both" ]]; then
    build_image "CPU" "$CPU_IMAGE" "cpu-base"
fi

# Test images
print_status "Testing built images..."

test_image() {
    local image_name=$1
    local image_type=$2
    
    print_status "Testing $image_type image: $image_name"
    
    if docker run --rm ${image_name} python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; then
        print_success "$image_type image test passed"
    else
        print_error "$image_type image test failed"
        return 1
    fi
}

# Test GPU image
if [[ "$BUILD_TYPE" == "gpu" || "$BUILD_TYPE" == "both" ]]; then
    test_image "$GPU_IMAGE" "GPU"
fi

# Test CPU image
if [[ "$BUILD_TYPE" == "cpu" || "$BUILD_TYPE" == "both" ]]; then
    test_image "$CPU_IMAGE" "CPU"
fi

# Push images if requested
if [[ "$PUSH" == "true" ]]; then
    if [[ -z "$DOCKER_REGISTRY" ]]; then
        print_error "DOCKER_REGISTRY environment variable must be set to push images"
        exit 1
    fi
    
    print_status "Pushing images to registry..."
    
    if [[ "$BUILD_TYPE" == "gpu" || "$BUILD_TYPE" == "both" ]]; then
        print_status "Pushing GPU image: $GPU_IMAGE"
        docker push "$GPU_IMAGE"
        print_success "Successfully pushed GPU image"
    fi
    
    if [[ "$BUILD_TYPE" == "cpu" || "$BUILD_TYPE" == "both" ]]; then
        print_status "Pushing CPU image: $CPU_IMAGE"
        docker push "$CPU_IMAGE"
        print_success "Successfully pushed CPU image"
    fi
fi

# Print summary
print_success "Build completed successfully!"
echo
echo "Built images:"
if [[ "$BUILD_TYPE" == "gpu" || "$BUILD_TYPE" == "both" ]]; then
    echo "  GPU: $GPU_IMAGE"
fi
if [[ "$BUILD_TYPE" == "cpu" || "$BUILD_TYPE" == "both" ]]; then
    echo "  CPU: $CPU_IMAGE"
fi
echo
echo "Usage examples:"
echo "  Docker Compose (GPU): docker-compose --profile gpu up"
echo "  Docker Compose (CPU): docker-compose --profile cpu up"
echo "  Direct run (GPU): docker run --gpus all -it $GPU_IMAGE"
echo "  Direct run (CPU): docker run -it $CPU_IMAGE"
echo
print_status "Build script completed successfully!"
