#!/bin/bash
# Build script for LSTM-VAE C++ Library with LibTorch
# This script handles building LibTorch from source and then building the LSTM-VAE library

set -e  # Exit on error

echo "========================================="
echo "LSTM-VAE C++ Library Build Script"
echo "========================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBTORCH_DIR="${SCRIPT_DIR}/libtorch"
PYTORCH_SRC_DIR="${SCRIPT_DIR}/pytorch"
BUILD_DIR="${SCRIPT_DIR}/build"
PYTHON_VERSION=${PYTHON_VERSION:-python3}
CUDA_VERSION=${CUDA_VERSION:-""}  # Set to e.g., "11.8" for CUDA support
BUILD_TYPE=${BUILD_TYPE:-Release}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required tools
    command -v cmake >/dev/null 2>&1 || missing_deps+=("cmake")
    command -v make >/dev/null 2>&1 || missing_deps+=("make")
    command -v g++ >/dev/null 2>&1 || missing_deps+=("g++")
    command -v ${PYTHON_VERSION} >/dev/null 2>&1 || missing_deps+=("python3")
    
    # Check for Python dependencies if building from source
    if [ ! -d "${LIBTORCH_DIR}" ]; then
        ${PYTHON_VERSION} -c "import torch" 2>/dev/null || missing_deps+=("python torch (pip install torch)")
        ${PYTHON_VERSION} -c "import yaml" 2>/dev/null || missing_deps+=("python yaml (pip install pyyaml)")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install them with:"
        log_info "  sudo apt-get install cmake make g++"
        log_info "  pip3 install torch pyyaml"
        exit 1
    fi
    
    log_info "All dependencies found!"
}

# Function to build LibTorch from source
build_libtorch_from_source() {
    log_info "Building LibTorch from source..."
    
    if [ ! -d "${PYTORCH_SRC_DIR}" ]; then
        log_info "Cloning PyTorch repository (this may take a while)..."
        git clone --recursive https://github.com/pytorch/pytorch.git "${PYTORCH_SRC_DIR}"
    else
        log_info "PyTorch source directory already exists, skipping clone"
        cd "${PYTORCH_SRC_DIR}"
        git pull
        git submodule update --init --recursive
        cd "${SCRIPT_DIR}"
    fi
    
    cd "${PYTORCH_SRC_DIR}"
    
    log_info "Building LibTorch (this may take 1-2 hours)..."
    
    # Set build options
    export MAX_JOBS=${MAX_JOBS:-$(nproc)}
    export USE_CUDA=${USE_CUDA:-0}
    export USE_CUDNN=${USE_CUDNN:-0}
    
    if [ -n "${CUDA_VERSION}" ]; then
        log_info "Building with CUDA ${CUDA_VERSION} support..."
        export USE_CUDA=1
        export USE_CUDNN=1
    fi
    
    # Build LibTorch
    ${PYTHON_VERSION} tools/build_libtorch.py
    
    if [ $? -ne 0 ]; then
        log_error "Failed to build LibTorch"
        exit 1
    fi
    
    # Find the built libtorch directory
    LIBTORCH_BUILD=$(find "${PYTORCH_SRC_DIR}" -name "libtorch" -type d | head -1)
    
    if [ -z "${LIBTORCH_BUILD}" ]; then
        log_error "Could not find built libtorch directory"
        exit 1
    fi
    
    log_info "LibTorch built successfully at: ${LIBTORCH_BUILD}"
    log_info "Creating symlink to ${LIBTORCH_DIR}..."
    
    cd "${SCRIPT_DIR}"
    ln -sf "${LIBTORCH_BUILD}" "${LIBTORCH_DIR}"
    
    cd "${SCRIPT_DIR}"
}

# Function to download pre-built LibTorch (alternative)
download_libtorch() {
    log_info "Downloading pre-built LibTorch..."
    
    local LIBTORCH_URL=""
    
    if [ -n "${CUDA_VERSION}" ]; then
        # CUDA version specified
        case "${CUDA_VERSION}" in
            "11.8")
                LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip"
                ;;
            "12.1")
                LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
                ;;
            *)
                log_error "Unsupported CUDA version: ${CUDA_VERSION}"
                exit 1
                ;;
        esac
    else
        # CPU-only version
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
    fi
    
    if [ ! -f "${SCRIPT_DIR}/libtorch.zip" ]; then
        log_info "Downloading from: ${LIBTORCH_URL}"
        wget -O "${SCRIPT_DIR}/libtorch.zip" "${LIBTORCH_URL}"
    else
        log_info "libtorch.zip already exists, skipping download"
    fi
    
    log_info "Extracting libtorch.zip..."
    unzip -q "${SCRIPT_DIR}/libtorch.zip" -d "${SCRIPT_DIR}"
    
    # Rename to libtorch if needed
    local EXTRACTED_DIR=$(ls -d "${SCRIPT_DIR}"/libtorch-* 2>/dev/null | head -1)
    if [ -n "${EXTRACTED_DIR}" ] && [ "${EXTRACTED_DIR}" != "${LIBTORCH_DIR}" ]; then
        mv "${EXTRACTED_DIR}" "${LIBTORCH_DIR}"
    fi
    
    log_info "LibTorch ready at: ${LIBTORCH_DIR}"
}

# Function to build the LSTM-VAE library
build_lstm_vae() {
    log_info "Building LSTM-VAE library..."
    
    if [ ! -d "${LIBTORCH_DIR}" ]; then
        log_error "LibTorch not found at ${LIBTORCH_DIR}"
        log_info "Please run with --build-libtorch or --download-libtorch first"
        exit 1
    fi
    
    # Create build directory
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    # Configure with CMake
    log_info "Running CMake configuration..."
    cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" \
          -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
          -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/install" \
          ..
    
    # Build
    log_info "Compiling (using ${MAX_JOBS:-$(nproc)} jobs)..."
    make -j${MAX_JOBS:-$(nproc)}
    
    if [ $? -ne 0 ]; then
        log_error "Build failed!"
        exit 1
    fi
    
    # Install
    log_info "Installing to ${SCRIPT_DIR}/install..."
    make install
    
    log_info "Build completed successfully!"
    log_info "Library installed to: ${SCRIPT_DIR}/install"
    log_info "Example executable: ${BUILD_DIR}/example"
}

# Function to clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    rm -rf "${BUILD_DIR}"
    rm -rf "${SCRIPT_DIR}/install"
    rm -f "${SCRIPT_DIR}/libtorch.zip"
    # Don't remove libtorch directory or pytorch source by default
    log_info "Clean completed. LibTorch and PyTorch source preserved."
}

# Function to clean everything including LibTorch
clean_all() {
    log_info "Cleaning all artifacts including LibTorch..."
    clean_build
    rm -rf "${LIBTORCH_DIR}"
    rm -rf "${PYTORCH_SRC_DIR}"
    log_info "Full clean completed."
}

# Print usage
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --build-libtorch      Build LibTorch from source (requires ~50GB disk space)
  --download-libtorch   Download pre-built LibTorch (faster, ~2GB)
  --build-library       Build the LSTM-VAE library (requires LibTorch)
  --cuda-version VER    Specify CUDA version (e.g., 11.8, 12.1) for GPU support
  --clean               Clean build artifacts (preserves LibTorch)
  --clean-all           Clean everything including LibTorch and PyTorch source
  --help                Show this help message

Examples:
  # Build everything from source (with CUDA 11.8)
  $0 --build-libtorch --build-library --cuda-version 11.8

  # Use pre-built LibTorch (CPU only)
  $0 --download-libtorch --build-library

  # Build with CUDA support using pre-built LibTorch
  $0 --download-libtorch --build-library --cuda-version 11.8

  # Just build the library (LibTorch already available)
  $0 --build-library

  # Clean build artifacts
  $0 --clean

Environment Variables:
  MAX_JOBS              Number of parallel jobs (default: nproc)
  BUILD_TYPE            CMake build type (default: Release)
  PYTHON_VERSION        Python interpreter (default: python3)

EOF
}

# Main script
main() {
    if [ $# -eq 0 ]; then
        print_usage
        exit 0
    fi
    
    local build_libtorch=false
    local download_libtorch=false
    local build_library=false
    local do_clean=false
    local do_clean_all=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-libtorch)
                build_libtorch=true
                shift
                ;;
            --download-libtorch)
                download_libtorch=true
                shift
                ;;
            --build-library)
                build_library=true
                shift
                ;;
            --cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            --clean)
                do_clean=true
                shift
                ;;
            --clean-all)
                do_clean_all=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Execute requested actions
    if [ "$do_clean_all" = true ]; then
        clean_all
        exit 0
    fi
    
    if [ "$do_clean" = true ]; then
        clean_build
        exit 0
    fi
    
    check_dependencies
    
    if [ "$build_libtorch" = true ]; then
        build_libtorch_from_source
    fi
    
    if [ "$download_libtorch" = true ]; then
        download_libtorch
    fi
    
    if [ "$build_library" = true ]; then
        build_lstm_vae
    fi
    
    log_info "All requested tasks completed!"
}

main "$@"
