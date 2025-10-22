#!/bin/bash
# Build script for KRadar CUDA extensions
# This script builds all the CUDA extensions needed for the KRadar model

set -e  # Exit on error

echo "========================================"
echo "Building KRadar CUDA Extensions"
echo "========================================"
echo ""

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch is not installed!"
    echo "Please install PyTorch first:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# Check if CUDA is available
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "WARNING: CUDA is not available in PyTorch!"
    echo "The extensions will be built but may not work properly."
    echo ""
fi

# Build iou3d_nms
echo "Building iou3d_nms..."
cd ops/iou3d_nms
python setup.py build_ext --inplace
echo "✓ iou3d_nms built successfully"
echo ""

# Build roiaware_pool3d
echo "Building roiaware_pool3d..."
cd ../roiaware_pool3d
python setup.py build_ext --inplace
echo "✓ roiaware_pool3d built successfully"
echo ""

# Build pointnet2_stack
echo "Building pointnet2_stack..."
cd ../pointnet2/pointnet2_stack
python setup.py build_ext --inplace
echo "✓ pointnet2_stack built successfully"
echo ""

# Build pointnet2_batch
echo "Building pointnet2_batch..."
cd ../pointnet2_batch
python setup.py build_ext --inplace
echo "✓ pointnet2_batch built successfully"
echo ""

cd "$SCRIPT_DIR"
echo "========================================"
echo "All CUDA extensions built successfully!"
echo "========================================"
echo ""
echo "The compiled extensions are now compatible with Python $PYTHON_VERSION"
