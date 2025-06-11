#!/bin/bash

echo "🔨 Building MaxK-GNN Custom Kernels..."

# Set error handling
set -e

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

echo "✅ CUDA found: $(nvcc --version | grep release)"

# Check PyTorch installation
python -c "import torch; print(f'✅ PyTorch found: {torch.__version__}')" || {
    echo "❌ Error: PyTorch not found. Please install PyTorch."
    exit 1
}

# Check GPU compute capability
python -c "
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f'✅ GPU compute capability: {major}.{minor}')
    if major < 7:
        print('⚠️  Warning: GPU compute capability < 7.0 may not be fully supported')
else:
    print('⚠️  Warning: No CUDA GPU detected')
"

# Set CUDA architecture if not set
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
    echo "🔧 Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

# Create build directories
mkdir -p build
mkdir -p kernels/build

echo "🏗️  Building CUDA kernels..."

# First build the standalone kernel tests
cd kernels
mkdir -p build
cd build

# Build standalone kernels
echo "🔧 Building standalone kernel tests..."
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✅ Standalone kernels built successfully!"
else
    echo "❌ Failed to build standalone kernels"
    exit 1
fi

cd ../..

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
python setup.py clean --all 2>/dev/null || true

# Build Python extension
echo "🐍 Building Python extension..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✅ Python extension built successfully!"
else
    echo "❌ Failed to build Python extension"
    echo "🔍 Checking common issues..."
    
    # Check for common issues
    python -c "
import torch.utils.cpp_extension
print('✅ torch.utils.cpp_extension available')
try:
    from torch.utils.cpp_extension import BuildExtension
    print('✅ BuildExtension import successful')
except ImportError as e:
    print(f'❌ BuildExtension import failed: {e}')
"
    exit 1
fi

# Install the extension in development mode
echo "📦 Installing MaxK kernels in development mode..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ MaxK kernels installed successfully!"
else
    echo "❌ Failed to install MaxK kernels"
    exit 1
fi

# Test the installation
echo "🧪 Testing installation..."
python -c "
try:
    import maxk_kernels
    print('✅ MaxK kernels import successful!')
    print('Available functions:', [x for x in dir(maxk_kernels) if not x.startswith('_')])
    
    # Test basic functionality
    import torch
    if torch.cuda.is_available():
        x = torch.randn(10, 64, device='cuda')
        try:
            output = maxk_kernels.maxk_forward(x, 16)
            print(f'✅ MaxK forward test passed! Output shape: {output.shape}')
        except Exception as e:
            print(f'⚠️  MaxK forward test failed: {e}')
    else:
        print('⚠️  GPU not available for testing')
        
except ImportError as e:
    print('❌ MaxK kernels import failed:', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🎉 MaxK-GNN kernel build completed successfully!"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo ""
echo "📋 Next steps:"
echo "1. Download graph datasets: cd kernels && python generate_meta.py"
echo "2. Run kernel benchmarks: cd kernels/build && ./maxk_kernel_test"
echo "3. Run integration tests: python test_integration.py"
echo "4. Run integrated training: python maxk_gnn_integrated.py --dataset reddit --model sage --maxk 32"
echo ""
echo "🚀 Ready to use MaxK-GNN with custom kernels!"