#!/bin/bash

echo "🔧 Fixing PyTorch library linking issue..."
echo "This resolves the 'libc10.so: cannot open shared object file' error"
echo ""

# Get PyTorch library path
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

echo "🔍 PyTorch libraries located at: $TORCH_LIB_PATH"

# Check if libc10.so exists there
if [ -f "$TORCH_LIB_PATH/libc10.so" ]; then
    echo "✅ Found libc10.so in PyTorch directory"
else
    echo "❌ libc10.so not found in expected location"
    echo "🔍 Searching for libc10.so..."
    find $(python -c "import torch; print(torch.__path__[0])") -name "libc10.so" 2>/dev/null || echo "Not found"
fi

echo ""
echo "🔧 Setting up library path..."

# Method 1: Set LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$TORCH_LIB_PATH" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_LIB_PATH

echo "✅ Added PyTorch lib path to LD_LIBRARY_PATH"

# Method 2: Set CUDA library path as well
CUDA_LIB_PATH="/usr/local/cuda/lib64"
if [ -d "$CUDA_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_PATH
    echo "✅ Added CUDA lib path to LD_LIBRARY_PATH"
fi

# Method 3: Check for newer CUDA version
CUDA_LIB_PATH_NEW="/usr/local/cuda-12.8/lib64"
if [ -d "$CUDA_LIB_PATH_NEW" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_PATH_NEW
    echo "✅ Added CUDA 12.8 lib path to LD_LIBRARY_PATH"
fi

echo ""
echo "🧪 Testing import with fixed library path..."

# Test the import
python -c "
import sys
sys.path.insert(0, '.')
try:
    import maxk_kernels
    print('✅ Import successful!')
    print('Available functions:', [f for f in dir(maxk_kernels) if not f.startswith('_')])
    
    # Test CUDA functionality
    import torch
    if torch.cuda.is_available():
        print('🔥 Testing CUDA...')
        x = torch.randn(4, 32, device='cuda')
        if hasattr(maxk_kernels, 'maxk_forward'):
            try:
                output = maxk_kernels.maxk_forward(x, 8)
                print(f'✅ MaxK test passed! {x.shape} -> {output.shape}')
            except Exception as e:
                print(f'⚠️ MaxK test failed: {e}')
        else:
            print('⚠️ MaxK functions not found, but import succeeded')
    
    print('🎉 Extension is working!')
    
except ImportError as e:
    print(f'❌ Import still failed: {e}')
    print('')
    print('🔍 Debugging library dependencies...')
    
    import os
    import subprocess
    
    # Find the .so file
    import glob
    so_files = glob.glob('maxk_kernels*.so')
    if so_files:
        so_file = so_files[0]
        print(f'Checking dependencies of: {so_file}')
        
        # Check what libraries it needs
        try:
            result = subprocess.run(['ldd', so_file], capture_output=True, text=True)
            print('Library dependencies:')
            print(result.stdout)
            
            # Look for missing libraries
            missing_libs = [line for line in result.stdout.split('\n') if 'not found' in line]
            if missing_libs:
                print('❌ Missing libraries:')
                for lib in missing_libs:
                    print(f'  {lib}')
            else:
                print('✅ All libraries found')
                
        except Exception as e:
            print(f'Could not check dependencies: {e}')
    else:
        print('No .so files found')
"

echo ""
echo "💡 If the import still fails, try these additional solutions:"
echo ""
echo "1. Reload your shell:"
echo "   source ~/.bashrc"
echo "   # or restart your terminal"
echo ""
echo "2. Set library path for current session:"
echo "   export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$TORCH_LIB_PATH"
echo ""
echo "3. Rebuild with RPATH (recommended):"
echo "   python setup_fixed.py build_ext --inplace"
echo ""
echo "4. Use conda environment activation:"
echo "   conda deactivate && conda activate maxkgnn"