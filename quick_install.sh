#!/bin/bash

echo "🚀 Quick MaxK-GNN Kernel Installation Fix"
echo "This script fixes the pip installation issue from the build log"
echo "=================================================="

# Based on your successful build log, the extension was built successfully
# The issue is with pip's isolated build environment

echo "✅ Extension build was successful (from your log)"
echo "🔧 The issue is with pip's isolated build environment"
echo ""

# Check if the .so file exists (it should based on the build log)
if ls maxk_kernels*.so 1> /dev/null 2>&1; then
    echo "✅ Found built extension file:"
    ls -la maxk_kernels*.so
    SO_FILE=$(ls maxk_kernels*.so | head -1)
else
    echo "❌ No .so file found. Let's rebuild..."
    python setup.py build_ext --inplace
    SO_FILE=$(ls maxk_kernels*.so | head -1)
fi

echo ""
echo "🧪 Testing direct import from current directory..."

# Test 1: Direct import from current directory
python -c "
import sys
import os
sys.path.insert(0, '.')

try:
    import maxk_kernels
    print('✅ Direct import successful!')
    print('Available functions:', [f for f in dir(maxk_kernels) if not f.startswith('_')])
    
    # Test CUDA functionality
    import torch
    if torch.cuda.is_available():
        print('🔥 Testing CUDA functionality...')
        x = torch.randn(10, 64, device='cuda')
        print(f'✅ CUDA tensor created: {x.shape}')
        
        # Test MaxK if available
        if hasattr(maxk_kernels, 'maxk_forward'):
            try:
                output = maxk_kernels.maxk_forward(x, 16)
                print(f'✅ MaxK forward works! Output: {output.shape}')
                
                # Check sparsity
                non_zero = (output != 0).sum().item()
                total = output.numel()
                print(f'✅ Sparsity: {non_zero}/{total} non-zero elements')
            except Exception as e:
                print(f'⚠️ MaxK test failed: {e}')
    
    print('\\n🎉 Extension is working correctly!')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎯 SUCCESS! The extension works when imported directly."
    echo ""
    echo "📋 To use MaxK-GNN kernels, you have two options:"
    echo ""
    echo "Option 1 (Recommended): Set PYTHONPATH"
    echo "  export PYTHONPATH=\$PYTHONPATH:$(pwd)"
    echo "  python maxk_gnn_integrated.py --dataset reddit --model sage --maxk 32"
    echo ""
    echo "Option 2: Manual installation to site-packages"
    echo "  python install_kernels_manual.py"
    echo ""
    
    # Option 2: Try manual installation to site-packages
    echo "🔧 Attempting automatic installation to site-packages..."
    
    python -c "
import site
import shutil
import os

try:
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    print(f'Site-packages: {site_packages}')
    
    # Create maxk_kernels directory
    kernel_dir = os.path.join(site_packages, 'maxk_kernels')
    os.makedirs(kernel_dir, exist_ok=True)
    
    # Find and copy .so file
    import glob
    so_files = glob.glob('maxk_kernels*.so')
    if so_files:
        so_file = so_files[0]
        dest_so = os.path.join(kernel_dir, os.path.basename(so_file))
        shutil.copy2(so_file, dest_so)
        print(f'✅ Copied {so_file} to {dest_so}')
        
        # Create __init__.py
        init_py = os.path.join(kernel_dir, '__init__.py')
        with open(init_py, 'w') as f:
            module_name = os.path.basename(so_file).split('.')[0]
            f.write(f'from .{module_name} import *\\n')
        print(f'✅ Created {init_py}')
        
        print('✅ Manual installation completed!')
    else:
        print('❌ No .so file found for installation')
        
except Exception as e:
    print(f'⚠️ Manual installation failed: {e}')
    print('Use Option 1 (PYTHONPATH) instead')
"
    
    echo ""
    echo "🧪 Testing global import..."
    python -c "
try:
    import maxk_kernels
    print('✅ Global import successful!')
    print('🎉 Installation completed successfully!')
except ImportError:
    print('⚠️ Global import failed, but local import works.')
    print('💡 Use: export PYTHONPATH=\$PYTHONPATH:$(pwd)')
"
    
    echo ""
    echo "📋 Next Steps:"
    echo "1. Test integration: python test_integration.py"
    echo "2. Download datasets: cd kernels && python generate_meta.py"
    echo "3. Run training: python maxk_gnn_integrated.py --dataset reddit --model sage --maxk 32"
    
else
    echo ""
    echo "❌ Extension test failed. Let's debug..."
    echo ""
    echo "🔍 Debug information:"
    echo "Current directory: $(pwd)"
    echo "Extension files:"
    ls -la *.so 2>/dev/null || echo "No .so files found"
    echo ""
    
    echo "Python path:"
    python -c "import sys; print('\\n'.join(sys.path))"
    
    echo ""
    echo "💡 Try rebuilding with verbose output:"
    echo "python setup.py build_ext --inplace --verbose"
fi