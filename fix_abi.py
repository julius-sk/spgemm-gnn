#!/usr/bin/env python3
"""
Fix C++ ABI compatibility issues
"""

import os
import sys
import subprocess
import glob

def check_abi_versions():
    """Check available C++ ABI versions"""
    print("🔍 Checking C++ ABI versions...")
    
    # Check system libstdc++
    system_paths = ['/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu', '/usr/lib64']
    conda_lib = os.path.join(os.path.dirname(sys.executable), '..', 'lib')
    
    paths_to_check = [conda_lib] + system_paths
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"\n📁 Checking {path}:")
            
            # Find libstdc++ files
            libstdcpp_files = []
            try:
                for file in os.listdir(path):
                    if 'libstdc++' in file and '.so' in file:
                        libstdcpp_files.append(os.path.join(path, file))
            except PermissionError:
                print(f"  ❌ Permission denied")
                continue
            
            if libstdcpp_files:
                for lib_file in libstdcpp_files:
                    print(f"  📄 {os.path.basename(lib_file)}")
                    
                    # Check GLIBCXX versions
                    try:
                        result = subprocess.run(['strings', lib_file], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            glibcxx_versions = [line for line in result.stdout.split('\n') 
                                              if line.startswith('GLIBCXX_')]
                            if glibcxx_versions:
                                latest_version = sorted(glibcxx_versions)[-1]
                                print(f"    Latest GLIBCXX: {latest_version}")
                                
                                # Check if we have the required version
                                if 'GLIBCXX_3.4.32' in glibcxx_versions:
                                    print(f"    ✅ Has required GLIBCXX_3.4.32")
                                else:
                                    print(f"    ❌ Missing GLIBCXX_3.4.32")
                    except Exception as e:
                        print(f"    ⚠️  Could not check versions: {e}")
            else:
                print(f"  No libstdc++ files found")

def rebuild_with_older_abi():
    """Rebuild extension with older C++ ABI"""
    print("\n🔨 Rebuilding with older C++ ABI...")
    
    # Create setup script with explicit ABI settings
    setup_content = '''
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os

# Force older ABI for compatibility
os.environ['TORCH_CXX_FLAGS'] = '-D_GLIBCXX_USE_CXX11_ABI=0'

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

ext = CUDAExtension(
    name='maxk_kernels',
    sources=[
        'kernels/maxk_bindings.cpp',
        'kernels/maxk_cuda_kernels.cu',
        'kernels/spmm_maxk.cu', 
        'kernels/spmm_maxk_backward.cu',
        'kernels/spmm_cusparse.cu'
    ],
    include_dirs=[
        'kernels/',
        '/usr/local/cuda/include',
    ],
    library_dirs=[
        torch_lib_path,
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12.8/lib64'
    ],
    libraries=['cusparse', 'cublas'],
    extra_compile_args={
        'cxx': [
            '-O3', '-std=c++14',  # Use C++14 instead of C++17
            '-fPIC',
            '-D_GLIBCXX_USE_CXX11_ABI=0'  # Force old ABI
        ],
        'nvcc': [
            '-O3', '--expt-relaxed-constexpr',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-D_GLIBCXX_USE_CXX11_ABI=0'  # Force old ABI for NVCC too
        ]
    },
    extra_link_args=[
        f'-Wl,-rpath,{torch_lib_path}',
        '-D_GLIBCXX_USE_CXX11_ABI=0'
    ]
)

setup(
    name='maxk_kernels_compat',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
'''
    
    with open('setup_compat.py', 'w') as f:
        f.write(setup_content)
    
    print("✅ Created setup_compat.py with ABI compatibility settings")
    
    # Clean old build
    print("🧹 Cleaning old build...")
    old_files = glob.glob('maxk_kernels*.so') + ['build']
    for item in old_files:
        if os.path.isfile(item):
            os.remove(item)
            print(f"  Removed {item}")
        elif os.path.isdir(item):
            import shutil
            shutil.rmtree(item)
            print(f"  Removed directory {item}")
    
    # Set environment variables for build
    env = os.environ.copy()
    env['TORCH_CXX_FLAGS'] = '-D_GLIBCXX_USE_CXX11_ABI=0'
    env['CXXFLAGS'] = '-D_GLIBCXX_USE_CXX11_ABI=0'
    
    # Build with compatibility settings
    print("🔨 Building with ABI compatibility...")
    try:
        result = subprocess.run([
            sys.executable, 'setup_compat.py', 'build_ext', '--inplace'
        ], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Build successful!")
            print("Output:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("❌ Build failed:")
            print("Error:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def use_conda_compiler():
    """Try using conda's compiler toolchain"""
    print("\n🔧 Trying conda compiler toolchain...")
    
    # Check if conda has compiler packages
    try:
        result = subprocess.run(['conda', 'list', 'gcc'], capture_output=True, text=True)
        if 'gcc' not in result.stdout:
            print("Installing conda compiler toolchain...")
            subprocess.run(['conda', 'install', '-y', 'gcc_linux-64', 'gxx_linux-64'], 
                         capture_output=True)
    except Exception as e:
        print(f"Could not install conda compilers: {e}")
    
    # Try building with conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        env = os.environ.copy()
        env['CC'] = os.path.join(conda_prefix, 'bin', 'gcc')
        env['CXX'] = os.path.join(conda_prefix, 'bin', 'g++')
        env['TORCH_CXX_FLAGS'] = '-D_GLIBCXX_USE_CXX11_ABI=0'
        
        print(f"Using conda compilers from {conda_prefix}")
        
        try:
            result = subprocess.run([
                sys.executable, 'setup_compat.py', 'build_ext', '--inplace'
            ], env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Conda build successful!")
                return True
            else:
                print("❌ Conda build failed")
        except Exception as e:
            print(f"Conda build error: {e}")
    
    return False

def try_prebuilt_extension():
    """Try to find a prebuilt extension or use fallback"""
    print("\n📦 Looking for alternative solutions...")
    
    # Option 1: Check if PyTorch was built with old ABI
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CXX ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}")
        
        if torch._C._GLIBCXX_USE_CXX11_ABI == False:
            print("✅ PyTorch uses old ABI - should be compatible")
        else:
            print("⚠️  PyTorch uses new ABI - this might be the issue")
    except:
        print("Could not determine PyTorch ABI")
    
    # Option 2: Suggest conda-forge PyTorch
    print("\n💡 Alternative solutions:")
    print("1. Use conda-forge PyTorch (often more compatible):")
    print("   conda install pytorch cpuonly -c conda-forge")
    print("   # or for CUDA:")
    print("   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia")
    
    print("\n2. Use the fallback mode (no custom kernels):")
    print("   python maxk_gnn_integrated.py --dataset reddit --model sage --maxk 32")
    print("   # This will use DGL's built-in operations")
    
    print("\n3. Install older GCC in conda:")
    print("   conda install gcc=9 gxx=9")

def test_import():
    """Test if import works after fixes"""
    print("\n🧪 Testing import...")
    
    test_script = '''
import sys
sys.path.insert(0, '.')

try:
    import maxk_kernels
    print("✅ Import successful!")
    print("Available functions:", [f for f in dir(maxk_kernels) if not f.startswith('_')])
    
    # Test basic functionality
    import torch
    if torch.cuda.is_available():
        x = torch.randn(4, 16, device='cuda')
        if hasattr(maxk_kernels, 'maxk_forward'):
            try:
                output = maxk_kernels.maxk_forward(x, 8)
                print(f"✅ MaxK test passed: {x.shape} -> {output.shape}")
            except Exception as e:
                print(f"⚠️ MaxK test failed: {e}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)
'''
    
    try:
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Test error: {e}")
        return False

def main():
    """Main ABI fixing routine"""
    print("🔧 MaxK-GNN C++ ABI Compatibility Fixer")
    print("=" * 50)
    
    print("The error indicates your extension was compiled with a newer")
    print("C++ standard library than what's available in your conda environment.")
    print()
    
    # Check current ABI versions
    check_abi_versions()
    
    # Try different solutions
    solutions = [
        ("Rebuild with older ABI", rebuild_with_older_abi),
        ("Use conda compiler", use_conda_compiler),
    ]
    
    for solution_name, solution_func in solutions:
        print(f"\n--- Trying: {solution_name} ---")
        try:
            if solution_func():
                print(f"✅ {solution_name} completed")
                if test_import():
                    print("🎉 Problem solved!")
                    return True
                else:
                    print("Import still fails, trying next solution...")
            else:
                print(f"❌ {solution_name} failed")
        except Exception as e:
            print(f"❌ {solution_name} error: {e}")
    
    # Show fallback options
    try_prebuilt_extension()
    
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 If all else fails, the training script will work with DGL fallback:")
        print("python maxk_gnn_integrated.py --dataset reddit --model sage --maxk 32")