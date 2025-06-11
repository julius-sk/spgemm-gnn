#!/usr/bin/env python3
"""
Troubleshooting script for MaxK-GNN setup issues
"""

import sys
import os
import subprocess
import importlib.util

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_cuda():
    """Check CUDA installation"""
    print("\n🔧 CUDA Installation Check")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            if version_line:
                print(f"✅ NVCC found: {version_line[0].strip()}")
            else:
                print("✅ NVCC found but version unclear")
            nvcc_ok = True
        else:
            print("❌ NVCC not found in PATH")
            nvcc_ok = False
    except FileNotFoundError:
        print("❌ NVCC not found in PATH")
        nvcc_ok = False
    
    # Check CUDA runtime
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver found")
            # Extract CUDA version from nvidia-smi
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"✅ CUDA Runtime: {cuda_version}")
                    break
        else:
            print("❌ nvidia-smi failed")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    return nvcc_ok

def check_pytorch():
    """Check PyTorch installation"""
    print("\n🔥 PyTorch Installation Check")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA support
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                print(f"  GPU {i}: {gpu_name} (compute {capability[0]}.{capability[1]})")
                
                if capability[0] < 7:
                    print(f"  ⚠️  GPU {i} compute capability < 7.0 may not be fully supported")
        else:
            print("❌ PyTorch CUDA not available")
            return False
            
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check other dependencies"""
    print("\n📦 Dependencies Check")
    
    dependencies = {
        'dgl': 'DGL',
        'numpy': 'NumPy', 
        'setuptools': 'Setuptools',
        'pybind11': 'pybind11'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            if hasattr(mod, '__version__'):
                print(f"✅ {name}: {mod.__version__}")
            else:
                print(f"✅ {name}: installed")
        except ImportError:
            print(f"❌ {name}: not installed")
            all_ok = False
    
    return all_ok

def check_build_tools():
    """Check build tools"""
    print("\n🛠️  Build Tools Check")
    
    tools = ['gcc', 'g++', 'cmake', 'make']
    all_ok = True
    
    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ {tool}: {version_line}")
            else:
                print(f"❌ {tool}: not working properly")
                all_ok = False
        except FileNotFoundError:
            print(f"❌ {tool}: not found")
            all_ok = False
    
    return all_ok

def check_torch_extension():
    """Check torch.utils.cpp_extension"""
    print("\n🔧 PyTorch Extension Check")
    
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        print("✅ torch.utils.cpp_extension imports OK")
        
        # Check if we can create a dummy extension
        try:
            dummy_ext = CUDAExtension(
                name='dummy',
                sources=[],
                include_dirs=[],
                libraries=[],
                library_dirs=[]
            )
            print("✅ CUDAExtension creation OK")
        except Exception as e:
            print(f"❌ CUDAExtension creation failed: {e}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ torch.utils.cpp_extension import failed: {e}")
        return False

def check_file_structure():
    """Check if required files exist"""
    print("\n📁 File Structure Check")
    
    required_files = [
        'setup.py',
        'kernels/maxk_bindings.cpp',
        'kernels/maxk_cuda_kernels.cu',
        'kernels/spmm_maxk.cu',
        'kernels/spmm_maxk_backward.cu',
        'kernels/spmm_maxk.h',
        'kernels/spmm_maxk_backward.h',
        'kernels/spmm_base.h',
        'kernels/CMakeLists.txt'
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            all_ok = False
    
    return all_ok

def suggest_fixes():
    """Suggest common fixes"""
    print("\n💡 Common Solutions")
    print("=" * 50)
    
    print("1. Install missing dependencies:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("   conda install -c dglteam/label/cu121 dgl")
    print("   pip install pybind11")
    
    print("\n2. Fix CUDA issues:")
    print("   export PATH=/usr/local/cuda/bin:$PATH")
    print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    print("   export CUDA_HOME=/usr/local/cuda")
    
    print("\n3. Fix build issues:")
    print("   export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6'")
    print("   rm -rf build/ && python setup.py clean --all")
    print("   python setup.py build_ext --inplace")
    
    print("\n4. Alternative installation:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html")
    
    print("\n5. If compute capability < 8.0:")
    print("   export TORCH_CUDA_ARCH_LIST='7.0;7.5'  # Adjust for your GPU")

def main():
    """Main troubleshooting routine"""
    print("🔍 MaxK-GNN Troubleshooting Script")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA Installation", check_cuda),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("Build Tools", check_build_tools),
        ("PyTorch Extensions", check_torch_extension),
        ("File Structure", check_file_structure)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📊 SUMMARY")
    print(f"{'=' * 50}")
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed < len(results):
        suggest_fixes()
    else:
        print("\n🎉 All checks passed! You should be able to build MaxK-GNN kernels.")
        print("Run: ./build_kernels.sh")

if __name__ == "__main__":
    main()