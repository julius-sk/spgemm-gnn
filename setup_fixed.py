from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os

# Ensure torch is available
try:
    import torch
    print(f"Using PyTorch {torch.__version__}")
except ImportError:
    raise ImportError("PyTorch is required but not found. Please install PyTorch first.")

# Get library paths
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
cuda_lib_paths = ['/usr/local/cuda/lib64', '/usr/local/cuda-12.8/lib64']

print(f"PyTorch lib path: {torch_lib_path}")

# Get CUDA architecture
def get_cuda_arch():
    if 'TORCH_CUDA_ARCH_LIST' in os.environ:
        return os.environ['TORCH_CUDA_ARCH_LIST'].split(';')
    
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return [f'{major}.{minor}']
    else:
        return ['7.5', '8.0', '8.6']

cuda_arches = get_cuda_arch()
nvcc_args = ['-O3', '--expt-relaxed-constexpr', '--use_fast_math']

# Add architecture-specific flags
for arch in cuda_arches:
    clean_arch = arch.replace('.', '')
    nvcc_args.extend([f'-gencode=arch=compute_{clean_arch},code=sm_{clean_arch}'])

# Library directories
library_dirs = [torch_lib_path, '/usr/local/cuda/lib64']

# Add existing CUDA paths
for cuda_path in cuda_lib_paths:
    if os.path.exists(cuda_path):
        library_dirs.append(cuda_path)

# RPATH settings to embed library paths
rpath_args = [f'-Wl,-rpath,{torch_lib_path}']
for lib_dir in library_dirs:
    if os.path.exists(lib_dir):
        rpath_args.append(f'-Wl,-rpath,{lib_dir}')

print(f"Library directories: {library_dirs}")
print(f"RPATH arguments: {rpath_args}")

# Define CUDA extension with proper RPATH
maxk_cuda_ext = CUDAExtension(
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
    libraries=['cusparse', 'cublas', 'c10', 'torch', 'torch_cpu', 'torch_cuda'],
    library_dirs=library_dirs,
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17', '-fPIC'],
        'nvcc': nvcc_args
    },
    extra_link_args=['-lcusparse', '-lcublas'] + rpath_args
)

# Custom build extension class
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Set library path during build
        original_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = ':'.join(library_dirs + [original_ld_path])
        
        try:
            super().build_extensions()
        finally:
            # Restore original LD_LIBRARY_PATH
            os.environ['LD_LIBRARY_PATH'] = original_ld_path

# Setup configuration
setup_config = {
    'name': 'maxk_gnn_kernels',
    'version': '1.0.0',
    'author': 'MaxK-GNN Team',
    'description': 'Custom CUDA kernels for MaxK-GNN with fixed linking',
    'long_description': 'High-performance CUDA kernels for MaxK-GNN with proper library linking',
    'packages': find_packages(),
    'ext_modules': [maxk_cuda_ext],
    'cmdclass': {'build_ext': CustomBuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)},
    'zip_safe': False,
    'python_requires': '>=3.7',
    'install_requires': [
        'torch>=1.8.0',
        'numpy',
    ]
}

if __name__ == '__main__':
    setup(**setup_config)