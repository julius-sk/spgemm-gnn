from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os

# Get CUDA architecture from environment or detect automatically
def get_cuda_arch():
    if 'TORCH_CUDA_ARCH_LIST' in os.environ:
        return os.environ['TORCH_CUDA_ARCH_LIST'].split(';')
    
    # Try to detect GPU architecture
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return [f'{major}.{minor}']
    else:
        # Default to common architectures
        return ['7.5', '8.0', '8.6']

cuda_arches = get_cuda_arch()
nvcc_args = ['-O3', '--expt-relaxed-constexpr'] + [f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}' for arch in cuda_arches]

# Define CUDA extension for MaxK kernels
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
    libraries=['cusparse', 'cublas'],
    library_dirs=['/usr/local/cuda/lib64'],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17', '-fPIC'],
        'nvcc': nvcc_args
    },
    extra_link_args=['-lcusparse', '-lcublas']
)

# Setup using setuptools
setup(
    name='maxk_gnn_kernels',
    version='1.0.0',
    author='MaxK-GNN Team',
    description='Custom CUDA kernels for MaxK-GNN',
    packages=find_packages(),
    ext_modules=[maxk_cuda_ext],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
        'dgl',
        'numpy',
    ]
)