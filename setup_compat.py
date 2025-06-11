
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
