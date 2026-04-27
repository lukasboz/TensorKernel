from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2"]
    # OpenMP support on MSVC if available
    extra_compile_args.append("/openmp")
else:
    extra_compile_args = ["-std=c++17", "-O3", "-march=native"]
    extra_compile_args.extend(["-fopenmp"])
    extra_link_args.extend(["-fopenmp"])

ext_modules = [
    Pybind11Extension(
        "tensorkernel",
        [
            "bindings/pybind_module.cpp",
            "src/tensor.cpp",
            "src/kernels_cpu.cpp",
            "src/profiler.cpp",
        ],
        include_dirs=["include"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]

setup(
    name="tensorkernel",
    version="0.1.0",
    description="TensorKernel Python bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
