#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "tensor.h"
#include "kernels_cpu.h"
#include "profiler.h"

#ifdef TK_CUDA_ENABLED
#include "kernels_cuda.h"
#endif

namespace py = pybind11;

PYBIND11_MODULE(tensorkernel, m) {
    m.doc() = "TensorKernel: High-performance tensor computation library";

    // ---- DType Enum ----
    py::enum_<tk::DType>(m, "DType")
        .value("Float32", tk::DType::Float32)
        .value("Float64", tk::DType::Float64);

    py::enum_<tk::Layout>(m, "Layout")
        .value("RowMajor", tk::Layout::RowMajor)
        .value("ColMajor", tk::Layout::ColMajor);

    // ---- Tensor Class ----
    py::class_<tk::Tensor>(m, "Tensor")
        .def(py::init<std::vector<int64_t>, tk::DType, tk::Layout>(),
             py::arg("shape"),
             py::arg("dtype") = tk::DType::Float32,
             py::arg("layout") = tk::Layout::RowMajor)
        .def_static("zeros", &tk::Tensor::zeros,
             py::arg("shape"), py::arg("dtype") = tk::DType::Float32)
        .def_static("ones", &tk::Tensor::ones,
             py::arg("shape"), py::arg("dtype") = tk::DType::Float32)
        .def_static("rand", &tk::Tensor::rand,
             py::arg("shape"), py::arg("dtype") = tk::DType::Float32)
        .def("shape", &tk::Tensor::shape)
        .def("numel", &tk::Tensor::numel)
        .def("ndim", &tk::Tensor::ndim)
        .def("nbytes", &tk::Tensor::nbytes)
        .def("fill", &tk::Tensor::fill)
        .def("clone", &tk::Tensor::clone)
        .def("__repr__", &tk::Tensor::to_string)

        // NumPy interop: zero-copy view into tensor memory
        .def("numpy", [](tk::Tensor& t) {
            auto shape = t.shape();
            std::vector<py::ssize_t> np_shape(shape.begin(), shape.end());
            std::vector<py::ssize_t> np_strides;
            for (auto s : t.strides())
                np_strides.push_back(s * sizeof(float));
            return py::array_t<float>(
                np_shape,
                np_strides,
                t.data_f32(),
                py::cast(&t)  // prevent GC of tensor while array is alive
            );
        }, py::return_value_policy::reference_internal)

        // Create tensor from NumPy array (copies data for alignment)
        .def_static("from_numpy", [](py::array_t<float, py::array::c_style> arr) {
            auto buf = arr.request();
            std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());
            return tk::Tensor::from_data(
                static_cast<const float*>(buf.ptr), shape);
        })

#ifdef TK_CUDA_ENABLED
        .def("to_device", &tk::Tensor::to_device)
        .def("to_host", &tk::Tensor::to_host)
        .def("is_on_device", &tk::Tensor::is_on_device)
        .def("free_device", &tk::Tensor::free_device)
#endif
    ;

    // ---- CPU Kernels ----
    auto cpu = m.def_submodule("cpu", "CPU kernel implementations");

    // Matmul variants
    cpu.def("matmul_naive", &tk::cpu::matmul_naive);
    cpu.def("matmul_tiled", &tk::cpu::matmul_tiled,
            py::arg("A"), py::arg("B"), py::arg("C"),
            py::arg("tile_size") = 64);
    cpu.def("matmul_simd", &tk::cpu::matmul_simd);
    cpu.def("matmul_openmp", &tk::cpu::matmul_openmp,
            py::arg("A"), py::arg("B"), py::arg("C"),
            py::arg("tile_size") = 64);

    // Element-wise
    cpu.def("add_naive", &tk::cpu::add_naive);
    cpu.def("add_simd", &tk::cpu::add_simd);
    cpu.def("multiply_naive", &tk::cpu::multiply_naive);
    cpu.def("multiply_simd", &tk::cpu::multiply_simd);

    // Activations
    cpu.def("relu_naive", &tk::cpu::relu_naive);
    cpu.def("relu_simd", &tk::cpu::relu_simd);
    cpu.def("sigmoid_naive", &tk::cpu::sigmoid_naive);
    cpu.def("sigmoid_simd", &tk::cpu::sigmoid_simd);

    // Reductions
    cpu.def("reduce_sum_naive", &tk::cpu::reduce_sum_naive);
    cpu.def("reduce_sum_simd", &tk::cpu::reduce_sum_simd);
    cpu.def("reduce_sum_openmp", &tk::cpu::reduce_sum_openmp);
    cpu.def("reduce_max_naive", &tk::cpu::reduce_max_naive);
    cpu.def("reduce_max_simd", &tk::cpu::reduce_max_simd);

    // Convolution
    cpu.def("conv2d_naive", &tk::cpu::conv2d_naive);
    cpu.def("conv2d_im2col", &tk::cpu::conv2d_im2col);

    // ---- CUDA Kernels ----
#ifdef TK_CUDA_ENABLED
    auto cuda_mod = m.def_submodule("cuda", "CUDA kernel implementations");

    cuda_mod.def("matmul", [](tk::Tensor& A, tk::Tensor& B, tk::Tensor& C) {
        int M = static_cast<int>(A.shape()[0]);
        int K = static_cast<int>(A.shape()[1]);
        int N = static_cast<int>(B.shape()[1]);
        tk::cuda::matmul(A.device_data(), B.device_data(), C.device_data(),
                         M, K, N);
    });
    cuda_mod.def("relu", [](tk::Tensor& X) {
        tk::cuda::relu(X.device_data(), static_cast<int>(X.numel()));
    });
    cuda_mod.def("sigmoid", [](tk::Tensor& X) {
        tk::cuda::sigmoid(X.device_data(), static_cast<int>(X.numel()));
    });
    cuda_mod.def("reduce_sum", [](tk::Tensor& X) {
        return tk::cuda::reduce_sum(X.device_data(), static_cast<int>(X.numel()));
    });
    cuda_mod.def("synchronize", &tk::cuda::device_synchronize);
#endif

    // ---- Profiler ----
    py::class_<tk::ProfileResult>(m, "ProfileResult")
        .def_readonly("kernel_name", &tk::ProfileResult::kernel_name)
        .def_readonly("variant", &tk::ProfileResult::variant)
        .def_readonly("elapsed_ms", &tk::ProfileResult::elapsed_ms)
        .def_readonly("gflops", &tk::ProfileResult::gflops)
        .def_readonly("bandwidth_gb_s", &tk::ProfileResult::bandwidth_gb_s)
        .def_readonly("speedup", &tk::ProfileResult::speedup);

    py::class_<tk::Profiler>(m, "Profiler")
        .def(py::init<>())
        .def("profile", &tk::Profiler::profile,
             py::arg("name"), py::arg("variant"),
             py::arg("flop_count"), py::arg("bytes_moved"),
             py::arg("kernel"), py::arg("warmup") = 3, py::arg("trials") = 10)
        .def("add_result", &tk::Profiler::add_result)
        .def("export_csv", &tk::Profiler::export_csv)
        .def("print_summary", &tk::Profiler::print_summary);
}
