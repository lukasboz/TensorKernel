#include "tensor.h"

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

#ifdef TK_CUDA_ENABLED
#include "kernels_cuda.h"
#endif

#include <numeric>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace tk {

// ============================================================
// Aligned Memory Allocation
// ============================================================

void* aligned_alloc_impl(size_t alignment, size_t size) {
    if (size == 0) size = alignment;
#ifdef _WIN32
    void* ptr = _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        ptr = nullptr;
#endif
    if (!ptr)
        throw std::bad_alloc();
    return ptr;
}

void aligned_free_impl(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================
// Tensor Construction / Destruction
// ============================================================

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Layout layout)
    : shape_(std::move(shape)), dtype_(dtype), layout_(layout) {
    compute_strides();
    size_t bytes = nbytes();
    data_.reset(aligned_alloc_impl(32, bytes));
    std::memset(data_.get(), 0, bytes);
}

Tensor::~Tensor() {
#ifdef TK_CUDA_ENABLED
    free_device();
#endif
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      dtype_(other.dtype_),
      layout_(other.layout_) {
#ifdef TK_CUDA_ENABLED
    device_data_ = other.device_data_;
    other.device_data_ = nullptr;
#endif
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
#ifdef TK_CUDA_ENABLED
        free_device();
        device_data_ = other.device_data_;
        other.device_data_ = nullptr;
#endif
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        layout_ = other.layout_;
    }
    return *this;
}

Tensor Tensor::clone() const {
    Tensor t(shape_, dtype_, layout_);
    std::memcpy(t.data_.get(), data_.get(), nbytes());
    return t;
}

// ============================================================
// Factory Methods
// ============================================================

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype) {
    return Tensor(std::move(shape), dtype);
}

Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype) {
    Tensor t(std::move(shape), dtype);
    if (dtype == DType::Float32) {
        float* d = t.data_f32();
        for (int64_t i = 0; i < t.numel(); ++i)
            d[i] = 1.0f;
    } else {
        double* d = t.data_f64();
        for (int64_t i = 0; i < t.numel(); ++i)
            d[i] = 1.0;
    }
    return t;
}

Tensor Tensor::rand(std::vector<int64_t> shape, DType dtype) {
    Tensor t(std::move(shape), dtype);
    static thread_local std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dtype == DType::Float32) {
        float* d = t.data_f32();
        for (int64_t i = 0; i < t.numel(); ++i)
            d[i] = dist(gen);
    } else {
        std::uniform_real_distribution<double> dist64(0.0, 1.0);
        double* d = t.data_f64();
        for (int64_t i = 0; i < t.numel(); ++i)
            d[i] = dist64(gen);
    }
    return t;
}

Tensor Tensor::from_data(const float* data, std::vector<int64_t> shape) {
    Tensor t(shape, DType::Float32);
    std::memcpy(t.data_f32(), data, t.nbytes());
    return t;
}

// ============================================================
// Data Access
// ============================================================

float* Tensor::data_f32() {
    return static_cast<float*>(data_.get());
}

const float* Tensor::data_f32() const {
    return static_cast<const float*>(data_.get());
}

double* Tensor::data_f64() {
    return static_cast<double*>(data_.get());
}

const double* Tensor::data_f64() const {
    return static_cast<const double*>(data_.get());
}

// ============================================================
// Metadata
// ============================================================

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    int64_t n = 1;
    for (auto s : shape_) n *= s;
    return n;
}

size_t Tensor::nbytes() const {
    return static_cast<size_t>(numel()) * element_size();
}

size_t Tensor::element_size() const {
    switch (dtype_) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        default: return sizeof(float);
    }
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;

    if (layout_ == Layout::RowMajor) {
        int64_t stride = 1;
        for (int i = ndim() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    } else {
        int64_t stride = 1;
        for (int i = 0; i < ndim(); ++i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }
}

// ============================================================
// Element Access
// ============================================================

float& Tensor::at(std::initializer_list<int64_t> indices) {
    auto it = indices.begin();
    int64_t offset = 0;
    for (int i = 0; i < ndim(); ++i) {
        int64_t idx = *(it + i);
        if (idx < 0 || idx >= shape_[i])
            throw std::out_of_range("Tensor index out of range");
        offset += idx * strides_[i];
    }
    return data_f32()[offset];
}

const float& Tensor::at(std::initializer_list<int64_t> indices) const {
    auto it = indices.begin();
    int64_t offset = 0;
    for (int i = 0; i < ndim(); ++i) {
        int64_t idx = *(it + i);
        if (idx < 0 || idx >= shape_[i])
            throw std::out_of_range("Tensor index out of range");
        offset += idx * strides_[i];
    }
    return data_f32()[offset];
}

void Tensor::fill(float value) {
    float* d = data_f32();
    for (int64_t i = 0; i < numel(); ++i)
        d[i] = value;
}

// ============================================================
// Debug
// ============================================================

std::string Tensor::to_string() const {
    std::ostringstream ss;
    ss << "Tensor(shape=[";
    for (int i = 0; i < ndim(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[i];
    }
    ss << "], dtype=";
    ss << (dtype_ == DType::Float32 ? "float32" : "float64");
    ss << ", layout=";
    ss << (layout_ == Layout::RowMajor ? "row_major" : "col_major");
    ss << ")";

    // Print first few elements for small tensors
    if (numel() <= 16 && dtype_ == DType::Float32) {
        ss << "\n  data: [";
        const float* d = data_f32();
        for (int64_t i = 0; i < numel(); ++i) {
            if (i > 0) ss << ", ";
            ss << std::fixed << std::setprecision(4) << d[i];
        }
        ss << "]";
    }
    return ss.str();
}

// ============================================================
// CUDA Device Memory
// ============================================================

#ifdef TK_CUDA_ENABLED
void Tensor::to_device() {
    if (device_data_) return; // already on device
    cuda::device_malloc(&device_data_, static_cast<size_t>(numel()));
    cuda::host_to_device(device_data_, data_f32(), static_cast<size_t>(numel()));
}

void Tensor::to_host() {
    if (!device_data_)
        throw std::runtime_error("Tensor has no device data to copy back");
    cuda::device_to_host(data_f32(), device_data_, static_cast<size_t>(numel()));
}

float* Tensor::device_data() {
    return device_data_;
}

const float* Tensor::device_data() const {
    return device_data_;
}

void Tensor::free_device() {
    if (device_data_) {
        cuda::device_free(device_data_);
        device_data_ = nullptr;
    }
}
#endif

} // namespace tk
