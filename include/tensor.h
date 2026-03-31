#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <initializer_list>
#include <stdexcept>
#include <cstring>
#include <random>

namespace tk {

enum class DType { Float32, Float64 };
enum class Layout { RowMajor, ColMajor };

// Platform-specific aligned allocation (32-byte for AVX2)
void* aligned_alloc_impl(size_t alignment, size_t size);
void  aligned_free_impl(void* ptr);

struct AlignedDeleter {
    void operator()(void* p) const { aligned_free_impl(p); }
};

class Tensor {
public:
    Tensor(std::vector<int64_t> shape, DType dtype = DType::Float32,
           Layout layout = Layout::RowMajor);
    ~Tensor();

    // Move-only semantics
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Deep copy
    Tensor clone() const;

    // Factory methods
    static Tensor zeros(std::vector<int64_t> shape, DType dtype = DType::Float32);
    static Tensor ones(std::vector<int64_t> shape, DType dtype = DType::Float32);
    static Tensor rand(std::vector<int64_t> shape, DType dtype = DType::Float32);
    static Tensor from_data(const float* data, std::vector<int64_t> shape);

    // Data access
    float*       data_f32();
    const float* data_f32() const;
    double*      data_f64();
    const double* data_f64() const;

    // Metadata
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t numel() const;
    size_t  nbytes() const;
    DType   dtype() const { return dtype_; }
    Layout  layout() const { return layout_; }
    int     ndim() const { return static_cast<int>(shape_.size()); }

    // Bounds-checked element access
    float& at(std::initializer_list<int64_t> indices);
    const float& at(std::initializer_list<int64_t> indices) const;

    // Fill
    void fill(float value);

    // Debug
    std::string to_string() const;

    // CUDA device memory
#ifdef TK_CUDA_ENABLED
    void to_device();
    void to_host();
    float* device_data();
    const float* device_data() const;
    bool is_on_device() const { return device_data_ != nullptr; }
    void free_device();
#endif

private:
    std::unique_ptr<void, AlignedDeleter> data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_;
    Layout layout_;

#ifdef TK_CUDA_ENABLED
    float* device_data_ = nullptr;
#endif

    void compute_strides();
    size_t element_size() const;
};

} // namespace tk
