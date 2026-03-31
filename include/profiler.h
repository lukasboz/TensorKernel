#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <functional>

namespace tk {

struct ProfileResult {
    std::string kernel_name;
    std::string variant;
    int64_t     M = 0, N = 0, K = 0;
    double      elapsed_ms   = 0.0;
    double      gflops       = 0.0;
    double      bandwidth_gb_s = 0.0;
    double      speedup      = 1.0;
};

class Profiler {
public:
    // Time a kernel: runs warmup iterations, then timed trials.
    // Returns median time to reject OS scheduling outliers.
    // flop_count: total floating-point operations (e.g., 2*M*N*K for matmul)
    // bytes_moved: total memory traffic in bytes
    ProfileResult profile(const std::string& name,
                          const std::string& variant,
                          int64_t flop_count,
                          int64_t bytes_moved,
                          std::function<void()> kernel,
                          int warmup = 3,
                          int trials = 10);

    void add_result(const ProfileResult& r);
    void export_csv(const std::string& filename) const;
    void print_summary() const;
    std::vector<ProfileResult> get_results(const std::string& name) const;
    const std::vector<ProfileResult>& all_results() const { return results_; }

private:
    std::vector<ProfileResult> results_;
    static double measure_ms(std::function<void()> fn);
};

// RAII timer for quick ad-hoc measurements
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& label);
    ~ScopedTimer();
private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

// Estimate theoretical CPU peak GFLOPS
// Assumes FMA = 2 FLOP, simd_width floats per vector register
double estimate_cpu_peak_gflops(int num_cores, double clock_ghz,
                                 int simd_width = 8, int fma_ports = 2);

} // namespace tk
