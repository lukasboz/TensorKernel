#include "profiler.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>

namespace tk {

double Profiler::measure_ms(std::function<void()> fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

ProfileResult Profiler::profile(const std::string& name,
                                 const std::string& variant,
                                 int64_t flop_count,
                                 int64_t bytes_moved,
                                 std::function<void()> kernel,
                                 int warmup, int trials) {
    // Warmup: populate caches, trigger JIT/turbo boost
    for (int i = 0; i < warmup; ++i)
        kernel();

    // Timed trials
    std::vector<double> times;
    times.reserve(trials);
    for (int i = 0; i < trials; ++i)
        times.push_back(measure_ms(kernel));

    // Use median (robust against outliers from context switches)
    std::sort(times.begin(), times.end());
    double median_ms = times[trials / 2];

    ProfileResult r;
    r.kernel_name = name;
    r.variant = variant;
    r.elapsed_ms = median_ms;
    r.gflops = (median_ms > 0.0)
        ? (static_cast<double>(flop_count) / 1e9) / (median_ms / 1e3)
        : 0.0;
    r.bandwidth_gb_s = (median_ms > 0.0)
        ? (static_cast<double>(bytes_moved) / 1e9) / (median_ms / 1e3)
        : 0.0;
    return r;
}

void Profiler::add_result(const ProfileResult& r) {
    results_.push_back(r);
}

void Profiler::export_csv(const std::string& filename) const {
    std::ofstream f(filename);
    f << "kernel,variant,M,N,K,elapsed_ms,gflops,bandwidth_gb_s,speedup\n";
    for (const auto& r : results_) {
        f << r.kernel_name << ","
          << r.variant << ","
          << r.M << "," << r.N << "," << r.K << ","
          << std::fixed << std::setprecision(4)
          << r.elapsed_ms << ","
          << r.gflops << ","
          << r.bandwidth_gb_s << ","
          << r.speedup << "\n";
    }
}

void Profiler::print_summary() const {
    std::cout << "\n"
              << std::left
              << std::setw(14) << "Kernel"
              << std::setw(10) << "Variant"
              << std::setw(12) << "Size"
              << std::right
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(14) << "BW (GB/s)"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(84, '-') << "\n";

    for (const auto& r : results_) {
        std::string size_str;
        if (r.M > 0 && r.N > 0 && r.K > 0)
            size_str = std::to_string(r.M) + "x" + std::to_string(r.K) + "x" + std::to_string(r.N);
        else if (r.N > 0)
            size_str = std::to_string(r.N);

        std::cout << std::left
                  << std::setw(14) << r.kernel_name
                  << std::setw(10) << r.variant
                  << std::setw(12) << size_str
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.elapsed_ms
                  << std::setw(12) << std::setprecision(2) << r.gflops
                  << std::setw(14) << r.bandwidth_gb_s
                  << std::setw(10) << std::setprecision(1) << r.speedup
                  << "\n";
    }
    std::cout << std::string(84, '-') << "\n";
}

std::vector<ProfileResult> Profiler::get_results(const std::string& name) const {
    std::vector<ProfileResult> filtered;
    for (const auto& r : results_)
        if (r.kernel_name == name)
            filtered.push_back(r);
    return filtered;
}

// RAII Timer
ScopedTimer::ScopedTimer(const std::string& label)
    : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    std::cout << "[Timer] " << label_ << ": " << std::fixed
              << std::setprecision(3) << ms << " ms\n";
}

// Theoretical peak GFLOPS estimate
double estimate_cpu_peak_gflops(int num_cores, double clock_ghz,
                                 int simd_width, int fma_ports) {
    // FMA = 2 FLOP (multiply + add fused)
    // Peak = cores * clock * simd_width * 2 (FMA) * fma_ports
    return num_cores * clock_ghz * simd_width * 2.0 * fma_ports;
}

} // namespace tk
