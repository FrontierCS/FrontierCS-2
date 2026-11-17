#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "protocol.hpp"

namespace fk = frontier_kami;

namespace {

constexpr const char *kEntry = "frontier_batched_block_gemm_entry";
constexpr const char *kExactGpuName = "NVIDIA H100 80GB HBM3";

void check_cuda(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS) return;
    const char *name = nullptr;
    const char *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::ostringstream out;
    out << operation << " failed: " << (name ? name : "CUDA_ERROR")
        << ": " << (message ? message : "unknown");
    throw std::runtime_error(out.str());
}

struct Options {
    std::string candidate;
    std::string reference;
    int m = 0;
    int n = 0;
    int k = 0;
    int active_ranks = 0;
    int allocated_ranks = 0;
    int batch = 0;
    int profile = 0;
    int repetitions = 1;
    std::uint64_t seed = 0;
};

std::string next_value(int &index, int argc, char **argv) {
    if (++index >= argc) throw std::runtime_error("missing command-line value");
    return argv[index];
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--candidate") options.candidate = next_value(index, argc, argv);
        else if (key == "--reference") options.reference = next_value(index, argc, argv);
        else if (key == "--m") options.m = std::stoi(next_value(index, argc, argv));
        else if (key == "--n") options.n = std::stoi(next_value(index, argc, argv));
        else if (key == "--k") options.k = std::stoi(next_value(index, argc, argv));
        else if (key == "--active-ranks") options.active_ranks = std::stoi(next_value(index, argc, argv));
        else if (key == "--allocated-ranks") options.allocated_ranks = std::stoi(next_value(index, argc, argv));
        else if (key == "--batch") options.batch = std::stoi(next_value(index, argc, argv));
        else if (key == "--profile") options.profile = std::stoi(next_value(index, argc, argv));
        else if (key == "--repetitions") options.repetitions = std::stoi(next_value(index, argc, argv));
        else if (key == "--seed") options.seed = std::stoull(next_value(index, argc, argv));
        else throw std::runtime_error("unknown command-line argument: " + key);
    }
    if (options.candidate.empty()) throw std::runtime_error("--candidate is required");
    if (options.batch != fk::approved_batch(options.m, options.n, options.k)) {
        throw std::runtime_error("batch does not match the public shape-adaptive table");
    }
    if (options.batch < fk::kMinimumBatch) {
        throw std::runtime_error("batch is below the public scheduling floor");
    }
    if (fk::device_bytes(options.m, options.n, options.k, options.batch) > fk::kOneGiB
        || fk::host_bytes(options.m, options.n, options.k, options.batch) > fk::kOneGiB) {
        throw std::runtime_error("batch exceeds its public 1 GiB data budget");
    }
    if (options.active_ranks <= 0 || options.allocated_ranks < options.active_ranks
        || options.allocated_ranks * 32 > 1024) {
        throw std::runtime_error("invalid rank-block policy");
    }
    if (options.profile < 0 || options.profile > 3 || options.repetitions <= 0) {
        throw std::runtime_error("invalid profile or repetition count");
    }
    if (!options.reference.empty() && options.repetitions % 2 != 0) {
        throw std::runtime_error("paired repetition count must be even for balanced order");
    }
    return options;
}

struct HostData {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    std::vector<double> golden;
};

std::size_t checked_elements(int batch, int rows, int columns) {
    if (batch <= 0 || rows <= 0 || columns <= 0) {
        throw std::runtime_error("nonpositive tensor extent");
    }
    const std::uint64_t count = static_cast<std::uint64_t>(batch)
        * static_cast<std::uint64_t>(rows)
        * static_cast<std::uint64_t>(columns);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::runtime_error("tensor extent overflows host address space");
    }
    return static_cast<std::size_t>(count);
}

HostData allocate_host_data(const Options &options) {
    HostData data;
    data.a.resize(checked_elements(options.batch, options.m, options.k));
    data.b.resize(checked_elements(options.batch, options.n, options.k));
    data.c.resize(checked_elements(options.batch, options.m, options.n));
    data.golden.resize(data.c.size());
    return data;
}

void fill_c(const Options &options, std::uint64_t phase, std::vector<double> &c) {
    const std::size_t c_stride = static_cast<std::size_t>(options.m) * options.n;
#pragma omp parallel for schedule(static)
    for (std::int64_t batch = 0; batch < options.batch; ++batch) {
        const std::size_t base = static_cast<std::size_t>(batch) * c_stride;
        for (std::size_t index = 0; index < c_stride; ++index) {
            c[base + index] = fk::sample_value(phase, 2, batch, index);
        }
    }
}

double prepare_phase(
    const Options &options,
    std::uint64_t phase,
    HostData &data,
    double &oracle_milliseconds) {
    const auto preprocess_start = std::chrono::steady_clock::now();
    const std::size_t a_stride = static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_stride = static_cast<std::size_t>(options.n) * options.k;
#pragma omp parallel for schedule(static)
    for (std::int64_t batch = 0; batch < options.batch; ++batch) {
        const std::size_t a_base = static_cast<std::size_t>(batch) * a_stride;
        const std::size_t b_base = static_cast<std::size_t>(batch) * b_stride;
        for (std::size_t index = 0; index < a_stride; ++index) {
            data.a[a_base + index] = fk::sample_value(phase, 0, batch, index);
        }
        for (std::size_t index = 0; index < b_stride; ++index) {
            data.b[b_base + index] = fk::sample_value(phase, 1, batch, index);
        }
    }
    fill_c(options, phase, data.c);
    const auto preprocess_stop = std::chrono::steady_clock::now();

    const auto oracle_start = std::chrono::steady_clock::now();
    const std::size_t c_stride = static_cast<std::size_t>(options.m) * options.n;
#pragma omp parallel for schedule(static)
    for (std::int64_t batch = 0; batch < options.batch; ++batch) {
        const std::size_t a_base = static_cast<std::size_t>(batch) * a_stride;
        const std::size_t b_base = static_cast<std::size_t>(batch) * b_stride;
        const std::size_t c_base = static_cast<std::size_t>(batch) * c_stride;
        for (int row = 0; row < options.m; ++row) {
            for (int col = 0; col < options.n; ++col) {
                double value = 0.0;
                for (int inner = 0; inner < options.k; ++inner) {
                    value += data.a[a_base + static_cast<std::size_t>(row) * options.k + inner]
                        * data.b[b_base + static_cast<std::size_t>(col) * options.k + inner];
                }
                const std::size_t c_index = c_base
                    + static_cast<std::size_t>(row) * options.n + col;
                data.golden[c_index] = fk::kAlpha * value + fk::kBeta * data.c[c_index];
            }
        }
    }
    const auto oracle_stop = std::chrono::steady_clock::now();
    oracle_milliseconds = std::chrono::duration<double, std::milli>(
        oracle_stop - oracle_start).count();
    return std::chrono::duration<double, std::milli>(
        preprocess_stop - preprocess_start).count();
}

void verify_every_element(
    const std::vector<double> &observed,
    const std::vector<double> &golden,
    const std::string &label) {
    if (observed.size() != golden.size()) {
        throw std::runtime_error(label + " output extent mismatch");
    }
    for (std::size_t index = 0; index < observed.size(); ++index) {
        const double expected = golden[index];
        const double tolerance = fk::kAbsoluteTolerance
            + fk::kRelativeTolerance * std::abs(expected);
        if (!std::isfinite(observed[index])
            || std::abs(observed[index] - expected) > tolerance) {
            std::ostringstream message;
            message << label << " failed full-output validation at element " << index
                    << ": observed=" << std::setprecision(17) << observed[index]
                    << " expected=" << expected;
            throw std::runtime_error(message.str());
        }
    }
}

class ModuleRunner {
public:
    ModuleRunner(CUdevice device, const std::string &module_path, const Options &options)
        : a_bytes_(checked_elements(options.batch, options.m, options.k) * sizeof(double)),
          b_bytes_(checked_elements(options.batch, options.n, options.k) * sizeof(double)),
          c_bytes_(checked_elements(options.batch, options.m, options.n) * sizeof(double)) {
        check_cuda(cuCtxCreate(&context_, 0, device), "cuCtxCreate");
        try {
            check_cuda(cuModuleLoad(&module_, module_path.c_str()), "cuModuleLoad");
            check_cuda(cuModuleGetFunction(&function_, module_, kEntry), "cuModuleGetFunction");
            check_cuda(cuMemAlloc(&d_a_, a_bytes_), "cuMemAlloc(A)");
            check_cuda(cuMemAlloc(&d_b_, b_bytes_), "cuMemAlloc(B)");
            check_cuda(cuMemAlloc(&d_c_, c_bytes_), "cuMemAlloc(C)");
            const int shared_bytes = (options.n / options.active_ranks)
                * options.k * static_cast<int>(sizeof(double));
            check_cuda(cuFuncSetAttribute(
                function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes), "cuFuncSetAttribute");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ModuleRunner(const ModuleRunner &) = delete;
    ModuleRunner &operator=(const ModuleRunner &) = delete;

    ~ModuleRunner() { cleanup(); }

    float run(const Options &options, const HostData &input, std::vector<double> &c) {
        check_cuda(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
        check_cuda(cuMemcpyHtoD(d_a_, input.a.data(), a_bytes_), "cuMemcpyHtoD(A)");
        check_cuda(cuMemcpyHtoD(d_b_, input.b.data(), b_bytes_), "cuMemcpyHtoD(B)");
        check_cuda(cuMemcpyHtoD(d_c_, c.data(), c_bytes_), "cuMemcpyHtoD(C)");

        CUevent start = nullptr;
        CUevent stop = nullptr;
        check_cuda(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
        try {
            check_cuda(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");
            CUdeviceptr a = d_a_;
            CUdeviceptr b = d_b_;
            CUdeviceptr output = d_c_;
            double alpha = fk::kAlpha;
            double beta = fk::kBeta;
            void *arguments[] = {&a, &b, &output, &alpha, &beta};
            const unsigned int shared_bytes = static_cast<unsigned int>(
                (options.n / options.active_ranks) * options.k * sizeof(double));
            check_cuda(cuEventRecord(start, nullptr), "cuEventRecord(start)");
            check_cuda(cuLaunchKernel(
                function_, options.batch, 1, 1,
                options.allocated_ranks * 32, 1, 1,
                shared_bytes, nullptr, arguments, nullptr), "cuLaunchKernel");
            check_cuda(cuEventRecord(stop, nullptr), "cuEventRecord(stop)");
            check_cuda(cuEventSynchronize(stop), "cuEventSynchronize(stop)");
            float milliseconds = 0.0F;
            check_cuda(cuEventElapsedTime(&milliseconds, start, stop), "cuEventElapsedTime");
            check_cuda(cuMemcpyDtoH(c.data(), d_c_, c_bytes_), "cuMemcpyDtoH(C)");
            check_cuda(cuEventDestroy(stop), "cuEventDestroy(stop)");
            check_cuda(cuEventDestroy(start), "cuEventDestroy(start)");
            return milliseconds;
        } catch (...) {
            if (stop) cuEventDestroy(stop);
            if (start) cuEventDestroy(start);
            throw;
        }
    }

private:
    void cleanup() noexcept {
        if (!context_) return;
        cuCtxSetCurrent(context_);
        if (d_c_) cuMemFree(d_c_);
        if (d_b_) cuMemFree(d_b_);
        if (d_a_) cuMemFree(d_a_);
        if (module_) cuModuleUnload(module_);
        cuCtxDestroy(context_);
        context_ = nullptr;
    }

    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;
    CUdeviceptr d_a_ = 0;
    CUdeviceptr d_b_ = 0;
    CUdeviceptr d_c_ = 0;
    std::size_t a_bytes_ = 0;
    std::size_t b_bytes_ = 0;
    std::size_t c_bytes_ = 0;
};

double to_tflops(std::uint64_t operations, float milliseconds) {
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0F) {
        throw std::runtime_error("CUDA event returned invalid elapsed time");
    }
    return static_cast<double>(operations) / (static_cast<double>(milliseconds) * 1e9);
}

void print_array(const std::vector<double> &values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << std::setprecision(17) << values[index];
    }
    std::cout << ']';
}

int run(const Options &options) {
    check_cuda(cuInit(0), "cuInit");
    int device_count = 0;
    check_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (device_count != 1) throw std::runtime_error("trusted host requires exactly one GPU");
    CUdevice device = 0;
    check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
    char name[256] = {};
    check_cuda(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    if (std::string(name) != kExactGpuName) {
        throw std::runtime_error("trusted host requires exactly one NVIDIA H100 80GB HBM3");
    }

    ModuleRunner candidate(device, options.candidate, options);
    std::unique_ptr<ModuleRunner> reference;
    if (!options.reference.empty()) {
        reference = std::make_unique<ModuleRunner>(device, options.reference, options);
    }
    HostData data = allocate_host_data(options);
    const std::uint64_t operations = fk::operation_count(
        options.m, options.n, options.k, options.batch);
    std::vector<double> candidate_tflops;
    std::vector<double> reference_tflops;
    std::vector<double> preprocess_ms;
    std::vector<double> oracle_ms;

    auto run_and_validate = [&](ModuleRunner &runner, const std::string &label) {
        const float elapsed = runner.run(options, data, data.c);
        verify_every_element(data.c, data.golden, label);
        return to_tflops(operations, elapsed);
    };

    double warm_oracle = 0.0;
    prepare_phase(options, fk::derive_phase(
        options.seed, options.profile, 0, true), data, warm_oracle);
    run_and_validate(candidate, "candidate warm-up");
    if (reference) {
        fill_c(options, fk::derive_phase(options.seed, options.profile, 0, true), data.c);
        run_and_validate(*reference, "reference warm-up");
    }

    for (int repetition = 0; repetition < options.repetitions; ++repetition) {
        const std::uint64_t phase = fk::derive_phase(
            options.seed, options.profile, repetition, false);
        double oracle = 0.0;
        preprocess_ms.push_back(prepare_phase(options, phase, data, oracle));
        oracle_ms.push_back(oracle);
        if (!reference || repetition % 2 == 0) {
            candidate_tflops.push_back(run_and_validate(candidate, "candidate timed"));
            if (reference) {
                fill_c(options, phase, data.c);
                reference_tflops.push_back(run_and_validate(*reference, "reference timed"));
            }
        } else {
            reference_tflops.push_back(run_and_validate(*reference, "reference timed"));
            fill_c(options, phase, data.c);
            candidate_tflops.push_back(run_and_validate(candidate, "candidate timed"));
        }
    }

    std::cout << "FCS_RESULT={\"batch_count\":" << options.batch
              << ",\"operation_count\":" << operations
              << ",\"num_iter\":1,\"alpha\":1.0,\"beta\":0.5"
              << ",\"warm_validated\":true,\"candidate_tflops\":";
    print_array(candidate_tflops);
    std::cout << ",\"reference_tflops\":";
    print_array(reference_tflops);
    std::cout << ",\"preprocess_milliseconds\":";
    print_array(preprocess_ms);
    std::cout << ",\"cpu_oracle_milliseconds\":";
    print_array(oracle_ms);
    std::cout << "}" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "trusted harness error: " << error.what() << std::endl;
        return 1;
    }
}
