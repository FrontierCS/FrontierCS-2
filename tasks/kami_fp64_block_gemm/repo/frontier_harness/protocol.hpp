#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace frontier_kami {

constexpr std::uint64_t kMask64 = ~std::uint64_t{0};
constexpr std::uint64_t kSplitMixGamma = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kWarmPhaseSalt = 0x6A09E667F3BCC909ULL;
constexpr std::uint64_t kTimedPhaseSalt = 0xBB67AE8584CAA73BULL;
constexpr std::uint64_t kTensorSalts[3] = {
    0x3C6EF372FE94F82BULL,
    0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL,
};
constexpr int kValueScaleLog2 = 10;
constexpr std::uint64_t kValueMask = (std::uint64_t{1} << 11) - 1;
constexpr std::int64_t kValueBias = std::int64_t{1} << 10;
constexpr double kAlpha = 1.0;
constexpr double kBeta = 0.5;
constexpr double kAbsoluteTolerance = 1e-12;
constexpr double kRelativeTolerance = 2e-13;
constexpr std::uint64_t kOneGiB = std::uint64_t{1} << 30;
constexpr int kMinimumBatch = 4096;

inline std::uint64_t splitmix64(std::uint64_t value) {
    value += kSplitMixGamma;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

inline std::uint64_t derive_phase(
    std::uint64_t input_seed,
    int input_profile,
    int repetition,
    bool warm) {
    std::uint64_t value = input_seed;
    value ^= warm ? kWarmPhaseSalt : kTimedPhaseSalt;
    value ^= static_cast<std::uint64_t>(input_profile + 1)
        * 0xD1B54A32D192ED03ULL;
    value ^= static_cast<std::uint64_t>(repetition + 1)
        * 0x94D049BB133111EBULL;
    return splitmix64(value);
}

inline std::uint64_t raw_sample(
    std::uint64_t phase,
    int tensor,
    std::uint64_t batch_index,
    std::uint64_t element_index) {
    const std::uint64_t linear = (batch_index << 32) | element_index;
    const std::uint64_t domain = splitmix64(phase ^ kTensorSalts[tensor]);
    return splitmix64(domain + linear * kSplitMixGamma);
}

inline std::int64_t sample_numerator(
    std::uint64_t phase,
    int tensor,
    std::uint64_t batch_index,
    std::uint64_t element_index) {
    return static_cast<std::int64_t>(
        raw_sample(phase, tensor, batch_index, element_index) & kValueMask)
        - kValueBias;
}

inline double sample_value(
    std::uint64_t phase,
    int tensor,
    std::uint64_t batch_index,
    std::uint64_t element_index) {
    return std::ldexp(
        static_cast<double>(sample_numerator(
            phase, tensor, batch_index, element_index)),
        -kValueScaleLog2);
}

inline int approved_batch(int m, int n, int k) {
    if (m == 16 && n == 16 && k == 24) return 65536;
    if (m == 32 && n == 64 && k == 48) return 8192;
    if (m == 64 && n == 64 && k == 32) return 8192;
    if (m == 128 && n == 64 && k == 64) return 4096;
    if (m == 16 && n == 16 && k == 48) return 65536;
    if (m == 32 && n == 64 && k == 24) return 16384;
    if (m == 64 && n == 64 && k == 64) return 8192;
    if (m == 128 && n == 64 && k == 32) return 4096;
    return 0;
}

inline std::uint64_t operation_count(int m, int n, int k, int batch) {
    return std::uint64_t{2}
        * static_cast<std::uint64_t>(m)
        * static_cast<std::uint64_t>(n)
        * static_cast<std::uint64_t>(k)
        * static_cast<std::uint64_t>(batch);
}

inline std::uint64_t device_bytes(int m, int n, int k, int batch) {
    const std::uint64_t elements =
        std::uint64_t{m} * k + std::uint64_t{n} * k + std::uint64_t{m} * n;
    return sizeof(double) * elements * static_cast<std::uint64_t>(batch);
}

inline std::uint64_t host_bytes(int m, int n, int k, int batch) {
    const std::uint64_t elements =
        std::uint64_t{m} * k + std::uint64_t{n} * k
        + std::uint64_t{2} * m * n;
    return sizeof(double) * elements * static_cast<std::uint64_t>(batch);
}

}  // namespace frontier_kami
