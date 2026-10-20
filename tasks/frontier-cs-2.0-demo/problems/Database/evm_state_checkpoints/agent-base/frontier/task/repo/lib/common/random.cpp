#include <spectrum/common/random.hpp>
#include <glog/logging.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <cstdlib>

/*
    To investigate high-contention rate circumstance. We have to use Zipfian distribution. 
    The following implementation is a direct port of: 
    https://github.com/apache/commons-rng/blob/6a1b0c16090912e8fc5de2c1fb5bd8490ac14699/commons-rng-sampling/src/main/java/org/apache/commons/rng/sampling/distribution/RejectionInversionZipfSampler.java
 */

namespace spectrum {

void SampleUniqueN(Random& random, std::vector<size_t>& samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        while (true) {
            // sample a random number
            auto rn = random.Next();
            // see if it is distinct from others
            auto eq = false;
            for (size_t j = 0; j < i; ++j) {
                eq |= samples[j] == rn;
            }
            // if it is distinct, assign and breaks
            if (!eq) { samples[i] = rn; break; }
        }
    }
}

ThreadLocalRandom::ThreadLocalRandom(
    std::function<std::unique_ptr<Random>()> random_fn, 
    size_t duplication
) {
    for (size_t i = 0; i < duplication; ++i) {
        thread_local_storage.push_back(random_fn());
    }
}

size_t ThreadLocalRandom::Next() {
    size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return thread_local_storage[thread_id % thread_local_storage.size()]->Next();
}

Unif::Unif(size_t num_elements):
    distribution(0, std::max(size_t(1), num_elements) - 1),
    rng(rand())
{}

size_t Unif::Next() {
    auto guard = Guard{mu};
    return distribution(rng);
}

double h(double x, double exponent) {
    return exp(-exponent * log(x));
}

double helper1(double x) {
    if (std::abs(x) > 1e-8) {
        return log1p(x) / x;
    } else {
        return 1 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x));
    }
}

double helper2(double x) {
    if (std::abs(x) > 1e-8) {
        return expm1(x) / x;
    } else {
        return 1 + x * 0.5 * (1 + x * 1.0 / 3.0 * (1 + 0.25 * x));
    }
}

double h_integral_inv(double x, double exponent) {
    double t = x * (1 - exponent);
    if (t < -1) {
        t = -1;
    }
    return exp(helper1(t) * x);
}

double h_integral(double x, double exponent) {
    double log_x = log(x);
    return helper2((1 - exponent) * log_x) * log_x;
}

Zipf::Zipf(size_t num_elements, double exponent):
    num_elements{(double) num_elements},
    exponent{exponent},
    rng(rand())
{
    if (num_elements == 0) {
        throw std::invalid_argument("Number of elements must be greater than 0");
    }
    if (exponent <= 0) {
        throw std::invalid_argument("Exponent must be greater than 0");
    }
    h_integral_x1 = h_integral(1.5, exponent) - 1;
    h_integral_num_elements = h_integral(num_elements + 0.5, exponent);
    s = 2 - h_integral_inv(h_integral(2.5, exponent) - h(2, exponent), exponent);
}

size_t Zipf::Next() {
    auto guard = Guard{mu};
    double hnum = h_integral_num_elements;
    while (true) {
        double u = hnum + std::generate_canonical<double, std::numeric_limits<double>::digits>(rng) * (h_integral_x1 - hnum);
        double x = h_integral_inv(u, exponent);
        double k64 = std::max(x, 1.0);
        k64 = std::min(k64, num_elements);
        size_t k = std::max(static_cast<size_t>(k64 + 0.5), static_cast<size_t>(1));
        if (k64 - x <= s || u >= h_integral(k64 + 0.5, exponent) - h(k64, exponent)) {
            return k;
        }
    }
}

} // namespace spectrum