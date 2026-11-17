#pragma once
#include <random>
#include <memory>
#include <functional>
#include <spectrum/common/lock-util.hpp>

namespace spectrum {

class Random {

    public:
    virtual size_t Next() = 0;
    virtual ~Random() = default;

};

void SampleUniqueN(Random& random, std::vector<size_t>& samples);

class ThreadLocalRandom: public Random {

    private:
    std::vector<std::unique_ptr<Random>> thread_local_storage;

    public:
    ThreadLocalRandom(std::function<std::unique_ptr<Random>()> random_fn, size_t duplication);
    ~ThreadLocalRandom() override = default;
    size_t Next() override;

};

class Unif : public Random {

    private:
    SpinLock        mu;
    std::mt19937    rng;
    std::uniform_int_distribution<size_t>   distribution;

    public:
    Unif(size_t num_elements);
    ~Unif() override = default;
    size_t Next() override;

};

class Zipf : public Random {

    private:
    SpinLock        mu;
    double          num_elements;
    double          exponent;
    double          h_integral_x1;
    double          h_integral_num_elements;
    double          s;
    std::mt19937    rng;

    public:
    Zipf(size_t num_elements, double exponent);
    ~Zipf() override = default;
    size_t Next() override;

};

} // namespace spectrum