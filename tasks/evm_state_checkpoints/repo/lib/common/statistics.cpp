#include <atomic>
#include <spectrum/common/statistics.hpp>
#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/args.h>
#include <iostream>
#include <glog/logging.h>
#include <cstdlib>
#include <ethash/keccak.hpp>

namespace spectrum {

void Statistics::JournalCommit(size_t latency) {
    auto count_commit_ = count_commit.fetch_add(1, std::memory_order_relaxed);
    if (latency <= 25) {
        count_latency_25us.fetch_add(1, std::memory_order_relaxed);
    }
    else if (latency <= 50) {
        count_latency_50us.fetch_add(1, std::memory_order_relaxed);
    }
    else if (latency <= 100) {
        count_latency_100us.fetch_add(1, std::memory_order_relaxed);
    }
    else {
        count_latency_100us_above.fetch_add(1, std::memory_order_relaxed);
    }
    DLOG(INFO) << "latency: " << latency << std::endl;
    
    if (count_commit_ < SAMPLE) {
        sample_latency[count_commit_].store(latency);
        return;
    }
    auto random = ethash::keccak256((uint8_t*)&count_commit_, 4).word64s[count_commit_ % 4] % count_commit_;
    if (random < SAMPLE) {
        sample_latency[random].store(latency);
    }
}

void Statistics::JournalMemory(size_t count) {
    count_memory.fetch_add(count, std::memory_order_relaxed);
}

void Statistics::JournalExecute() {
    count_execution.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::JournalOperations(size_t count) {
    count_operation.fetch_add(count, std::memory_order_relaxed);
}

std::string Statistics::Print() {
    #define PERCENTILE(X) sample_latency_[X * sample_latency_.size() / 100]
    auto sample_latency_ = std::vector<size_t>();
    std::transform(
        sample_latency.begin(), 
        sample_latency.end(), 
        std::back_inserter(sample_latency_),
        [](auto& x) { return x.load(); }
    );
    std::sort(sample_latency_.begin(), sample_latency_.end());
    return std::string(fmt::format(
        "{}\n"
        "commit             {}\n"
        "memory             {}\n"
        "execution          {}\n"
        "operation          {}\n"
        "25us               {}\n"
        "50us               {}\n"
        "100us              {}\n"
        ">100us             {}\n"
        "latency(50%)       {}us\n"
        "latency(75%)       {}us\n"
        "latency(95%)       {}us\n"
        "latency(99%)       {}us\n",
        std::chrono::system_clock::now(),
        count_commit.load(),
        count_memory.load(),
        count_execution.load(),
        count_operation.load(),
        count_latency_25us.load(),
        count_latency_50us.load(),
        count_latency_100us.load(),
        count_latency_100us_above.load(),
        PERCENTILE(50),
        PERCENTILE(75),
        PERCENTILE(95),
        PERCENTILE(99)
    ));
    #undef PERCENTILE
}

std::string Statistics::PrintWithDuration(std::chrono::milliseconds duration) {
    #define AVG(X) ((double)(X.load()) / (double)(duration.count()) * (double)(1000))
    #define PERCENTILE(X) sample_latency_[X * sample_latency_.size() / 100]
    auto sample_latency_ = std::vector<size_t>();
    std::transform(
        sample_latency.begin(), 
        sample_latency.end(), 
        std::back_inserter(sample_latency_),
        [](auto& x) { return x.load(); }
    );
    std::sort(sample_latency_.begin(), sample_latency_.end());
    return std::string(fmt::format(
        "{}\n"
        "duration      {}\n"
        "commit        {:.4f} tx/s\n"
        "memory        {:.4f} bytes/s\n"
        "execution     {:.4f} tx/s\n"
        "operation     {:.4f} op/s\n"
        "25us          {:.4f} tx/s\n"
        "50us          {:.4f} tx/s\n"
        "100us         {:.4f} tx/s\n"
        ">100us        {:.4f} tx/s\n"
        "latency(50%)  {}us\n"
        "latency(75%)  {}us\n"
        "latency(95%)  {}us\n"
        "latency(99%)  {}us\n",
        std::chrono::system_clock::now(),
        duration,
        AVG(count_commit),
        AVG(count_memory),
        AVG(count_execution),
        AVG(count_operation),
        AVG(count_latency_25us),
        AVG(count_latency_50us),
        AVG(count_latency_100us),
        AVG(count_latency_100us_above),
        PERCENTILE(50),
        PERCENTILE(75),
        PERCENTILE(95),
        PERCENTILE(99)
    ));
    #undef AVG
    #undef PERCENTILE
}

} // namespace spectrum
