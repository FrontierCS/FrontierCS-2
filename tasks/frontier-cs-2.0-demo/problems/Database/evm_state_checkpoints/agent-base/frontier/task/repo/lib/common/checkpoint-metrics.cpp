#include <spectrum/common/checkpoint-metrics.hpp>
#include <atomic>
#include <cstddef>
#include <fmt/core.h>
#include <glog/logging.h>

extern "C" int mallctl(const char* name, void* oldp, size_t* oldlenp, void* newp, size_t newlen);

namespace spectrum {

namespace {
std::atomic<uint64_t> create_count{0};
std::atomic<uint64_t> create_total_ns{0};
std::atomic<uint64_t> restore_count{0};
std::atomic<uint64_t> restore_total_ns{0};
std::atomic<uint64_t> retained_count{0};
std::atomic<uint64_t> retained_total_bytes{0};

struct ThreadCounters {
    uint64_t* allocated{nullptr};
    uint64_t* deallocated{nullptr};
    ThreadCounters() {
        size_t size = sizeof(uint64_t*);
        if (mallctl("thread.allocatedp", &allocated, &size, nullptr, 0) != 0) allocated = nullptr;
        size = sizeof(uint64_t*);
        if (mallctl("thread.deallocatedp", &deallocated, &size, nullptr, 0) != 0) deallocated = nullptr;
        CHECK(allocated != nullptr && deallocated != nullptr) << "jemalloc thread counters are unavailable";
    }
};
}  // namespace

void frontier_checkpoint_record_create(uint64_t nanoseconds) {
    create_count.fetch_add(1, std::memory_order_relaxed);
    create_total_ns.fetch_add(nanoseconds, std::memory_order_relaxed);
}

void frontier_checkpoint_record_restore(uint64_t nanoseconds) {
    restore_count.fetch_add(1, std::memory_order_relaxed);
    restore_total_ns.fetch_add(nanoseconds, std::memory_order_relaxed);
}

void frontier_checkpoint_record_retained(uint64_t bytes_per_checkpoint) {
    retained_count.fetch_add(1, std::memory_order_relaxed);
    retained_total_bytes.fetch_add(bytes_per_checkpoint, std::memory_order_relaxed);
}

int64_t frontier_checkpoint_thread_net_bytes() {
    thread_local ThreadCounters counters;
    return static_cast<int64_t>(*counters.allocated) - static_cast<int64_t>(*counters.deallocated);
}

std::string frontier_checkpoint_print() {
    const auto average = [](const std::atomic<uint64_t>& total, const std::atomic<uint64_t>& count) {
        const auto n = count.load();
        return n == 0 ? 0.0 : static_cast<double>(total.load()) / static_cast<double>(n);
    };
    return fmt::format(
        "checkpoint create_ns {:.2f} restore_ns {:.2f} retained_bytes {:.2f} creates {} restores {} retained_samples {}\n",
        average(create_total_ns, create_count), average(restore_total_ns, restore_count),
        average(retained_total_bytes, retained_count), create_count.load(), restore_count.load(),
        retained_count.load());
}

} // namespace spectrum
