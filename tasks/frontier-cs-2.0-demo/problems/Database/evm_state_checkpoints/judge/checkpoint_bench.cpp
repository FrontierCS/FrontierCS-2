#include "lib/evmcow/execution_state.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <utility>
#include <vector>

namespace allocation_tracker {
struct Header { void* raw; size_t size; bool counted; };
std::atomic<size_t> live{0};
std::atomic<size_t> peak{0};
std::atomic<bool> enabled{false};

void update_peak(size_t value) {
    size_t seen = peak.load(std::memory_order_relaxed);
    while (seen < value && !peak.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {}
}

void* allocate(size_t size, size_t alignment) {
    if (size == 0) size = 1;
    alignment = std::max(alignment, alignof(Header));
    const size_t total = size + alignment - 1 + sizeof(Header);
    void* raw = std::malloc(total);
    if (!raw) throw std::bad_alloc();
    uintptr_t start = reinterpret_cast<uintptr_t>(raw) + sizeof(Header);
    uintptr_t aligned = (start + alignment - 1) & ~(uintptr_t(alignment - 1));
    auto* header = reinterpret_cast<Header*>(aligned - sizeof(Header));
    header->raw = raw;
    header->size = size;
    header->counted = enabled.load(std::memory_order_relaxed);
    if (header->counted) {
        size_t value = live.fetch_add(size, std::memory_order_relaxed) + size;
        update_peak(value);
    }
    return reinterpret_cast<void*>(aligned);
}

void release(void* ptr) noexcept {
    if (!ptr) return;
    auto* header = reinterpret_cast<Header*>(reinterpret_cast<uintptr_t>(ptr) - sizeof(Header));
    if (header->counted) live.fetch_sub(header->size, std::memory_order_relaxed);
    std::free(header->raw);
}

void begin() {
    live.store(0, std::memory_order_relaxed);
    peak.store(0, std::memory_order_relaxed);
    enabled.store(true, std::memory_order_relaxed);
}
void end() { enabled.store(false, std::memory_order_relaxed); }
}  // namespace allocation_tracker

void* operator new(size_t size) { return allocation_tracker::allocate(size, alignof(std::max_align_t)); }
void* operator new[](size_t size) { return allocation_tracker::allocate(size, alignof(std::max_align_t)); }
void* operator new(size_t size, std::align_val_t alignment) {
    return allocation_tracker::allocate(size, static_cast<size_t>(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment) {
    return allocation_tracker::allocate(size, static_cast<size_t>(alignment));
}
void operator delete(void* ptr) noexcept { allocation_tracker::release(ptr); }
void operator delete[](void* ptr) noexcept { allocation_tracker::release(ptr); }
void operator delete(void* ptr, size_t) noexcept { allocation_tracker::release(ptr); }
void operator delete[](void* ptr, size_t) noexcept { allocation_tracker::release(ptr); }
void operator delete(void* ptr, std::align_val_t) noexcept { allocation_tracker::release(ptr); }
void operator delete[](void* ptr, std::align_val_t) noexcept { allocation_tracker::release(ptr); }
void operator delete(void* ptr, size_t, std::align_val_t) noexcept { allocation_tracker::release(ptr); }
void operator delete[](void* ptr, size_t, std::align_val_t) noexcept { allocation_tracker::release(ptr); }

using namespace evmcow;
using clock_type = std::chrono::steady_clock;

int main(int argc, char** argv) {
    constexpr size_t height = 1000;
    constexpr size_t checkpoint_count = 16;
    constexpr size_t writes_per_checkpoint = 24;
    const int rounds = argc > 1 ? std::stoi(argv[1]) : 200;
    const uint64_t seed = argc > 2 ? std::stoull(argv[2]) : 1;
    uint64_t checksum = 0;
    uint64_t snapshot_ns = 0;
    uint64_t rollback_ns = 0;
    size_t retained_per_checkpoint = 0;
    static const uint8_t code[] = {0};

    for (int r = 0; r < rounds; ++r) {
        std::array<uint256, height> oracle{};
        std::array<std::array<uint256, height>, checkpoint_count> oracle_snapshots{};
        std::vector<Checkpoint> snapshots;

        allocation_tracker::begin();
        auto* state = new ExecutionState();
        state->stack_top = StackTop(&state->stack_space.m_stack_space[0],
                                    &state->stack_space.m_stack_space[state->stack_space.limit]);
        state->position = Position{code};
        auto& stack = state->stack_top;
        snapshots.reserve(checkpoint_count);
        for (size_t i = 0; i < height; ++i) {
            oracle[i] = seed + uint64_t(r) * height + i;
            stack.push(oracle[i]);
        }

        for (size_t c = 0; c < checkpoint_count; ++c) {
            oracle_snapshots[c] = oracle;
            auto t0 = clock_type::now();
            auto checkpoint = state->save_checkpoint();
            snapshots.push_back(std::move(checkpoint));
            auto t1 = clock_type::now();
            snapshot_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            for (size_t j = 0; j < writes_per_checkpoint; ++j) {
                size_t from_top = (seed + uint64_t(r) * 17 + c * 53 + j * 37) % stack.height;
                uint256 value = seed ^ (uint64_t(r) << 32) ^ (c << 16) ^ j;
                stack.get_mut(from_top) = value;
                oracle[height - 1 - from_top] = value;
            }
        }

        const size_t peak = allocation_tracker::peak.load(std::memory_order_relaxed);
        retained_per_checkpoint = std::max(retained_per_checkpoint,
                                           (peak + checkpoint_count - 1) / checkpoint_count);
        allocation_tracker::end();

        const size_t selected = (seed + r * 7) % checkpoint_count;
        Checkpoint copied = snapshots[selected];
        Checkpoint moved = std::move(copied);
        auto t0 = clock_type::now();
        state->load_checkpoint(moved);
        auto t1 = clock_type::now();
        rollback_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        if (stack.height != height) return 2;
        for (size_t i = 0; i < height; ++i) {
            if (stack.get(height - 1 - i) != oracle_snapshots[selected][i]) {
                std::cerr << "state mismatch at checkpoint " << selected << " item " << i << '\n';
                return 3;
            }
        }

        const size_t second = (selected + checkpoint_count / 2) % checkpoint_count;
        state->load_checkpoint(snapshots[second]);
        for (size_t i = 0; i < height; ++i) {
            if (stack.get(height - 1 - i) != oracle_snapshots[second][i]) return 4;
        }
        checksum ^= static_cast<uint64_t>(stack.top()[0]);

        while (!snapshots.empty()) {
            const size_t erase_at = (seed + r + snapshots.size()) % snapshots.size();
            snapshots.erase(snapshots.begin() + static_cast<std::ptrdiff_t>(erase_at));
        }
        std::vector<Checkpoint>().swap(snapshots);
        delete state;
        if (allocation_tracker::live.load(std::memory_order_relaxed) != 0) return 5;
    }

    const uint64_t create_per_checkpoint = snapshot_ns / (uint64_t(rounds) * checkpoint_count);
    const uint64_t restore_per_checkpoint = rollback_ns / uint64_t(rounds);
    std::cout << create_per_checkpoint << ' ' << restore_per_checkpoint << ' '
              << retained_per_checkpoint << ' ' << checkpoint_count << ' ' << checksum << '\n';
}
