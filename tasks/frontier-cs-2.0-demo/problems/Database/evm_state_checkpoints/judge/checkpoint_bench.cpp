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

// Rollback validation at several stack heights, through every access path
// (get, top, pop) and the restored code position, with the stack height changing
// between checkpoints. Not timed. Returns 0 on success.
int validate_rollbacks(uint64_t seed) {
    static const uint8_t code[16] = {};
    constexpr size_t heights[] = {1, 16, 64, 65, 200, 1000};
    constexpr size_t checkpoint_count = 6;
    // Newest first, as in Spectrum, where a rollback discards later checkpoints.
    constexpr size_t order[checkpoint_count] = {5, 4, 3, 2, 1, 0};
    for (size_t height : heights) {
        auto* state = new ExecutionState();
        state->stack_top = StackTop(&state->stack_space.m_stack_space[0],
                                    &state->stack_space.m_stack_space[state->stack_space.limit]);
        auto& stack = state->stack_top;
        std::vector<uint256> oracle;
        for (size_t i = 0; i < height; ++i) {
            oracle.push_back(seed * 1000003 + height * 7919 + i);
            stack.push(oracle.back());
        }
        std::vector<Checkpoint> snapshots;
        std::vector<std::vector<uint256>> expected;
        for (size_t c = 0; c < checkpoint_count; ++c) {
            state->position = Position{code + c};
            expected.push_back(oracle);
            snapshots.push_back(state->save_checkpoint());
            for (size_t j = 0; j < 3; ++j) {
                const size_t from_top = (seed + c * 31 + j * 17) % stack.height;
                const uint256 value = (seed << 20) ^ (height << 8) ^ (c << 4) ^ j ^ 0x5a5a;
                stack.get_mut(from_top) = value;
                oracle[oracle.size() - 1 - from_top] = value;
            }
            for (size_t j = 0; j < 2; ++j) {
                oracle.push_back(seed ^ (height << 12) ^ (c << 2) ^ j ^ 0xa5a5);
                stack.push(oracle.back());
            }
            if (stack.pop() != oracle.back()) return 10;
            oracle.pop_back();
        }
        for (size_t k = 0; k < checkpoint_count; ++k) {
            const size_t c = order[k];
            state->position = Position{code + 15};
            state->load_checkpoint(snapshots[c]);
            const auto& want = expected[c];
            if (stack.height != want.size()) return 11;
            if (state->position.value().code_it != code + c) return 12;
            // Every other restore is read only through top() and pop(), with no
            // get() call that could complete a deferred restore first.
            if (k % 2 == 0) {
                for (size_t i = 0; i < want.size(); ++i) {
                    if (stack.get(i) != want[want.size() - 1 - i]) return 14;
                }
            }
            for (size_t i = want.size(); i-- > 0;) {
                if (stack.top() != want[i]) return 13;
                if (stack.pop() != want[i]) return 17;
            }
            if (stack.height != 0) return 18;
        }
        snapshots.clear();
        delete state;
    }
    return 0;
}


// The interpreter-like access mix that follows every timed restore: reads and
// writes near the top of the stack through top(), get(), get_mut(), pop() and
// push(), keeping the height constant. Results are checked against the oracle.
// Returns false on a mismatch.
template <size_t N>
bool access_mix(StackTop& stack, std::array<uint256, N>& oracle, uint64_t seed, uint64_t round) {
    constexpr size_t steps = 64;
    const size_t height = stack.height;
    for (size_t i = 0; i < steps; ++i) {
        const uint64_t h = (seed * 0x9e3779b97f4a7c15ULL) ^ (round * 0xbf58476d1ce4e5b9ULL) ^ (i * 0x94d049bb133111ebULL);
        const size_t depth = static_cast<size_t>(h >> 8) % std::min<size_t>(16, height);
        const uint256 value = uint256{h} ^ uint256{i};
        switch (h % 4) {
        case 0:
            if (stack.top() != oracle[height - 1]) return false;
            break;
        case 1:
            if (stack.get(depth) != oracle[height - 1 - depth]) return false;
            break;
        case 2:
            stack.get_mut(depth) = value;
            oracle[height - 1 - depth] = value;
            break;
        default:
            if (stack.pop() != oracle[height - 1]) return false;
            stack.push(value);
            oracle[height - 1] = value;
            break;
        }
    }
    return stack.height == height;
}

int main(int argc, char** argv) {
    constexpr size_t height = 1000;
    constexpr size_t checkpoint_count = 16;
    constexpr size_t writes_per_checkpoint = 24;
    const int rounds = argc > 1 ? std::stoi(argv[1]) : 200;
    const uint64_t seed = argc > 2 ? std::stoull(argv[2]) : 1;
    uint64_t checksum = 0;
    uint64_t snapshot_ns = 0;
    uint64_t rollback_ns = 0;
    uint64_t end_to_end_ns = 0;
    size_t retained_per_checkpoint = 0;
    static const uint8_t code[] = {0};

    if (const int failed = validate_rollbacks(seed)) {
        std::cerr << "rollback validation failed\n";
        return failed;
    }

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
            auto t2 = clock_type::now();
            end_to_end_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t0).count();
        }

        const size_t peak = allocation_tracker::peak.load(std::memory_order_relaxed);
        retained_per_checkpoint = std::max(retained_per_checkpoint,
                                           (peak + checkpoint_count - 1) / checkpoint_count);
        allocation_tracker::end();

        const size_t selected = (seed + r * 7) % checkpoint_count;
        Checkpoint copied = snapshots[selected];
        Checkpoint moved = std::move(copied);
        // restore_ns covers load_checkpoint() alone. The end-to-end time also
        // covers the access mix that follows each restore.
        auto expected = oracle_snapshots[selected];
        auto t0 = clock_type::now();
        state->load_checkpoint(moved);
        auto t1 = clock_type::now();
        const bool first_mix = stack.height == height && access_mix(stack, expected, seed, uint64_t(r) * 2);
        auto t2 = clock_type::now();
        rollback_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        end_to_end_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t0).count();
        if (!first_mix) return 2;
        for (size_t i = 0; i < height; ++i) {
            if (stack.get(height - 1 - i) != expected[i]) {
                std::cerr << "state mismatch at checkpoint " << selected << " item " << i << '\n';
                return 3;
            }
        }
        for (size_t i = height; i-- > 0;) {
            if (stack.top() != expected[i]) return 6;
            if (stack.pop() != expected[i]) return 7;
        }

        // As in Spectrum, a rollback discards the checkpoints created after its
        // target, so once the restored state has been modified only an older
        // checkpoint is loaded.
        if (selected > 0) {
            const size_t second = (seed + uint64_t(r)) % selected;
            expected = oracle_snapshots[second];
            auto t3 = clock_type::now();
            state->load_checkpoint(snapshots[second]);
            const bool second_mix = stack.height == height && access_mix(stack, expected, seed, uint64_t(r) * 2 + 1);
            auto t4 = clock_type::now();
            end_to_end_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
            if (!second_mix) return 4;
            for (size_t i = 0; i < height; ++i) {
                if (stack.get(height - 1 - i) != expected[i]) return 4;
            }
            checksum ^= static_cast<uint64_t>(stack.top()[0]);
            for (size_t i = height; i-- > 0;) {
                if (stack.top() != expected[i]) return 8;
                if (stack.pop() != expected[i]) return 9;
            }
        }

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
    const uint64_t end_to_end_per_round = end_to_end_ns / uint64_t(rounds);
    std::cout << create_per_checkpoint << ' ' << restore_per_checkpoint << ' '
              << retained_per_checkpoint << ' ' << checkpoint_count << ' ' << checksum << ' '
              << end_to_end_per_round << '\n';
}
