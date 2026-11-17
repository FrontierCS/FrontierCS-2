// evmone: Fast Ethereum Virtual Machine implementation
// Copyright 2019 The evmone Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <string>
#include <vector>
#include <optional>

namespace evmone
{
namespace baseline
{
class CodeAnalysis;
}

using code_iterator = const uint8_t*;
using uint256 = intx::uint256;
using bytes = std::basic_string<uint8_t>;
using bytes_view = std::basic_string_view<uint8_t>;

/// Provides memory for EVM stack.
class StackSpace
{
public:
    /// The maximum number of EVM stack items.
    static constexpr auto limit = 4096;

    /// Returns the pointer to the "bottom", i.e. below the stack space.
    [[nodiscard, clang::no_sanitize("bounds")]] uint256* bottom() const noexcept
    {
        return const_cast<uint256*>(m_stack_space - 1);
    }

private:
    /// The storage allocated for maximum possible number of items.
    /// Items are aligned to 256 bits for better packing in cache lines.
    alignas(sizeof(uint256)) uint256 m_stack_space[limit];
};


/// The EVM memory.
///
/// The implementations uses initial allocation of 4k and then grows capacity with 2x factor.
/// Some benchmarks has been done to confirm 4k is ok-ish value.
class Memory
{
    /// The size of allocation "page".
    static constexpr size_t page_size = 4 * 1024;

    /// The "virtual" size of the memory.
    size_t m_size = 0;

    /// Actual allocated memory.
    std::vector<uint8_t> m_data;

public:
    /// Creates Memory object with initial capacity allocation.
    Memory() noexcept: m_data{std::vector<uint8_t>(page_size)} { }

    /// Frees all allocated memory.
    ~Memory() noexcept {  }

    Memory(const Memory&) = default;
    Memory& operator=(const Memory&) = delete;

    uint8_t& operator[](size_t index) noexcept { return m_data[index]; }

    [[nodiscard]] const uint8_t* data() const noexcept {
        if (m_data.size() == 0) { return nullptr; }
        return &m_data.front();
    }
    [[nodiscard]] size_t size() const noexcept { return m_size; }

    /// Grows the memory to the given size. The extend is filled with zeros.
    ///
    /// @param new_size  New memory size. Must be larger than the current size and multiple of 32.
    void grow(size_t new_size) noexcept
    {
        // Restriction for future changes. EVM always has memory size as multiple of 32 bytes.
        INTX_REQUIRE(new_size % 32 == 0);

        // Allow only growing memory. Include hint for optimizing compiler.
        INTX_REQUIRE(new_size > m_size);

        m_data.resize(new_size, 0);
        m_size = new_size;
    }

    /// Virtually clears the memory by setting its size to 0. The capacity stays unchanged.
    void clear() noexcept { m_size = 0; }
};

/// The execution position.
struct Position
{
    code_iterator code_it;      ///< The position in the code.
    uint256* stack_top;         ///< The pointer to the stack top.
};

/// Generic execution state for generic instructions implementations.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class ExecutionState
{
public:
    std::optional<Position> position{std::nullopt};
    int64_t gas_refund = 0;
    Memory memory;
    const evmc_message* msg = nullptr;
    evmc::HostContext host;
    evmc_revision rev = {};
    bytes return_data;
    bool will_break = false;

    /// Reference to original EVM code container.
    /// For legacy code this is a reference to entire original code.
    /// For EOF-formatted code this is a reference to entire container.
    bytes_view original_code;

    /// Reference to the EOF data section. May be empty.
    bytes_view data;

    evmc_status_code status = EVMC_SUCCESS;
    size_t output_offset = 0;
    size_t output_size = 0;

private:
    evmc_tx_context m_tx = {};

public:
    /// Pointer to code analysis.
    /// This should be set and used internally by execute() function of a particular interpreter.
    union
    {
        const baseline::CodeAnalysis* baseline = nullptr;
    } analysis{};

    std::vector<const uint8_t*> call_stack;

    /// Stack space allocation.
    ///
    /// This is the last field to make other fields' offsets of reasonable values.
    StackSpace stack_space;

    ExecutionState() = default;
    ~ExecutionState() = default;
    ExecutionState(const ExecutionState& state_ref):
        gas_refund{state_ref.gas_refund},
        memory{state_ref.memory},
        msg{state_ref.msg},
        host{state_ref.host},
        rev{state_ref.rev},
        return_data{state_ref.return_data},
        will_break{state_ref.will_break},
        original_code{state_ref.original_code},
        data{state_ref.data},
        status{state_ref.status},
        output_offset{state_ref.output_offset},
        output_size{state_ref.output_size},
        m_tx{state_ref.m_tx},
        analysis{state_ref.analysis},
        call_stack{state_ref.call_stack},
        stack_space{state_ref.stack_space},
        position{state_ref.position}
    {
        if (state_ref.position != std::nullopt) {
            position->stack_top = stack_space.bottom() 
                + (state_ref.position->stack_top - state_ref.stack_space.bottom());
        }
    }

    ExecutionState(const evmc_message& message, evmc_revision revision,
        const evmc_host_interface& host_interface, evmc_host_context* host_ctx, bytes_view _code,
        bytes_view _data) noexcept
      : msg{&message},
        host{host_interface, host_ctx},
        rev{revision},
        original_code{_code},
        data{_data}
    {}

    /// Resets the contents of the ExecutionState so that it could be reused.
    void reset(const evmc_message& message, evmc_revision revision,
        const evmc_host_interface& host_interface, evmc_host_context* host_ctx, bytes_view _code,
        bytes_view _data) noexcept
    {
        position = std::nullopt;
        gas_refund = 0;
        memory.clear();
        msg = &message;
        host = {host_interface, host_ctx};
        rev = revision;
        return_data.clear();
        original_code = _code;
        data = _data;
        status = EVMC_SUCCESS;
        output_offset = 0;
        output_size = 0;
        m_tx = {};
    }

    [[nodiscard]] bool in_static_mode() const { return (msg->flags & EVMC_STATIC) != 0; }

    const evmc_tx_context& get_tx_context() noexcept
    {
        if (INTX_UNLIKELY(m_tx.block_timestamp == 0))
            m_tx = host.get_tx_context();
        return m_tx;
    }
};
}  // namespace evmone
