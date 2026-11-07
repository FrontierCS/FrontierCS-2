// evmcow: Fast Ethereum Virtual Machine implementation
// Copyright 2021 The evmcow Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "./execution_state.hpp"
#include "./tracing.hpp"
#include "./baseline.hpp"
#include <evmc/evmc.h>
#include <optional>
#include <vector>
#include <memory>

#if defined(_MSC_VER) && !defined(__clang__)
#define EVMONE_CGOTO_SUPPORTED 0
#else
#define EVMONE_CGOTO_SUPPORTED 1
#endif

namespace evmcow
{
/// The evmcow EVMC instance.
class VM : public evmc_vm
{
public:
    std::optional<std::unique_ptr<evmcow::ExecutionState>>  state{std::nullopt};
    std::vector<evmcow::Checkpoint>                         checkpoints{};
    std::unique_ptr<evmcow::baseline::CodeAnalysis>         analysis{nullptr};
    bool cgoto = EVMONE_CGOTO_SUPPORTED;
    bool validate_eof = false;

    size_t op_count{0};

private:
    std::unique_ptr<Tracer> m_first_tracer;

public:
    VM() noexcept;

    void add_tracer(std::unique_ptr<Tracer> tracer) noexcept
    {
        // Find the first empty unique_ptr and assign the new tracer to it.
        auto* end = &m_first_tracer;
        while (*end)
            end = &(*end)->m_next_tracer;
        *end = std::move(tracer);
    }

    [[nodiscard]] Tracer* get_tracer() const noexcept { return m_first_tracer.get(); }
};
}  // namespace evmcow
