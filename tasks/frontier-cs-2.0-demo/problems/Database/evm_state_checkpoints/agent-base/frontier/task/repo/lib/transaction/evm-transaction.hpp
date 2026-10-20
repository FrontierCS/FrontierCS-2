#pragma once
#include "spectrum/transaction/evm-hash.hpp"
#include <spectrum/transaction/evm-host-impl.hpp>
#include <spectrum/evmcow/baseline.hpp>
#include <spectrum/evmcow/vm.hpp>
#include <spectrum/evmone/baseline.hpp>
#include <spectrum/evmone/vm.hpp>
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <fmt/core.h>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <variant>
#include <vector>

namespace spectrum {

#define K std::tuple<evmc::address, evmc::bytes32>

enum EVMType { BASIC = 0, STRAWMAN = 1, CHECKPOINTED = 2 };

EVMType ParseEVMType(std::basic_string_view<char> s);

/// @brief the evmc_result that automatically destructs itself
struct Result : public evmc_result {

    Result(evmc_result result);
    ~Result();
};

/// @brief a structure tailored for Transaction::Analyze to dump r/w keys into
struct Prediction {
    std::vector<K> get; // the read keys
    std::vector<K> put; // the write keys
};

/// @brief the base class for evm transactions, providing CHECKPOINTED, BASIC, STRAWMAN mode for mini-checkpointing
class Transaction {

    private:
    std::variant<evmone::VM, evmcow::VM> vm;
    spectrum::Host host;
    spectrum::EVMType evm_type;
    evmc_tx_context tx_context;
    std::span<uint8_t> code;
    std::vector<uint8_t> input;
    evmc_message message;
    size_t  op_count{0};

    public:
    size_t  mm_count{0};
    std::unordered_set<K, KeyHasher>  predicted_get_storage;
    std::unordered_set<K, KeyHasher>  predicted_set_storage;
    Transaction(EVMType evm_type, evmc::address from, evmc::address to,
                std::span<uint8_t> code, std::span<uint8_t> input);
    void InstallSetStorageHandler(spectrum::SetStorage &&handler);
    void InstallGetStorageHandler(spectrum::GetStorage &&handler);
    void Analyze(Prediction& prediction);
    void Execute();
    void Break();
    void ApplyCheckpoint(size_t checkpoint_id);
    size_t MakeCheckpoint();

    size_t CountOperations();
    void   FlushOperations();

};

} // namespace spectrum

/// @brief specify template fmt::formatter for formatting EVMType values
template <> struct fmt::formatter<spectrum::EVMType> {
    template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(spectrum::EVMType const &value, FormatContext &ctx) const {
        #define OPT(X) case spectrum::EVMType::X: return fmt::format_to(ctx.out(), "{}", #X);
        switch (value) { OPT(BASIC) OPT(STRAWMAN) OPT(CHECKPOINTED) }
        #undef OPT
        throw std::runtime_error("unreachable");
    }
};

#undef K