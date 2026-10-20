#include <ostream>
#include <spectrum/transaction/evm-host-impl.hpp>
#include <spectrum/transaction/evm-transaction.hpp>
#include <spectrum/common/hex.hpp>
#include <spectrum/evmcow/baseline.hpp>
#include <spectrum/evmone/baseline.hpp>
#include <spectrum/evmcow/vm.hpp>
#include <spectrum/evmone/vm.hpp>
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include <variant>
#include <iostream>
#include <span>
#include <glog/logging.h>
#include <fmt/core.h>
#include <stdexcept>

namespace spectrum {

/// @brief create a EVMType enum value from given string
/// @param s BASIC | STRAWMAN | CHECKPOINTED
/// @return the indicated EVMType
EVMType ParseEVMType(std::basic_string_view<char> s) {
    if (s == "BASIC")       { return EVMType::BASIC; }
    if (s == "STRAWMAN")    { return EVMType::STRAWMAN; }
    if (s == "CHECKPOINTED") { return EVMType::CHECKPOINTED; }
    throw std::runtime_error(std::string{fmt::format("unknown evmtype {}", s)});
}

// constructor for transaction object
Transaction::Transaction(
    EVMType evm_type, 
    evmc::address from, 
    evmc::address to, 
    std::span<uint8_t> code,
    std::span<uint8_t> input
):
    input(input.begin(), input.end()),
    evm_type{evm_type},
    vm{(evm_type == EVMType::BASIC || evm_type == EVMType::STRAWMAN) ? 
        std::variant<evmone::VM, evmcow::VM>(evmone::VM()) : 
        std::variant<evmone::VM, evmcow::VM>(evmcow::VM())
    },
    tx_context{evmc_tx_context{
        .block_number = 42,
        .block_timestamp = 66,
        .block_gas_limit = 10000000000,
    }},
    host{Host(this->tx_context)},
    code{code}
{
    this->message = evmc_message{
        .kind = EVMC_CALL,
        .depth = 0,
        .gas = 999999999,
        .recipient = to,
        .sender = from,
        .input_data = &this->input[0],
        .input_size = this->input.size(),
        .value{0},
    };
    mm_count += 32 * 1024;
}

// update set_storage handler
void Transaction::InstallSetStorageHandler(spectrum::SetStorage&& handler) {
    host.set_storage_inner = handler;
}

// update get_storage handler
void Transaction::InstallGetStorageHandler(spectrum::GetStorage&& handler) {
    host.get_storage_inner = handler;
}

/// @brief making checkpoint
/// @return checkpoint id, see Transaction::ApplyCheckpoint
size_t Transaction::MakeCheckpoint() {
    DLOG(INFO) << "transaction make checkpoint " << std::endl;
    // can only be called inside execution
    if (evm_type == EVMType::BASIC) {
        mm_count += 32 * 1024;
        return 0;
    }
    if (evm_type == EVMType::STRAWMAN) {
        auto& _vm = std::get<evmone::VM>(vm);
        mm_count += 32 * 1024;
        _vm.checkpoints.push_back(std::make_unique<evmone::ExecutionState>(*_vm.state.value()));
        return _vm.checkpoints.size() - 1;
    }
    if (evm_type == EVMType::CHECKPOINTED) {
        auto& _vm = std::get<evmcow::VM>(vm);
        mm_count += sizeof(evmcow::Checkpoint);
        _vm.checkpoints.push_back(_vm.state.value()->save_checkpoint());
        return _vm.checkpoints.size() - 1;
    }
    return 0;
}

/// @brief making checkpoint
/// @param checkpoint_id the checkpoint id to go back to
void Transaction::ApplyCheckpoint(size_t checkpoint_id) {
    DLOG(INFO) << "transaction apply checkpoint " << checkpoint_id << std::endl;
    FlushOperations();
    if (evm_type == EVMType::BASIC) {
        vm.emplace<evmone::VM>();
        return;
    }
    if (evm_type == EVMType::STRAWMAN) {
        auto& _vm = std::get<evmone::VM>(vm);
        _vm.state = std::make_unique<evmone::ExecutionState>(*_vm.checkpoints[checkpoint_id]);
        _vm.checkpoints.resize(checkpoint_id);
        return;
    }
    if (evm_type == EVMType::CHECKPOINTED) {
        auto& _vm = std::get<evmcow::VM>(vm);
        _vm.state.value()->load_checkpoint(_vm.checkpoints[checkpoint_id]);
        _vm.checkpoints.resize(checkpoint_id);
        return;
    }
}

/// @brief break current execution
void Transaction::Break() {
    DLOG(INFO) << "transaction break" << std::endl;
    // can only be called inside execution
    if (evm_type == EVMType::BASIC || evm_type == EVMType::STRAWMAN) {
        auto& _vm = std::get<evmone::VM>(vm);
        _vm.state->get()->will_break = true;
    }
    if (evm_type == EVMType::CHECKPOINTED) {
        auto& _vm = std::get<evmcow::VM>(vm);
        _vm.state->get()->will_break = true;
    }
}

/// @brief execute a transaction
void Transaction::Execute() {
    DLOG(INFO) << "transaction execute" << std::endl;
    if (evm_type == EVMType::BASIC || evm_type == EVMType::STRAWMAN) {
        auto& _vm = std::get<evmone::VM>(vm);
        auto host_interface = &host.get_interface();
        auto host_context   = host.to_context();
        const auto result = evmone::baseline::execute(
            _vm, host_interface, host_context, 
            EVMC_SHANGHAI, &message,
            &code[0], code.size() - 1
        );
        if (result.status_code != evmc_status_code::EVMC_SUCCESS) {
            LOG(ERROR) << "function hash: " << to_hex(std::span{&input[0], 4}) <<  " transaction status: " << result.status_code << std::endl;
        }
        if (result.output_data) { result.release(&result); }
        return;
    }
    if (evm_type == EVMType::CHECKPOINTED) {
        auto& _vm = std::get<evmcow::VM>(vm);
        auto host_interface = &host.get_interface();
        auto host_context   = host.to_context();
        const auto result = evmcow::baseline::execute(
            _vm, host_interface, host_context,
            EVMC_SHANGHAI, &message,
            &code[0], code.size() - 1
        );
        if (result.status_code != evmc_status_code::EVMC_SUCCESS) {
            LOG(ERROR) << "transaction status: " << result.status_code << std::endl;
        }
        if (result.output_data) {
            result.release(&result);
        }
        return;
    }
    LOG(FATAL) << "not possible";
}

/// @brief run the transaction once, append read and write keys into the prediction struct
void Transaction::Analyze(Prediction& prediction) {
    // store current get storage and set storage handler
    auto _get_storage_handler = host.get_storage_inner;
    auto _set_storage_handler = host.set_storage_inner;
    this->InstallGetStorageHandler([&prediction](auto& address, auto& key) {
        prediction.get.push_back({address, key});
        return evmc::bytes32{0};
    });
    this->InstallSetStorageHandler([&prediction](auto& address, auto& key, auto& /* value */) {
        prediction.put.push_back({address, key});
        return evmc_storage_status::EVMC_STORAGE_MODIFIED;
    });
    // execute the evmone once with basic strategy
    auto _vm = evmone::VM();
    auto result = evmone::baseline::execute(
        _vm, &host.get_interface(), host.to_context(),
        EVMC_SHANGHAI, &message,
        &code[0], code.size() - 1
    );
    if (result.output_data) { result.release(&result); }
    // restore access storage handlers
    host.get_storage_inner = _get_storage_handler;
    host.set_storage_inner = _set_storage_handler;
}

/// @brief flush operations from inner vm to transaction, useful when vm is exchanged
void Transaction::FlushOperations() {
    if (evm_type == EVMType::BASIC || evm_type == EVMType::STRAWMAN) {
        auto& _vm = std::get<evmone::VM>(vm);
        op_count += _vm.op_count;
        _vm.op_count = 0; return;
    }
    if (evm_type == EVMType::CHECKPOINTED) {
        auto& _vm = std::get<evmcow::VM>(vm);
        op_count += _vm.op_count;
        _vm.op_count = 0; return;
    }
}

/// @brief return currently executed #operations
size_t Transaction::CountOperations() {
    FlushOperations();
    size_t _op_count = op_count; op_count = 0;
    return _op_count;
}

/// @brief result of evmc
/// @param result 
Result::Result(evmc_result result):
    evmc_result{result}
{}

Result::~Result() {
    if (this->output_data) {
        this->release(this);
    }
}

} // namespace spectrum
