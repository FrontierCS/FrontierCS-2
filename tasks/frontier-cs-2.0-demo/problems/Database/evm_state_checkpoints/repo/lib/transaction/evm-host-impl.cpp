#include <spectrum/transaction/evm-host-impl.hpp>
#include <evmc/evmc.hpp>
#include <iostream>
#include <glog/logging.h>

namespace spectrum {

using namespace evmc::literals;
using SetStorage = std::function<evmc_storage_status(
    const evmc::address &addr, const evmc::bytes32 &key,
    const evmc::bytes32 &value)>;
using GetStorage = std::function<evmc::bytes32(const evmc::address &addr,
                                               const evmc::bytes32 &key)>;

Host::Host(evmc_tx_context &_tx_context) noexcept : tx_context{_tx_context} {}

bool Host::account_exists(const evmc::address &addr) const noexcept {
    return true;
}

evmc::bytes32 Host::get_storage(const evmc::address &addr,
                                const evmc::bytes32 &key) const noexcept {
    return get_storage_inner(addr, key);
}

evmc_storage_status Host::set_storage(const evmc::address &addr,
                                      const evmc::bytes32 &key,
                                      const evmc::bytes32 &value) noexcept {
    return set_storage_inner(addr, key, value);
}

evmc::uint256be Host::get_balance(const evmc::address &addr) const noexcept {
    (void)addr;
    return {};
}

size_t Host::get_code_size(const evmc::address &addr) const noexcept {
    (void)addr;
    return 0;
}

evmc::bytes32 Host::get_code_hash(const evmc::address &addr) const noexcept {
    (void)addr;
    return {};
}

size_t Host::copy_code(const evmc::address &addr, size_t code_offset,
                       uint8_t *buffer_data,
                       size_t buffer_size) const noexcept {
    (void)addr;
    (void)code_offset;
    (void)buffer_data;
    (void)buffer_size;
    return 0;
}

bool Host::selfdestruct(const evmc::address &addr,
                        const evmc::address &beneficiary) noexcept {
    (void)addr;
    (void)beneficiary;
    return false;
}

evmc::Result Host::call(const evmc_message &msg) noexcept {
    DLOG(INFO) << "call";
    return evmc::Result{EVMC_REVERT, msg.gas, 0, msg.input_data,
                        msg.input_size};
}

evmc_tx_context Host::get_tx_context() const noexcept { return tx_context; }

evmc::bytes32 Host::get_block_hash(int64_t number) const noexcept {
    const int64_t current_block_number = get_tx_context().block_number;

    return (number < current_block_number &&
            number >= current_block_number - 256)
               ? 0xb10c8a5fb10c8a5fb10c8a5fb10c8a5fb10c8a5fb10c8a5fb10c8a5fb10c8a5f_bytes32
               : 0x0000000000000000000000000000000000000000000000000000000000000000_bytes32;
}

void Host::emit_log(const evmc::address &addr, const uint8_t *data,
                    size_t data_size, const evmc::bytes32 topics[],
                    size_t topics_count) noexcept {
    (void)addr;
    (void)data;
    (void)data_size;
    (void)topics;
    (void)topics_count;
}

evmc_access_status Host::access_account(const evmc::address &addr) noexcept {
    (void)addr;
    return EVMC_ACCESS_COLD;
}

evmc_access_status Host::access_storage(const evmc::address &addr,
                                        const evmc::bytes32 &key) noexcept {
    (void)addr;
    (void)key;
    return EVMC_ACCESS_COLD;
}

evmc::bytes32
Host::get_transient_storage(const evmc::address &addr,
                            const evmc::bytes32 &key) const noexcept {
    return get_storage_inner(addr, key);
}

void Host::set_transient_storage(const evmc::address &addr,
                                 const evmc::bytes32 &key,
                                 const evmc::bytes32 &value) noexcept {
    set_storage_inner(addr, key, value);
}

} // namespace spectrum
