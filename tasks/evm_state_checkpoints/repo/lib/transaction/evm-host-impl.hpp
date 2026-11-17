#pragma once
#include <evmc/evmc.hpp>

namespace spectrum {
using namespace evmc::literals;
using SetStorage = std::function<evmc_storage_status(
    const evmc::address &addr, const evmc::bytes32 &key,
    const evmc::bytes32 &value)>;
using GetStorage = std::function<evmc::bytes32(const evmc::address &addr,
                                               const evmc::bytes32 &key)>;

class Host : public evmc::Host {
    evmc_tx_context tx_context{};

  public:
    spectrum::GetStorage get_storage_inner; // these inner implementations can be externally set up
    spectrum::SetStorage set_storage_inner;
    explicit Host(evmc_tx_context &_tx_context) noexcept;
    bool account_exists(const evmc::address &addr) const noexcept final;
    evmc::bytes32 get_storage(const evmc::address &addr,
                              const evmc::bytes32 &key) const noexcept final;
    evmc_storage_status set_storage(const evmc::address &addr,
                                    const evmc::bytes32 &key,
                                    const evmc::bytes32 &value) noexcept final;
    evmc::uint256be get_balance(const evmc::address &addr) const noexcept final;
    size_t get_code_size(const evmc::address &addr) const noexcept final;
    evmc::bytes32 get_code_hash(const evmc::address &addr) const noexcept final;
    size_t copy_code(const evmc::address &addr, size_t code_offset,
                     uint8_t *buffer_data,
                     size_t buffer_size) const noexcept final;
    bool selfdestruct(const evmc::address &addr,
                      const evmc::address &beneficiary) noexcept final;
    evmc::Result call(const evmc_message &msg) noexcept final;
    evmc_tx_context get_tx_context() const noexcept final;
    evmc::bytes32 get_block_hash(int64_t number) const noexcept final;
    void emit_log(const evmc::address &addr, const uint8_t *data,
                  size_t data_size, const evmc::bytes32 topics[],
                  size_t topics_count) noexcept final;
    evmc_access_status access_account(const evmc::address &addr) noexcept final;
    evmc_access_status access_storage(const evmc::address &addr,
                                      const evmc::bytes32 &key) noexcept final;
    evmc::bytes32
    get_transient_storage(const evmc::address &addr,
                          const evmc::bytes32 &key) const noexcept override;
    void set_transient_storage(const evmc::address &addr,
                               const evmc::bytes32 &key,
                               const evmc::bytes32 &value) noexcept override;
};
} // namespace spectrum
