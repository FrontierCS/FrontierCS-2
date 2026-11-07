#include <spectrum/workload/tpcc.hpp>
#include <spectrum/common/hex.hpp>
#include <fmt/core.h>
#include <glog/logging.h>
#include <optional>
#include <iomanip>
#include <iostream>
#include <sstream>


const static char* CODE = 
    #include "../../contracts/tpcc.bin"
;

namespace spectrum {

// we fix warehouse = 0, district = 0 now
TPCC::TPCC(size_t num_items, size_t num_orders):
    evm_type{EVMType::STRAWMAN},
    rng{std::unique_ptr<Random>(new ThreadLocalRandom([&]{return (
        std::unique_ptr<Random>(new Unif(num_items))
    );}, std::thread::hardware_concurrency()))},
    num_orders{num_orders}
{
    LOG(INFO) << fmt::format("TPCC({}, {})", num_items, num_orders);
    this->code = spectrum::from_hex(std::string{CODE}).value();
}

void TPCC::SetEVMType(EVMType ty) { this->evm_type = ty; }

inline std::string to_hex_string(uint32_t key) {
    auto ss = std::ostringstream();
    ss << std::hex << std::setw(64) << std::setfill('0') << key;
    return ss.str();
}

inline size_t non_uniform(Random &rng, size_t A, size_t x, size_t y) {
    return (((rng.Next() % A) | (rng.Next() % y + x)) % (y - x + 1)) + x;
}

std::basic_string<uint8_t> TPCC::NewOrder() { return spectrum::from_hex([&]() {
    auto s = std::string{"fb2bdc7d"};
    // fix warehouse = 0
    auto w_id = to_hex_string(0);
    // fix district = 0
    auto d_id = to_hex_string(0);
    s = s + w_id + d_id;
    // customer id are random now
    auto c_id = non_uniform(*rng, 1023, 1, 3000);
    // order entry date is the order count
    auto o_entry_d = order_count.fetch_add(1);
    s = s + to_hex_string(c_id) + to_hex_string(o_entry_d);
    // compute last three params index
    // the first index are fixed: 224
    auto i_ids_idx_num = 224;
    // auto num_orders = rng->Next() % 11 + 5;
    auto i_w_ids_idx_num = i_ids_idx_num + 32 * (num_orders + 1);
    auto i_qtys_idx_num = i_w_ids_idx_num + 32 * (num_orders + 1);
    auto i_ids_idx = to_hex_string(i_ids_idx_num);
    auto i_w_ids_idx = to_hex_string(i_w_ids_idx_num);
    auto i_qtys_idx = to_hex_string(i_qtys_idx_num);
    s = s + i_ids_idx + i_w_ids_idx + i_qtys_idx;
    // generate num_orders random items
    auto i_ids = std::vector<size_t>(num_orders, 0);
    for (int i = 0; i < num_orders; i++) { i_ids[i] = non_uniform(*rng, 8191, 1, 100000); }
    s += to_hex_string(num_orders);
    for (int i = 0; i < num_orders; i++) { s += to_hex_string(i_ids[i]); }
    // fix all items' warehouse id to 0
    auto i_w_ids = std::vector<size_t>(num_orders, 0);
    s += to_hex_string(num_orders);
    for (int i = 0; i < num_orders; i++) { s += to_hex_string(i_w_ids[i]); }
    // fix all items' quantity to 1
    auto i_qtys = std::vector<size_t>(num_orders, 1);
    s += to_hex_string(num_orders);
    for (int i = 0; i < num_orders; i++) { s += to_hex_string(i_qtys[i]); }
    // LOG(ERROR) << s;
    return s;
}()).value(); }

std::basic_string<uint8_t> TPCC::Delivery() { return spectrum::from_hex([&]() {
    auto s = std::string{"2690e6b3"};
    // fix warehouse = 0
    auto w_id = to_hex_string(0);
    // fix district = 0
    auto d_id = to_hex_string(0);
    // o_id is which not delivery
    auto o_id = delivery_count.fetch_add(10);
    if (delivery_count.load() > order_count.load() - 10) { 
        o_id = rng->Next() % order_count.load(); 
    }
    // fix delivery date = 0
    auto o_delivery_d = to_hex_string(0);
    s = s + w_id + d_id + to_hex_string(o_id) + o_delivery_d;
    // LOG(ERROR) << s;
    return s;
}()).value(); }

std::basic_string<uint8_t> TPCC::Payment() { return spectrum::from_hex([&]() {
    auto s = std::string{"7c3f9309"};
    // customer id and h_amount are random now, h_amount is in [0, 1000)
    auto c_id = rng->Next();
    auto h_amount = rng->Next() % 1000;
    return s + to_hex_string(c_id) + to_hex_string(h_amount);
}()).value(); }

Transaction TPCC::Next() {
    auto option = rng->Next() % 23;
    if (option < 11) {
        auto input = NewOrder();
        return Transaction(this->evm_type, evmc::address{0x1}, evmc::address{0x1}, std::span{code}, std::span{input});
    } 
    else if (option < 12) {
        auto input = Delivery();
        return Transaction(this->evm_type, evmc::address{0x1}, evmc::address{0x1}, std::span{code}, std::span{input});
    } 
    else {
        auto input = Payment();
        return Transaction(this->evm_type, evmc::address{0x1}, evmc::address{0x1}, std::span{code}, std::span{input});
    }
}

} // namespace spectrum
