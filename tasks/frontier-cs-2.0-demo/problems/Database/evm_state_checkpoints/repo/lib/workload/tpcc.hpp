#include <spectrum/workload/abstraction.hpp>
#include <spectrum/common/random.hpp>
#include <map>

namespace spectrum {

class TPCC : public Workload {

    private:
    std::basic_string<uint8_t>  code;
    EVMType                     evm_type;
    std::unique_ptr<Random>     rng;
    size_t                      num_items;
    size_t                      num_orders;
    std::atomic<size_t>         order_count{0};
    std::atomic<size_t>         delivery_count{0};
    std::atomic<bool>           is_first_transaction;
    std::basic_string<uint8_t> CreateTable();
    std::basic_string<uint8_t> Payment();
    std::basic_string<uint8_t> Delivery();
    std::basic_string<uint8_t> NewOrder();

    public:
    TPCC(size_t num_items, size_t num_orders);
    Transaction Next(); // override;
    void SetEVMType(EVMType ty); // override;

};

} // namespace spectrum
