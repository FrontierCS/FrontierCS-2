#include <spectrum/workload/abstraction.hpp>
#include <spectrum/common/random.hpp>

namespace spectrum {

class YCSB: public Workload {

    private:
    std::basic_string<uint8_t>  code;
    EVMType                     evm_type;
    std::unique_ptr<Random>     rng;
    evmc::bytes32               pred_keys[21];
    public:
    YCSB(size_t num_elements, double zipf_exponent);
    Transaction Next() override;
    void SetEVMType(EVMType ty) override;

};

};