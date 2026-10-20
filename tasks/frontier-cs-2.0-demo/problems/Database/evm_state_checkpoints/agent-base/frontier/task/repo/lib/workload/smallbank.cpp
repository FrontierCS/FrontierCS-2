#include "smallbank.hpp"
#include <spectrum/common/hex.hpp>
#include <optional>
#include <glog/logging.h>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace spectrum {

const static char* CODE = 
    #include "../../contracts/smallbank.bin"
;

Smallbank::Smallbank(size_t num_elements, double zipf_exponent): 
    evm_type{EVMType::STRAWMAN},
    rng{std::unique_ptr<Random>(new ThreadLocalRandom([&]{return (zipf_exponent > 0.0 ? 
        std::unique_ptr<Random>(new Zipf(num_elements, zipf_exponent)) : 
        std::unique_ptr<Random>(new Unif(num_elements))
    );}, std::thread::hardware_concurrency()))}
{
    LOG(INFO) << fmt::format("Smallbank({}, {})", num_elements, zipf_exponent);
    this->code = spectrum::from_hex(std::string{CODE}).value();
}

void Smallbank::SetEVMType(EVMType ty) {
    this->evm_type = ty;
}

inline std::string to_string(uint32_t key) {
    auto ss = std::ostringstream();
    ss << std::setw(64) << std::setfill('0') << key;
    return ss.str();
}

Transaction Smallbank::Next() {
    DLOG(INFO) << "smallbank next" << std::endl;
    auto option = rng->Next() % 6;
    #define X to_string(rng->Next())
    auto input = spectrum::from_hex([&]{switch (option) {
        case 0: return std::string{"1e010439"} + X;
        case 1: return std::string{"bb27eb2c"} + X + X;
        case 2: return std::string{"ad0f98c0"} + X + X;
        case 3: return std::string{"83406251"} + X + X;
        case 4: return std::string{"8ac10b9c"} + X + X + X;
        case 5: return std::string{"97b63212"} + X + X;
        default: throw "unreachable";
    }}()).value();
    #undef X
    return Transaction(this->evm_type, evmc::address{0x1}, evmc::address{0x1}, std::span{code}, std::span{input});
}

} // namespace spectrum
