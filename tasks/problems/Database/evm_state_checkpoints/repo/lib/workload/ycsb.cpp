#include "ycsb.hpp"
#include "evmc/evmc.hpp"
#include "spectrum/transaction/evm-hash.hpp"
#include <spectrum/common/hex.hpp>
#include <fmt/core.h>
#include <glog/logging.h>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace spectrum {

const static char* CODE = 
    #include "../../contracts/ycsb.bin"
;

evmc::bytes32 hexStringToBytes32(const std::string& hexString) {
    if (hexString.size() != 64) {
        throw std::invalid_argument("Input hex string must be exactly 64 characters long.");
    }

    evmc::bytes32 result;
    for (size_t i = 0, j = 0; i < 32; ++i, j += 2) {
        std::string byteString = hexString.substr(j, 2);
        result.bytes[i] = std::stoi(byteString, nullptr, 16);
    }

    return result;
}

const static std::string predicated_keys[] = {
    "ad3228b676f7d3cd4284a5443f17f1962b36e491b30a40b2405849e597ba5fb5",
    "ada5013122d395ba3c54772283fb069b10426056ef8ca54750cb9bb552a59e7d",
    "abbb5caa7dda850e60932de0934eb1f9d0f59695050f761dc64e443e5030a569",
    "101e368776582e57ab3d116ffe2517c0a585cd5b23174b01e275c2d8329c3d83",
    "52d75039926638d3c558b2bdefb945d5be8dae29dedd1c313212a4d472d9fde5",
    "2b232c97452f0950c94e2539fdc7e69d21166113cf7a9bcb99b220a3fe5d720a",
    "62103cf3131c85df57aad364d21cba02556d3092d6cb54c298c2e7726a7870bd",
    "870253054e3d98b71abec8fff9ebf8a15d167f15909091a800d4acaab9266d2b",
    "5b8b9143058ba3a137192c563ca6541845e62f0a2f9a667aac4db2fa3c334e3c",
    "324fdf7bfe7bd2828491073f0b7868a9a19ee3eff384c2805040be3e426447f5",
    "020abee21eef15c21bc31a406c2b8ac3afc5df94a4b02b38abb286f4334e6c5b",
    "a29f2962b8badecbf4d3036e28fcd7dcf22db126f130193790f7698ee4d3dd84",
    "1cb7ce0668e72b96f704af9e1445a9dc6f6ac599eec355bfcfe4d3befbb001be",
    "50a82f9cbcdfaca82fe46b4a494d325ee6dc33d1fa55b218ab142e6cc2c8a58b",
    "9998fe8c12a1a1395171fc2449145bb1f0c273bfc80ab4ea62eb7a9cb439450c",
    "52774d722ab93275a0199da6072cca5400bf7f03bf064dd4a2b1af238c418d49",
    "b44b86596a635358e7aa60b17d32860c3f1efe2d3e53fb82c0bb23213b9c4be3",
    "dc275f13e83bcad5305f77e8f2f06c8d9840ee8b7d606ee958f86f59784b2de3",
    "1da244b7f8b81d82e17fde46fbf307da20557945243b38ef4c87c9487b59901b",
    "b0bad370a213ac7dcb3dfe3423b8d60077054da2a57d974f5e9768ef98fd60b6",
    "569e75fc77c1a856f6daaf9e69d8a9566ca34aa47f9133711ce065a571af0cfd",
    "20b8703968a1421b7417fbc8615cbe909dc21c5d6ab0d6fd4579bae332adc937",
    "660960054c44c4d83c420af65c54e6f3437517cffaa9996495231fee01bdba41",
    "2a400e8b310fbd326e4c5c4aed5c8212a6b2bd98d80dbef0077b1b2ee20907d4",
    "faaf1897615a4d5824a81780f33dd422a304cae5e7b14f0f9215d1a3deeea9e2",
    "d554518f31b5833712aff068198f0745641f5708162a7286bfaa31c5f461f1ba",
    "d54e89dee95b843939c00dce8c49063df67efe61a3c8ec818ccaee72d7b7f5a3",
    "d6866844f1a8c30f9968e80fa46697288da4b41c675b67a51469639e5afc2435",
    "5bbfe49fc46bd3c0efd63509af9eb2a12636d52d97f4d1cf52bf53227ef389c2",
    "49bfe3c65dd9f3794050d39474bec15417a1630abed1727c133877e89823c14f",
    "28c3dd0e618e8bddf8cf30cd1a2e788f719d9c1f097596a24acbe0c7cb1e05c6",
    "a20a1f31e7e0c47a407717c0b73e822e9ce414e0fc1925c4df69c43f77ac765e",
    "9e91c95f108ead30341468db9696e39ed81de6a68bc8c6702bcb66393fce6e75",
    "3884095ca26754c19839d1d65ed3564b4558896672ae6763c55fd1245d56ef7b",
    "34c816dab9c312a23fa45b66dbf9c10a2294a4455b7a50973d5e33fd47c39f4a",
    "0de8d66b2bd125dbab6c2314a8cd08ca7e2ac381b265aefa7ce27c5d9d75d933",
    "e6bccefd92fc2fa71227cbd31f39b085fabc5c0f7b7d07eb4a639c53ad5822f4",
    "4e349fca397a2203dbb3e457a82dc8ae1a5129362fdfcdae82619128ca92c033",
    "463154dffd5ddce82baa6ddaa825ffda4b4cc1dd8abd5af72f421c487b1335db",
    "a68e398350be805d9560b1ec3307f811e9f6dda9bf8906288e5b598ec4ea1928",
    "55a5bd2f1561a275dba97a3e23bf64b586a74cc4c4a2dc9f1c147ffdbbce1122"
};

YCSB::YCSB(size_t num_elements, double zipf_exponent): 
    evm_type{EVMType::STRAWMAN},
    rng{std::unique_ptr<Random>(new ThreadLocalRandom([&]{return (zipf_exponent > 0.0 ? 
        std::unique_ptr<Random>(new Zipf(num_elements, zipf_exponent)) : 
        std::unique_ptr<Random>(new Unif(num_elements))
    );}, std::thread::hardware_concurrency()))}
{
    LOG(INFO) << fmt::format("YCSB({}, {})", num_elements, zipf_exponent);
    this->code = spectrum::from_hex(std::string{CODE}).value();
    for(int i=0; i<=20; i++){
        pred_keys[i] = hexStringToBytes32(predicated_keys[i]);
    }
}

void YCSB::SetEVMType(EVMType ty) { this->evm_type = ty; }

inline std::string to_string(uint32_t key) {
    auto ss = std::ostringstream();
    ss << std::setw(64) << std::setfill('0') << key;
    return ss.str();
}

Transaction YCSB::Next() {
    DLOG(INFO) << "ycsb next" << std::endl;
    auto predicted_get_storage = std::unordered_set<std::tuple<evmc::address, evmc::bytes32>, KeyHasher>();
    auto predicted_set_storage = std::unordered_set<std::tuple<evmc::address, evmc::bytes32>, KeyHasher>();
    auto input = spectrum::from_hex([&]() {
        //  10 key 5 read 5 write(may be blind)
        auto s = std::string{"f3d7af72"};
        auto v = std::vector<size_t>(11, 0);
        SampleUniqueN(*rng, v);
        for (int i = 0; i <= 10; i++) {
            s += to_string(v[i]);
            if(v[i] >= 10 && v[i] <= 20){
                switch (i % 2) {
                    case 0: predicted_get_storage.insert({evmc::address{0x1}, pred_keys[v[i]]}); break;
                    case 1: predicted_set_storage.insert({evmc::address{0x1}, pred_keys[v[i]]}); break;
                }
            }
        }
        return s;
    }()).value();
    #undef X
    auto tx = Transaction(this->evm_type, evmc::address{0x1}, evmc::address{0x1}, std::span{code}, std::span{input});
    tx.predicted_get_storage = predicted_get_storage;
    tx.predicted_set_storage = predicted_set_storage;
    return tx;
}

} // namespace spectrum