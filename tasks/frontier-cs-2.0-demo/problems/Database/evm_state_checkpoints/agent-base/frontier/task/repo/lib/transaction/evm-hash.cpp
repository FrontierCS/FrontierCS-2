#include <spectrum/transaction/evm-hash.hpp>
#include <cstddef>
#include <functional>
#include <tuple>

namespace spectrum
{

#define K std::tuple<evmc::address, evmc::bytes32>

/// @brief the hashing method
/// @param key the hashed key
/// @return the hash value
size_t KeyHasher::operator()(const K& key) const { 
    auto addr = std::get<0>(key);
    auto keyx = std::get<1>(key);
    size_t h = 0;
    for (auto x: addr.bytes) {
        h ^= std::hash<int>{}(x)  + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    for (auto x: keyx.bytes) {
        h ^= std::hash<int>{}(x)  + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

#undef K

} // namespace spectrum