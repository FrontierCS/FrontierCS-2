#pragma once

#include <cstdint>
#include <random>
#include <type_traits>
#include <x86intrin.h>

// utilize AES method to generate perfect hash for integers
class aes_hash {
private:
  static __m128i aes_keygen() {
    static std::mt19937 _rand(std::random_device{}());
    return _mm_set_epi32(_rand(), _rand(), _rand(), _rand());
  }

  static inline uint64_t hash64(uint64_t x) {
    static const __m128i _key = aes_keygen();
    static const __m128i _key64 = aes_keygen();
    __m128i r = _mm_set1_epi64x(x);
    r = _mm_aesenc_si128(r, _key);
    r = _mm_aesenc_si128(r, _key64);
    return *reinterpret_cast<uint64_t*>(&r);
  }

  static inline uint32_t hash32(uint32_t x) {
    static const __m128i _key = aes_keygen();
    __m128i r = _mm_set1_epi32(x);
    r = _mm_aesenc_si128(r, _key);
    return *reinterpret_cast<uint32_t*>(&r);
  }

  static inline uint16_t hash16(uint16_t x) {
    static const __m128i _key = aes_keygen();
    __m128i r = _mm_set1_epi16(x);
    r = _mm_aesenc_si128(r, _key);
    return *reinterpret_cast<uint16_t*>(&r);
  }

  static inline uint8_t hash8(uint8_t x) {
    static const __m128i _key = aes_keygen();
    __m128i r = _mm_set1_epi8(x);
    r = _mm_aesenc_si128(r, _key);
    return *reinterpret_cast<uint8_t*>(&r);
  }

public:
  template <typename T>
  static inline auto hash(T x) {
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) { return hash8(x); }
    else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return hash16(x); }
    else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return hash32(x); }
    else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return hash64(x); }
    else { return std::void_t<T>(); }
  }
}; // class aes_hash
