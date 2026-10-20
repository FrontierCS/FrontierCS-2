//
// Created by endrias on 3/28/19.
//
#pragma once

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <tuple>
#include <utility>
#include <iostream>

namespace detail {
  template<typename count_type, typename It, typename OutIt, typename ExtractKey>

    inline bool to_unsigned_or_bool(bool b) {
    return b;
  }

  inline unsigned char to_unsigned_or_bool(unsigned char c) {
    return c;
  }

  inline unsigned char to_unsigned_or_bool(signed char c) {
    return static_cast<unsigned char>(c) + 128;
  }

  inline unsigned char to_unsigned_or_bool(char c) {
    return static_cast<unsigned char>(c);
  }

  inline std::uint16_t to_unsigned_or_bool(char16_t c) {
    return static_cast<std::uint16_t>(c);
  }

  inline std::uint32_t to_unsigned_or_bool(char32_t c) {
    return static_cast<std::uint32_t>(c);
  }

  inline std::uint32_t to_unsigned_or_bool(wchar_t c) {
    return static_cast<std::uint32_t>(c);
  }

  inline unsigned short to_unsigned_or_bool(short i) {
    return static_cast<unsigned short>(i) + static_cast<unsigned short>(1 << (sizeof(short) * 8 - 1));
  }

  inline unsigned short to_unsigned_or_bool(unsigned short i) {
    return i;
  }

  inline unsigned int to_unsigned_or_bool(int i) {
    return static_cast<unsigned int>(i) + static_cast<unsigned int>(1 << (sizeof(int) * 8 - 1));
  }

  inline unsigned int to_unsigned_or_bool(unsigned int i) {
    return i;
  }

  inline unsigned long to_unsigned_or_bool(long l) {
    return static_cast<unsigned long>(l) + static_cast<unsigned long>(1l << (sizeof(long) * 8 - 1));
  }

  inline unsigned long to_unsigned_or_bool(unsigned long l) {
    return l;
  }

  inline unsigned long long to_unsigned_or_bool(long long l) {
    return static_cast<unsigned long long>(l) + static_cast<unsigned long long>(1ll << (sizeof(long long) * 8 - 1));
  }

  inline unsigned long long to_unsigned_or_bool(unsigned long long l) {
    return l;
  }

  inline std::uint32_t to_unsigned_or_bool(float f) {
    union {
      float f;
      std::uint32_t u;
    } as_union = {f};
    std::uint32_t sign_bit = -std::int32_t(as_union.u >> 31);
    return as_union.u ^ (sign_bit | 0x80000000);
  }

  inline std::uint64_t to_unsigned_or_bool(double f) {
    union {
      double d;
      std::uint64_t u;
    } as_union = {f};
    std::uint64_t sign_bit = -std::int64_t(as_union.u >> 63);
    return as_union.u ^ (sign_bit | 0x8000000000000000);
  }

  template<typename T>
    inline size_t to_unsigned_or_bool(T *ptr) {
    return reinterpret_cast<size_t>(ptr);
  }



  template<size_t>
    struct UnsignedForSize;
  template<>
    struct UnsignedForSize<1>
    {
      typedef uint8_t type;
    };
  template<>
    struct UnsignedForSize<2>
    {
      typedef uint16_t type;
    };
  template<>
    struct UnsignedForSize<4>
    {
      typedef uint32_t type;
    };
  template<>
    struct UnsignedForSize<8>
    {
      typedef uint64_t type;
    };
}


