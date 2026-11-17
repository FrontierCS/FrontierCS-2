#pragma once

#include <cstddef>
#include <cstdint>

namespace DB {

class Interface {
public:
  virtual void Init(size_t n, size_t m, uint64_t* elems) = 0;
  virtual void Close() { }
  virtual int Query(uint64_t l, uint64_t r) = 0;
  virtual int Insert(uint64_t key) = 0;
  virtual int Delete(uint64_t key) = 0;
  virtual ~Interface() { }
};

}
