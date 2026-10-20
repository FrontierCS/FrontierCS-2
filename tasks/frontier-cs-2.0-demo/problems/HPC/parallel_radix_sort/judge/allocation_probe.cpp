#define main benchmark_unused_main
#include "benchmark.cpp"
#undef main
#include <cilk/cilk_api.h>

static char *held[16];
static int worker[16];

int main() {
  const size_t piece = 5 * 1024 * 1024;
  begin_accounting();
  cilk_for (int i = 0; i < 16; i++) {
    worker[i] = __cilkrts_get_worker_number();
    held[i] = new char[piece];
    for (size_t j = 0; j < piece; j += 4096) held[i][j] = char(i);
  }
  const size_t worker_peak = end_accounting();
  for (int i = 0; i < 16; i++) delete[] held[i];
  std::sort(worker, worker + 16);
  const int worker_count = std::unique(worker, worker + 16) - worker;

  const size_t retained = 48 * 1024 * 1024;
  begin_accounting();
  held[0] = new char[retained];
  held[0][0] = 1;
  const size_t first = end_accounting();
  begin_accounting();
  held[1] = new char[retained];
  held[1][0] = 2;
  const size_t second = end_accounting();
  delete[] held[0];
  delete[] held[1];
  begin_accounting();
  const size_t after_free = end_accounting();

  std::cout << "{\"worker_peak\":" << worker_peak
            << ",\"worker_count\":" << worker_count
            << ",\"first_interval_peak\":" << first
            << ",\"second_interval_peak\":" << second
            << ",\"expected_live_peak\":" << 2 * retained
            << ",\"after_free\":" << after_free << "}\n";
}
