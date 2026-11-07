#pragma once
// The counting worker: the pristine CLI loads and preprocesses the graph, then loads the
// candidate's CountTotal library only after the judge starts the clock. The whole process is
// one shared isolate instance with syscalls="no_fork" (the judge's isolate.start):
// no judge files, no network, a private TMPDIR, threads but no new processes. The judge times
// it from its own side: "FS2_READY" on stdout -> 'G' on stdin -> the count record and the end
// of stdout. Candidate code never sees the judge's pipes on 0/1/2 (they are /dev/null), and
// whatever it writes to the record descriptor cannot be an earlier exact count than its own.
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace fs2_timing {
static void fail(const char* message) { throw std::runtime_error(message); }
static long invoke(bipartiteCSR& graph, bool use_v, long wedges, long max_wedges,
                   long max_array, CountType type, RankType rank) {
  const char* variable = getenv("FS2_WORKER_LIBRARY");
  if (!variable || variable[0] != '/') fail("absolute worker library required");
  const std::string library(variable);
  if (omp_get_max_threads() != 1) fail("trusted preprocessing must be single threaded");
  fflush(nullptr);
  int record = fcntl(1, F_DUPFD_CLOEXEC, 3);
  int nullfd = open("/dev/null", O_RDWR | O_CLOEXEC);
  if (record < 0 || nullfd < 0) fail("worker setup failed");
  static const char ready[] = "FS2_READY\n";
  if (write(record, ready, sizeof(ready) - 1) != sizeof(ready) - 1) fail("worker setup failed");
  char go = 0;
  if (read(0, &go, 1) != 1 || go != 'G') fail("worker start failed");
  if (dup2(nullfd, 0) < 0 || dup2(nullfd, 1) < 0 || dup2(nullfd, 2) < 0) _exit(117);
  close(nullfd);
  clearenv();
  setenv("OMP_NUM_THREADS", "4", 1);
  setenv("OMP_DYNAMIC", "FALSE", 1);
  omp_set_dynamic(0);
  omp_set_num_threads(4);
  // Candidate constructors and all CountTotal code run only after the judge's clock starts.
  void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) _exit(119);
  using Function = long (*)(bipartiteCSR*, bool, long, long, long, int, int);
  auto function = reinterpret_cast<Function>(dlsym(handle, "fs2_count_total"));
  if (!function) _exit(120);
  long count = function(&graph, use_v, wedges, max_wedges, max_array, type, rank);
  char line[64];
  int length = snprintf(line, sizeof(line), "FS2_COUNT=%ld\n", count);
  if (write(record, line, length) != length) _exit(121);
  _exit(0);  // no candidate destructors or exit handlers; the judge's clock stops at EOF
}
}

long CountTotal(bipartiteCSR& graph, bool use_v, long wedges, long max_wedges,
                long max_array, CountType type, RankType rank) {
  return fs2_timing::invoke(graph, use_v, wedges, max_wedges, max_array, type, rank);
}
