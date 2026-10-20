#pragma once
#include "trusted_isolation.h"
namespace fs2_timing {
static long invoke(bipartiteCSR& graph, bool use_v, long wedges, long max_wedges,
                   long max_array, CountType type, RankType rank) {
  const char* library = getenv("FS2_WORKER_LIBRARY");
  if (!library || library[0] != '/') fail("absolute worker library required");
  if (omp_get_max_threads() != 1) fail("trusted preprocessing must be single threaded before fork");
  if (prctl(PR_SET_DUMPABLE, 0)) fail("parent ptrace protection failed");
  int control[2], result[2];
  if (pipe2(control, O_CLOEXEC) || pipe2(result, O_CLOEXEC)) fail("pipe failed");
  fflush(nullptr);
  pid_t child = fork();
  if (child < 0) fail("fork failed");
  if (child == 0) {
    close(control[1]); close(result[0]);
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() == 1) _exit(116);
    int nullfd = open("/dev/null", O_RDWR);
    if (nullfd < 0 || dup2(nullfd, 0) < 0 || dup2(nullfd, 1) < 0 || dup2(nullfd, 2) < 0) _exit(117);
    close(nullfd);
    // No inherited judge descriptors remain accessible to candidate code.
    if (control[0] > 3 && syscall(__NR_close_range, 3, control[0] - 1, 0)) _exit(122);
    if (result[1] > control[0] + 1 && syscall(__NR_close_range, control[0] + 1, result[1] - 1, 0)) _exit(122);
    if (syscall(__NR_close_range, result[1] + 1, ~0U, 0)) _exit(122);
    clearenv();
    setenv("OMP_NUM_THREADS", "4", 1);
    setenv("OMP_DYNAMIC", "FALSE", 1);
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    filesystem(library);
    syscalls();
    char ready = 'R', go = 0;
    if (write(result[1], &ready, 1) != 1 || read(control[0], &go, 1) != 1 || go != 'G') _exit(118);
    close(control[0]);
    // Candidate constructors and all CountTotal code run only after the trusted clock starts.
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!handle) _exit(119);
    using Function = long (*)(bipartiteCSR*, bool, long, long, long, int, int);
    auto function = reinterpret_cast<Function>(dlsym(handle, "fs2_count_total"));
    if (!function) _exit(120);
    long count = function(&graph, use_v, wedges, max_wedges, max_array, type, rank);
    if (write(result[1], &count, sizeof(count)) != sizeof(count)) _exit(121);
    _exit(0);
  }
  close(control[0]); close(result[1]);
  char ready = 0;
  pollfd pending = {result[0], POLLIN, 0};
  if (poll(&pending, 1, 10000) != 1 || read(result[0], &ready, 1) != 1 || ready != 'R') {
    int status = 0;
    kill(child, SIGKILL); waitpid(child, &status, 0);
    fprintf(stderr, "worker setup status=%d\n", status);
    fail("worker isolation setup failed");
  }
  alarm(45);
  auto start = std::chrono::steady_clock::now();
  char go = 'G';
  if (write(control[1], &go, 1) != 1) fail("worker start failed");
  close(control[1]);
  long count = 0; size_t received = 0;
  while (received < sizeof(count)) {
    ssize_t n = read(result[0], reinterpret_cast<char*>(&count) + received, sizeof(count) - received);
    if (n <= 0) break;
    received += n;
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) fail("worker wait failed");
  auto end = std::chrono::steady_clock::now();
  alarm(0); close(result[0]);
  if (received != sizeof(count) || !WIFEXITED(status) || WEXITSTATUS(status)) {
    fprintf(stderr, "worker status=%d received=%zu\n", status, received);
    fail("isolated CountTotal failed");
  }
  double seconds = std::chrono::duration<double>(end - start).count();
  printf("FS2_TRUSTED_SECONDS=%.9f\n", seconds);
  return count;
}
}

long CountTotal(bipartiteCSR& graph, bool use_v, long wedges, long max_wedges,
                long max_array, CountType type, RankType rank) {
  return fs2_timing::invoke(graph, use_v, wedges, max_wedges, max_array, type, rank);
}
