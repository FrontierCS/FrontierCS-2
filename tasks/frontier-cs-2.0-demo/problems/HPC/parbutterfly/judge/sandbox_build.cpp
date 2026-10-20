#define FS2_COMPILER_SANDBOX
#include "trusted_isolation.h"
int main(int argc, char** argv) {
  if (argc < 3 || argv[1][0] != '/') return 125;
  // The Verifier command can legitimately be PID 1 in its container.
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) || prctl(PR_SET_DUMPABLE, 0)) return 125;
  rlimit cpu = {180, 180}, memory = {8ULL << 30, 8ULL << 30}, files = {64ULL << 20, 64ULL << 20}, core = {0, 0};
  if (setrlimit(RLIMIT_CPU, &cpu) || setrlimit(RLIMIT_AS, &memory) || setrlimit(RLIMIT_FSIZE, &files) || setrlimit(RLIMIT_CORE, &core)) return 125;
  if (syscall(__NR_close_range, 3, ~0U, 0)) return 125;
  clearenv();
  setenv("PATH", "/usr/bin:/bin", 1); setenv("LANG", "C", 1); setenv("TMPDIR", argv[1], 1);
  fs2_timing::filesystem(argv[1], true); fs2_timing::syscalls();
  execv(argv[2], argv + 2);
  return 125;
}
