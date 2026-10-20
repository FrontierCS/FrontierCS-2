#pragma once
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>
#include <omp.h>
#include <sys/resource.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdexcept>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs2_timing {
static void fail(const char* message) { throw std::runtime_error(message); }

static void filesystem(const char* library, bool compiler = false) {
  int abi = syscall(__NR_landlock_create_ruleset, nullptr, 0, LANDLOCK_CREATE_RULESET_VERSION);
  if (abi < 3) _exit(110);
  landlock_ruleset_attr attr = {};
  attr.handled_access_fs = (1ULL << 15) - 1;
  int rules = syscall(__NR_landlock_create_ruleset, &attr, sizeof(attr), 0);
  if (rules < 0) _exit(111);
  auto allow = [&](const char* path, unsigned long long access, bool required) {
    int fd = open(path, O_PATH | O_CLOEXEC);
    if (fd < 0) { if (required) _exit(112); else return; }
    landlock_path_beneath_attr rule = {};
    rule.parent_fd = fd; rule.allowed_access = access;
    if (syscall(__NR_landlock_add_rule, rules, LANDLOCK_RULE_PATH_BENEATH, &rule, 0)) _exit(113);
    close(fd);
  };
  const auto read_tree = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR | LANDLOCK_ACCESS_FS_EXECUTE;
  allow(library, compiler ? attr.handled_access_fs : LANDLOCK_ACCESS_FS_READ_FILE, true);
  for (const char* path : {"/usr/lib", "/lib", "/lib64", "/sys/devices/system/cpu"}) allow(path, read_tree, false);
  allow("/etc/ld.so.cache", LANDLOCK_ACCESS_FS_READ_FILE, false);
  if (compiler) {
    allow("/usr/bin", read_tree, true);
    allow("/usr/include", read_tree, true);
    allow("/usr/libexec/gcc", read_tree, false);
    allow("/dev/null", LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE, true);
  }
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) ||
      syscall(__NR_landlock_restrict_self, rules, 0)) _exit(114);
  close(rules);
}

#define FS2_ALLOW(name) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_##name, 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
static void syscalls() {
  // Threads may compute; process creation/control, networking and ptrace cannot escape the worker.
  sock_filter filter[] = {
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
#ifdef FS2_COMPILER_SANDBOX
    FS2_ALLOW(execve), FS2_ALLOW(wait4), FS2_ALLOW(vfork), FS2_ALLOW(fork), FS2_ALLOW(clone),
    FS2_ALLOW(pipe), FS2_ALLOW(pipe2), FS2_ALLOW(dup), FS2_ALLOW(dup2), FS2_ALLOW(dup3),
    FS2_ALLOW(fcntl), FS2_ALLOW(getcwd), FS2_ALLOW(umask), FS2_ALLOW(unlink), FS2_ALLOW(unlinkat),
    FS2_ALLOW(rename), FS2_ALLOW(renameat), FS2_ALLOW(chmod), FS2_ALLOW(fchmod), FS2_ALLOW(ioctl),
    FS2_ALLOW(getppid), FS2_ALLOW(uname), FS2_ALLOW(ftruncate), FS2_ALLOW(stat), FS2_ALLOW(lstat),
#endif
    FS2_ALLOW(read), FS2_ALLOW(write), FS2_ALLOW(close), FS2_ALLOW(fstat),
    FS2_ALLOW(newfstatat), FS2_ALLOW(statx), FS2_ALLOW(lseek), FS2_ALLOW(pread64),
    FS2_ALLOW(openat), FS2_ALLOW(access), FS2_ALLOW(readlink), FS2_ALLOW(readlinkat),
    FS2_ALLOW(getdents64), FS2_ALLOW(mmap), FS2_ALLOW(mprotect), FS2_ALLOW(munmap),
    FS2_ALLOW(brk), FS2_ALLOW(madvise), FS2_ALLOW(rt_sigaction), FS2_ALLOW(rt_sigprocmask),
    FS2_ALLOW(rt_sigreturn), FS2_ALLOW(sigaltstack), FS2_ALLOW(clock_gettime),
    FS2_ALLOW(clock_getres), FS2_ALLOW(gettimeofday), FS2_ALLOW(time),
    FS2_ALLOW(getpid), FS2_ALLOW(gettid), FS2_ALLOW(getuid), FS2_ALLOW(geteuid),
    FS2_ALLOW(getgid), FS2_ALLOW(getegid), FS2_ALLOW(futex), FS2_ALLOW(sched_yield),
    FS2_ALLOW(sched_getaffinity), FS2_ALLOW(set_robust_list), FS2_ALLOW(rseq),
    FS2_ALLOW(nanosleep), FS2_ALLOW(clock_nanosleep), FS2_ALLOW(getrandom),
    FS2_ALLOW(arch_prctl), FS2_ALLOW(getrusage), FS2_ALLOW(getrlimit),
    FS2_ALLOW(set_tid_address), FS2_ALLOW(restart_syscall), FS2_ALLOW(exit), FS2_ALLOW(exit_group),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone3, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_prlimit64, 0, 8),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 5),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[2])),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 3),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[2]) + 4),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 0, 5),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[0])),
    BPF_STMT(BPF_ALU | BPF_AND | BPF_K, CLONE_THREAD),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, CLONE_THREAD, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
  };
  sock_fprog program = {static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0])), filter};
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program)) _exit(115);
}
#undef FS2_ALLOW

}
