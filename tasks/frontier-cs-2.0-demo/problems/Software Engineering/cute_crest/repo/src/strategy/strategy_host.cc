// Entry point of the sandboxed strategy process.  Fixed harness code.
//
// The process is started by bin/search_harness with a pipe on descriptors 0
// and 1.  It reads the static program description, drops every privilege it
// does not need (no new file descriptors, no child processes, a seccomp
// filter that leaves only read/write on the two pipes, memory management and
// exit), and only then constructs and runs the solution's strategy.

#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <map>
#include <vector>

#include "strategy/protocol.h"
#include "strategy/strategy_api.h"

namespace frontier {

namespace {

void Die(int code) { _exit(code); }

// Exit codes understood by the harness.
const int kExitOk = 0;
const int kExitProtocol = 70;
const int kExitSandbox = 71;
const int kExitApiMisuse = 72;

void ReadAll(void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  while (n > 0) {
    ssize_t r = read(0, p, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) Die(kExitProtocol);
    p += r;
    n -= static_cast<size_t>(r);
  }
}

void WriteAll(const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  while (n > 0) {
    ssize_t r = write(1, p, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) Die(kExitProtocol);
    p += r;
    n -= static_cast<size_t>(r);
  }
}

int32_t ReadInt() {
  int32_t v;
  ReadAll(&v, sizeof(v));
  return v;
}

void ReadInts(std::vector<int>* out, size_t n) {
  out->resize(n);
  if (n) ReadAll(&(*out)[0], n * sizeof(int));
}

void WriteInts(const int32_t* v, size_t n) { WriteAll(v, n * sizeof(int32_t)); }

// Leaves the process able to do nothing but compute and talk on the two
// pipes.  Any other system call kills it; the harness records that as a
// strategy fault.
bool EnterSandbox() {
  struct rlimit zero = {0, 0};
  if (setrlimit(RLIMIT_NOFILE, &zero) != 0) return false;
  if (setrlimit(RLIMIT_NPROC, &zero) != 0) return false;
  if (setrlimit(RLIMIT_FSIZE, &zero) != 0) return false;
  if (setrlimit(RLIMIT_CORE, &zero) != 0) return false;
  if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) return false;
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) return false;

#define ALLOW(nr) \
  BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), \
  BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)

  struct sock_filter filter[] = {
    // Architecture check.
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
    // read(0, ...) only.
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_read, 0, 4),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    // write(1, ...) only.
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_write, 0, 4),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    // mmap: anonymous memory without PROT_EXEC only.
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_mmap, 0, 7),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[2])),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, PROT_EXEC, 4, 0),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[3])),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, MAP_ANONYMOUS, 0, 2),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    // Memory management and exit.
    ALLOW(SYS_brk),
    ALLOW(SYS_munmap),
    ALLOW(SYS_mremap),
    ALLOW(SYS_madvise),
    ALLOW(SYS_exit),
    ALLOW(SYS_exit_group),
    ALLOW(SYS_rt_sigreturn),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
  };
#undef ALLOW

  struct sock_fprog prog;
  prog.len = static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0]));
  prog.filter = filter;
  if (syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) != 0) return false;
  return true;
}

}  // namespace

class HostImpl {
 public:
  Program program;
  int budget;
  int used;
  int num_covered;
  std::vector<char> covered;
  uint64_t rng;
  std::map<int, Execution*> live;

  HostImpl() : budget(0), used(0), num_covered(0), rng(0) {}

  void ReadInit() {
    if (ReadInt() != wire::kMagic) Die(kExitProtocol);
    uint32_t seed_lo = static_cast<uint32_t>(ReadInt());
    uint32_t seed_hi = static_cast<uint32_t>(ReadInt());
    rng = (static_cast<uint64_t>(seed_hi) << 32) | seed_lo;
    budget = ReadInt();
    program.num_branch_ids = ReadInt();
    int n = ReadInt();
    if (program.num_branch_ids < 1 || n < 0) Die(kExitProtocol);
    ReadInts(&program.branches, static_cast<size_t>(n));
    ReadInts(&program.paired, static_cast<size_t>(program.num_branch_ids));
    ReadInts(&program.function_of, static_cast<size_t>(program.num_branch_ids));
    program.successors.resize(static_cast<size_t>(program.num_branch_ids));
    for (int i = 0; i < n; i++) {
      int b = ReadInt();
      int len = ReadInt();
      if (b < 1 || b >= program.num_branch_ids || len < 0) Die(kExitProtocol);
      ReadInts(&program.successors[static_cast<size_t>(b)], static_cast<size_t>(len));
    }
    covered.assign(static_cast<size_t>(program.num_branch_ids), 0);
  }

  const Execution* ReadReply() {
    int status = ReadInt();
    if (status == wire::kReplyBudgetExhausted) Die(kExitOk);
    if (status == wire::kReplyInfeasible) return NULL;
    if (status != wire::kReplyExecution) Die(kExitProtocol);
    Execution* e = new Execution;
    e->id = ReadInt();
    e->completed = ReadInt() != 0;
    used = ReadInt();
    int path_len = ReadInt();
    if (path_len < 0) Die(kExitProtocol);
    ReadInts(&e->path, static_cast<size_t>(path_len));
    int num_constraints = ReadInt();
    if (num_constraints < 0) Die(kExitProtocol);
    e->constraint_pos.resize(static_cast<size_t>(num_constraints));
    if (num_constraints)
      ReadAll(&e->constraint_pos[0], static_cast<size_t>(num_constraints) * sizeof(uint32_t));
    for (size_t i = 0; i < e->path.size(); i++) {
      int b = e->path[i];
      if (b > 0 && b < program.num_branch_ids && !covered[static_cast<size_t>(b)]) {
        covered[static_cast<size_t>(b)] = 1;
        num_covered++;
      }
    }
    live[e->id] = e;
    return e;
  }

  void CheckRoom() {
    if (live.size() >= static_cast<size_t>(Host::kMaxLiveExecutions)) Die(kExitApiMisuse);
  }
};

const Program& Host::program() const { return impl_->program; }
int Host::budget() const { return impl_->budget; }
int Host::iterations_used() const { return impl_->used; }

bool Host::covered(int branch) const {
  return branch > 0 && branch < impl_->program.num_branch_ids &&
         impl_->covered[static_cast<size_t>(branch)] != 0;
}

int Host::num_covered() const { return impl_->num_covered; }

uint32_t Host::Random() {
  // splitmix64
  uint64_t z = (impl_->rng += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<uint32_t>(z >> 32);
}

uint32_t Host::Uniform(uint32_t n) {
  if (n == 0) Die(kExitApiMisuse);
  // Lemire's multiply-shift with rejection: exactly uniform.
  uint64_t m = static_cast<uint64_t>(Random()) * n;
  uint32_t low = static_cast<uint32_t>(m);
  if (low < n) {
    uint32_t threshold = (0u - n) % n;
    while (low < threshold) {
      m = static_cast<uint64_t>(Random()) * n;
      low = static_cast<uint32_t>(m);
    }
  }
  return static_cast<uint32_t>(m >> 32);
}

const Execution* Host::Restart() {
  impl_->CheckRoom();
  int32_t req = wire::kReqRestart;
  WriteInts(&req, 1);
  const Execution* e = impl_->ReadReply();
  if (!e) Die(kExitProtocol);
  return e;
}

const Execution* Host::Negate(const Execution* e, size_t k) {
  if (!e) Die(kExitApiMisuse);
  std::map<int, Execution*>::iterator it = impl_->live.find(e->id);
  if (it == impl_->live.end() || it->second != e) Die(kExitApiMisuse);
  if (k >= e->constraint_pos.size()) Die(kExitApiMisuse);
  impl_->CheckRoom();
  int32_t req[3] = {wire::kReqNegate, e->id, static_cast<int32_t>(k)};
  WriteInts(req, 3);
  return impl_->ReadReply();
}

void Host::Release(const Execution* e) {
  if (!e) return;
  std::map<int, Execution*>::iterator it = impl_->live.find(e->id);
  if (it == impl_->live.end() || it->second != e) Die(kExitApiMisuse);
  int32_t req[2] = {wire::kReqRelease, e->id};
  WriteInts(req, 2);
  delete it->second;
  impl_->live.erase(it);
}

}  // namespace frontier

int main() {
  frontier::HostImpl impl;
  impl.ReadInit();
  if (!frontier::EnterSandbox()) frontier::Die(frontier::kExitSandbox);
  // Tell the harness the sandbox is in place before any solution code runs.
  int32_t ready = frontier::wire::kMagic;
  frontier::WriteInts(&ready, 1);

  frontier::Host host(&impl);
  frontier::Strategy* strategy = frontier::MakeStrategy();
  if (!strategy) frontier::Die(frontier::kExitApiMisuse);
  strategy->Run(host);
  int32_t done = frontier::wire::kReqDone;
  frontier::WriteInts(&done, 1);
  frontier::Die(frontier::kExitOk);
  return 0;
}
