// Copyright (c) 2008, Jacob Burnim (jburnim@cs.berkeley.edu)
//
// This file is part of CREST, which is distributed under the revised
// BSD license.  A copy of this license can be found in the file LICENSE.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See LICENSE
// for details.
//
// search_harness: the trusted half of the concolic tester.  It is CREST's
// run_crest with the search strategy moved out of the process.  The harness
// keeps everything that decides the measurement -- constraint solving, input
// files, launching the program under test under resource limits, coverage
// and the iteration count -- and asks a sandboxed strategy process only which
// constraint of which earlier execution to negate next.
//
// Usage (run from a directory holding `branches` and `cfg_branches`):
//   search_harness PROGRAM_COMMAND BUDGET SEED STRATEGY_HOST RESULT_FILE
//
// Environment (all optional):
//   CREST_RELABEL_KEY       decimal 64-bit key of the per-run id permutation
//   CREST_EXEC_WALL_SECONDS outer wall limit of one launch (default 20)
//   CREST_EXEC_AS_KB        address-space limit of one launch (default 2000000)
//   CREST_COMPUTE_CPU_SECONDS  per-run compute budget: CPU seconds of the
//                           strategy process plus constraint solving
//                           (default 0 = none)
//   CREST_MAX_SOLVER_ATTEMPTS  per-run cap on solver attempts (default 0 = none)
//   CREST_MAX_NEGATE_REQUESTS  per-run cap on Negate requests, including those
//                           rejected without a solver query (default 0 = none)
//   CREST_RUN_WALL_SECONDS  per-run wall-clock budget: the run ends when it is
//                           spent, whatever the strategy is doing (default 0 = none)
//   CREST_STRATEGY_AS_KB    address-space limit of the strategy (default 1048576)
//   CREST_COVERAGE_FILE     write the covered instrumentation ids there at the end
//
// A run that exhausts the iteration budget or one of the caps above
// ends there and reports the coverage reached; none of them is an error.

#include <assert.h>
#include <errno.h>
#include <poll.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "base/basic_types.h"
#include "base/symbolic_execution.h"
#include "base/yices_solver.h"
#include "strategy/protocol.h"
#include "strategy/strategy_api.h"

using std::ifstream;
using std::ios;
using std::map;
using std::string;
using std::vector;

namespace crest {

namespace {

const int kCheckpoints[] = {100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000};

long long EnvInt(const char* name, long long fallback) {
  const char* v = getenv(name);
  if (!v || !*v) return fallback;
  return atoll(v);
}

double ThreadCpu() {
  struct timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

double Now() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

uint64_t SplitMix(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void SetLimit(int resource, rlim_t value) {
  struct rlimit lim = {value, value};
  setrlimit(resource, &lim);
}

}  // namespace


class Harness {
 public:
  Harness(const string& program, int budget, long long seed,
          const string& strategy_host, const string& result_file)
      : program_(program), budget_(budget), seed_(seed),
        strategy_host_(strategy_host), result_file_(result_file),
        num_iters_(0), total_num_covered_(0), solver_calls_(0), negate_requests_(0),
        infeasible_(0), incomplete_(0), next_exec_id_(1), peak_live_(0),
        child_(-1), to_child_(-1), from_child_(-1),
        strategy_cpu_seconds_(0.0), strategy_peak_kb_(0),
        solver_cpu_seconds_(0.0), solver_infeasible_cpu_seconds_(0.0),
        solver_infeasible_(0), program_wall_seconds_(0.0), run_time_expired_(false),
        program_cpu_seconds_(0.0), have_child_clock_(false) {
    exec_wall_ = static_cast<double>(EnvInt("CREST_EXEC_WALL_SECONDS", 20));
    exec_as_kb_ = EnvInt("CREST_EXEC_AS_KB", 2000000);
    compute_cpu_ = static_cast<double>(EnvInt("CREST_COMPUTE_CPU_SECONDS", 0));
    run_wall_cap_ = static_cast<double>(EnvInt("CREST_RUN_WALL_SECONDS", 0));
    strategy_as_kb_ = EnvInt("CREST_STRATEGY_AS_KB", 1048576);
    max_solver_calls_ = EnvInt("CREST_MAX_SOLVER_ATTEMPTS", 0);
    max_negate_requests_ = EnvInt("CREST_MAX_NEGATE_REQUESTS", 0);
    start_ = Now();
    ReadBranches();
    ReadCfg();
    BuildRelabelling();
  }

  // Returns the final status string.
  string Run() {
    if (!SpawnStrategy()) return Finish("harness_error:spawn");
    if (!SendInit()) return Finish(ChildFault("strategy_fault:init"));
    int ready = 0;
    if (!ReadInt(&ready) || ready != frontier::wire::kMagic) {
      if (run_time_expired_) return Finish(Kill("run_time_budget_exhausted"));
      int code = ReapChild();
      if (code == 71) return Finish("harness_error:sandbox_unavailable");
      return Finish("strategy_fault:init");
    }

    while (true) {
      int req = 0;
      if (!ReadInt(&req)) return Finish(ChildFault("strategy_fault:exited"));
      if (compute_cpu_ > 0 && ComputeUsed() >= compute_cpu_)
        return Finish(Kill("compute_budget_exhausted"));
      if (req == frontier::wire::kReqDone) {
        ReapChild();
        return Finish("strategy_returned");
      }
      if (req == frontier::wire::kReqRelease) {
        int id = 0;
        if (!ReadInt(&id)) return Finish(ChildFault("strategy_fault:exited"));
        map<int, SymbolicExecution*>::iterator it = live_.find(id);
        if (it == live_.end()) return Finish(Kill("strategy_fault:protocol"));
        delete it->second;
        live_.erase(it);
        continue;
      }
      if (req != frontier::wire::kReqRestart && req != frontier::wire::kReqNegate)
        return Finish(Kill("strategy_fault:protocol"));

      vector<value_t> input;
      if (req == frontier::wire::kReqNegate) {
        int id = 0, k = 0;
        if (!ReadInt(&id) || !ReadInt(&k)) return Finish(ChildFault("strategy_fault:exited"));
        map<int, SymbolicExecution*>::iterator it = live_.find(id);
        if (it == live_.end() || k < 0 ||
            static_cast<size_t>(k) >= it->second->path().constraints().size())
          return Finish(Kill("strategy_fault:protocol"));
        if (max_negate_requests_ > 0 && negate_requests_ >= max_negate_requests_)
          return Finish(Kill("negate_request_budget_exhausted"));
        negate_requests_++;
        if (max_solver_calls_ > 0 && solver_calls_ >= max_solver_calls_)
          return Finish(Kill("solver_attempt_budget_exhausted"));
        if (!SolveAtBranch(*it->second, static_cast<size_t>(k), &input)) {
          infeasible_++;
          int reply = frontier::wire::kReplyInfeasible;
          if (!WriteInts(&reply, 1)) return Finish(ChildFault("strategy_fault:exited"));
          continue;
        }
      }

      if (live_.size() >= static_cast<size_t>(frontier::Host::kMaxLiveExecutions))
        return Finish(Kill("strategy_fault:protocol"));
      if (num_iters_ >= budget_) {
        int reply = frontier::wire::kReplyBudgetExhausted;
        WriteInts(&reply, 1);
        ReapChild();
        return Finish("budget_exhausted");
      }
      if (run_wall_cap_ > 0 && Now() - start_ >= run_wall_cap_)
        return Finish(Kill("run_time_budget_exhausted"));

      SymbolicExecution* ex = new SymbolicExecution();
      bool completed = RunProgram(input, ex);
      UpdateCoverage(*ex);
      int id = next_exec_id_++;
      live_[id] = ex;
      peak_live_ = std::max(peak_live_, live_.size());
      if (!SendExecution(id, completed, *ex))
        return Finish(ChildFault("strategy_fault:exited"));
    }
  }

 private:
  const string program_;
  const int budget_;
  const long long seed_;
  const string strategy_host_;
  const string result_file_;

  // Original (instrumentation) ids.
  vector<branch_id_t> branches_;
  vector<branch_id_t> paired_branch_;
  vector<function_id_t> branch_function_;
  vector<vector<branch_id_t> > cfg_;
  branch_id_t max_branch_;
  function_id_t max_function_;

  // Relabelling: original id -> id shown to the strategy.
  vector<int> relabel_;
  vector<int> function_relabel_;

  int num_iters_;
  vector<bool> total_covered_;
  unsigned int total_num_covered_;
  long long solver_calls_;
  long long negate_requests_;
  long long infeasible_;
  long long incomplete_;
  vector<std::pair<int, unsigned int> > curve_;

  map<int, SymbolicExecution*> live_;
  int next_exec_id_;
  size_t peak_live_;

  pid_t child_;
  int to_child_;
  int from_child_;
  double strategy_cpu_seconds_;
  long strategy_peak_kb_;

  double solver_cpu_seconds_;
  double solver_infeasible_cpu_seconds_;
  long long solver_infeasible_;
  double program_wall_seconds_;
  double program_cpu_seconds_;
  bool have_child_clock_;
  clockid_t child_clock_;

  double exec_wall_;
  long long exec_as_kb_;
  double compute_cpu_;
  double run_wall_cap_;
  bool run_time_expired_;
  long long strategy_as_kb_;
  long long max_solver_calls_;
  long long max_negate_requests_;
  double start_;

  void ReadBranches() {
    max_branch_ = 0;
    vector<unsigned int> branch_count;
    branch_count.push_back(0);
    ifstream in("branches");
    if (!in) { fprintf(stderr, "search_harness: cannot read branches\n"); exit(2); }
    function_id_t fid;
    int num_branches;
    while (in >> fid >> num_branches) {
      branch_count.push_back(2 * num_branches);
      branch_id_t b1, b2;
      for (int i = 0; i < num_branches; i++) {
        if (!(in >> b1 >> b2)) { fprintf(stderr, "search_harness: bad branches file\n"); exit(2); }
        branches_.push_back(b1);
        branches_.push_back(b2);
        max_branch_ = std::max(max_branch_, std::max(b1, b2));
      }
    }
    max_branch_++;
    max_function_ = branch_count.size();
    paired_branch_.assign(max_branch_, 0);
    for (size_t i = 0; i + 1 < branches_.size(); i += 2) {
      paired_branch_[branches_[i]] = branches_[i + 1];
      paired_branch_[branches_[i + 1]] = branches_[i];
    }
    branch_function_.assign(max_branch_, 0);
    size_t i = 0;
    for (function_id_t j = 0; j < branch_count.size(); j++)
      for (size_t k = 0; k < branch_count[j]; k++)
        branch_function_[branches_[i++]] = j;
    total_covered_.assign(max_branch_, false);
    std::sort(branches_.begin(), branches_.end());
  }

  void ReadCfg() {
    cfg_.resize(max_branch_);
    ifstream in("cfg_branches", ios::in | ios::binary);
    if (!in) { fprintf(stderr, "search_harness: cannot read cfg_branches\n"); exit(2); }
    size_t num_branches = 0;
    in.read(reinterpret_cast<char*>(&num_branches), sizeof(num_branches));
    if (num_branches != branches_.size()) { fprintf(stderr, "search_harness: cfg mismatch\n"); exit(2); }
    for (size_t i = 0; i < num_branches; i++) {
      branch_id_t src;
      size_t len;
      in.read(reinterpret_cast<char*>(&src), sizeof(src));
      in.read(reinterpret_cast<char*>(&len), sizeof(len));
      if (!in || src < 0 || src >= max_branch_ || len > branches_.size()) {
        fprintf(stderr, "search_harness: bad cfg_branches\n"); exit(2);
      }
      cfg_[src].resize(len);
      if (len) in.read(reinterpret_cast<char*>(&cfg_[src].front()), len * sizeof(branch_id_t));
    }
  }

  // A fresh permutation of the branch ids and of the function ids per run,
  // determined by CREST_RELABEL_KEY alone when it is set and derived from the
  // seed otherwise.
  void BuildRelabelling() {
    uint64_t state = static_cast<uint64_t>(seed_) * 0xD1B54A32D192ED03ULL + 0x2545F4914F6CDD1DULL;
    const char* key = getenv("CREST_RELABEL_KEY");
    if (key && *key) state = strtoull(key, NULL, 10) ^ 0x2545F4914F6CDD1DULL;
    vector<int> order(branches_.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i) + 1;
    for (size_t i = order.size(); i > 1; i--)
      std::swap(order[i - 1], order[SplitMix(&state) % i]);
    relabel_.assign(max_branch_, 0);
    for (size_t i = 0; i < branches_.size(); i++) relabel_[branches_[i]] = order[i];
    vector<int> forder(max_function_);
    for (size_t i = 0; i < forder.size(); i++) forder[i] = static_cast<int>(i);
    for (size_t i = forder.size(); i > 1; i--)
      std::swap(forder[i - 1], forder[SplitMix(&state) % i]);
    function_relabel_ = forder;
  }

  int Relabel(branch_id_t b) const {
    if (b < 0) return b;  // call / return markers
    if (b >= max_branch_) return 0;
    return relabel_[b];
  }

  bool SpawnStrategy() {
    int down[2], up[2];
    if (pipe(down) != 0 || pipe(up) != 0) return false;
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
      dup2(down[0], 0);
      dup2(up[1], 1);
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) dup2(devnull, 2);
      for (int fd = 3; fd < 1024; fd++) close(fd);
      // The strategy alone can never use more than the whole compute budget.
      if (compute_cpu_ > 0) SetLimit(RLIMIT_CPU, static_cast<rlim_t>(compute_cpu_) + 1);
      SetLimit(RLIMIT_AS, static_cast<rlim_t>(strategy_as_kb_) * 1024);
      SetLimit(RLIMIT_CORE, 0);
      // A minimal environment: the strategy learns nothing from ours.
      char* const argv[] = {const_cast<char*>("strategy_host"), NULL};
      char* const envp[] = {NULL};
      execve(strategy_host_.c_str(), argv, envp);
      _exit(127);
    }
    close(down[0]);
    close(up[1]);
    child_ = pid;
    have_child_clock_ = (clock_getcpuclockid(pid, &child_clock_) == 0);
    to_child_ = down[1];
    from_child_ = up[0];
    return true;
  }

  // CPU seconds used so far by the strategy process.
  double StrategyCpu() {
    if (child_ > 0 && have_child_clock_) {
      struct timespec ts;
      if (clock_gettime(child_clock_, &ts) == 0)
        strategy_cpu_seconds_ =
            static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
    }
    return strategy_cpu_seconds_;
  }

  double ComputeUsed() { return StrategyCpu() + solver_cpu_seconds_; }

  // Reads one integer from the strategy.  Gives up when the run's wall-clock
  // budget is spent, so a strategy that computes without asking for anything
  // cannot keep a run alive.
  bool ReadInt(int* v) {
    char* p = reinterpret_cast<char*>(v);
    size_t n = sizeof(int);
    while (n > 0) {
      if (run_wall_cap_ > 0) {
        double left = start_ + run_wall_cap_ - Now();
        struct pollfd pfd;
        pfd.fd = from_child_;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int ready = left > 0 ? poll(&pfd, 1, static_cast<int>(left * 1000.0) + 1) : 0;
        if (ready < 0 && errno == EINTR) continue;
        if (ready == 0) { run_time_expired_ = true; return false; }
      }
      ssize_t r = read(from_child_, p, n);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) return false;
      p += r;
      n -= static_cast<size_t>(r);
    }
    return true;
  }

  bool WriteAll(const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
      ssize_t r = write(to_child_, p, n);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) return false;
      p += r;
      n -= static_cast<size_t>(r);
    }
    return true;
  }

  bool WriteInts(const int* v, size_t n) { return WriteAll(v, n * sizeof(int)); }

  bool SendInit() {
    vector<int> out;
    uint64_t rng = static_cast<uint64_t>(seed_) * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
    uint64_t strategy_seed = SplitMix(&rng);
    out.push_back(frontier::wire::kMagic);
    out.push_back(static_cast<int>(strategy_seed & 0xffffffffu));
    out.push_back(static_cast<int>(strategy_seed >> 32));
    out.push_back(budget_);
    int num_ids = static_cast<int>(branches_.size()) + 1;
    out.push_back(num_ids);
    out.push_back(static_cast<int>(branches_.size()));
    for (int b = 1; b < num_ids; b++) out.push_back(b);
    vector<int> paired(num_ids, 0), function_of(num_ids, 0);
    for (size_t i = 0; i < branches_.size(); i++) {
      branch_id_t b = branches_[i];
      paired[relabel_[b]] = relabel_[paired_branch_[b]];
      function_of[relabel_[b]] = function_relabel_[branch_function_[b]];
    }
    out.insert(out.end(), paired.begin(), paired.end());
    out.insert(out.end(), function_of.begin(), function_of.end());
    vector<vector<int> > successors(num_ids);
    for (size_t i = 0; i < branches_.size(); i++) {
      branch_id_t b = branches_[i];
      vector<int>& s = successors[relabel_[b]];
      for (size_t j = 0; j < cfg_[b].size(); j++) {
        int t = Relabel(cfg_[b][j]);
        if (t > 0) s.push_back(t);
      }
      std::sort(s.begin(), s.end());
    }
    for (int b = 1; b < num_ids; b++) {
      out.push_back(b);
      out.push_back(static_cast<int>(successors[b].size()));
      out.insert(out.end(), successors[b].begin(), successors[b].end());
    }
    return WriteInts(&out[0], out.size());
  }

  bool SendExecution(int id, bool completed, const SymbolicExecution& ex) {
    const vector<branch_id_t>& path = ex.path().branches();
    const vector<size_t>& idx = ex.path().constraints_idx();
    vector<int> out;
    out.reserve(path.size() + idx.size() + 8);
    out.push_back(frontier::wire::kReplyExecution);
    out.push_back(id);
    out.push_back(completed ? 1 : 0);
    out.push_back(num_iters_);
    out.push_back(static_cast<int>(path.size()));
    for (size_t i = 0; i < path.size(); i++) out.push_back(Relabel(path[i]));
    out.push_back(static_cast<int>(idx.size()));
    for (size_t i = 0; i < idx.size(); i++) out.push_back(static_cast<int>(idx[i]));
    return WriteInts(&out[0], out.size());
  }

  // Waits for the strategy process and returns its exit code, or -signal.
  int ReapChild() {
    if (child_ <= 0) return 0;
    if (to_child_ >= 0) { close(to_child_); to_child_ = -1; }
    int status = 0;
    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    pid_t r;
    do { r = wait4(child_, &status, 0, &usage); } while (r < 0 && errno == EINTR);
    child_ = -1;
    if (r > 0) strategy_cpu_seconds_ =
        static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
        1e-6 * static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
    strategy_peak_kb_ = usage.ru_maxrss;
    if (WIFSIGNALED(status)) return -WTERMSIG(status);
    return WEXITSTATUS(status);
  }

  string Kill(const string& status) {
    if (child_ > 0) kill(child_, SIGKILL);
    ReapChild();
    return status;
  }

  // The strategy process went away without saying so: classify how.
  string ChildFault(const string& fallback) {
    if (run_time_expired_) return Kill("run_time_budget_exhausted");
    int code = ReapChild();
    // Killed by its CPU limit: the compute budget is spent, the run ends.
    if ((code == -SIGXCPU || code == -SIGKILL) && compute_cpu_ > 0 &&
        strategy_cpu_seconds_ >= compute_cpu_ - 1.0)
      return "compute_budget_exhausted";
    if (code == -SIGSYS) return "strategy_fault:sandbox_kill";
    if (code < 0) return "strategy_fault:crashed";
    if (code == 72) return "strategy_fault:api_misuse";
    if (code == 71) return "harness_error:sandbox_unavailable";
    if (code == 127) return "harness_error:spawn";
    return fallback;
  }

  void WriteInput(const vector<value_t>& input) {
    FILE* f = fopen("input", "w");
    if (!f) { perror("search_harness: input"); exit(2); }
    for (size_t i = 0; i < input.size(); i++) fprintf(f, "%lld\n", input[i]);
    fclose(f);
  }

  // Launches the program under test once, under hard limits, and waits for
  // it.  The command itself applies the per-execution timeout; the limits
  // here are the backstop for anything that escapes it.
  void LaunchProgram(const vector<value_t>& input) {
    WriteInput(input);
    unlink("szd_execution");

    char seed_buf[64];
    unsigned long long z = static_cast<unsigned long long>(seed_) * 0x9E3779B97F4A7C15ULL +
                           static_cast<unsigned long long>(num_iters_) * 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 31;
    snprintf(seed_buf, sizeof(seed_buf), "%u", static_cast<unsigned>(z & 0x7fffffff));

    sigset_t chld, old;
    sigemptyset(&chld);
    sigaddset(&chld, SIGCHLD);
    sigprocmask(SIG_BLOCK, &chld, &old);

    pid_t pid = fork();
    if (pid < 0) { perror("search_harness: fork"); exit(2); }
    if (pid == 0) {
      sigprocmask(SIG_SETMASK, &old, NULL);
      setpgid(0, 0);
      if (to_child_ >= 0) close(to_child_);
      if (from_child_ >= 0) close(from_child_);
      int devnull = open("/dev/null", O_RDWR);
      if (devnull >= 0) { dup2(devnull, 0); dup2(devnull, 1); dup2(devnull, 2); }
      SetLimit(RLIMIT_AS, static_cast<rlim_t>(exec_as_kb_) * 1024);
      SetLimit(RLIMIT_CORE, 0);
      SetLimit(RLIMIT_CPU, static_cast<rlim_t>(exec_wall_) + 5);
      SetLimit(RLIMIT_FSIZE, static_cast<rlim_t>(1) << 30);
      prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
      setenv("CREST_PROG_SEED", seed_buf, 1);
      unsetenv("CREST_RELABEL_KEY");
      execl("/bin/sh", "sh", "-c", program_.c_str(), static_cast<char*>(NULL));
      _exit(127);
    }
    setpgid(pid, pid);

    struct rusage before;
    getrusage(RUSAGE_CHILDREN, &before);
    double launched = Now();
    double deadline = launched + exec_wall_;
    while (true) {
      int status = 0;
      pid_t r = waitpid(pid, &status, WNOHANG);
      if (r == pid) break;
      if (r < 0 && errno != EINTR) break;
      double left = deadline - Now();
      if (left <= 0) {
        kill(-pid, SIGKILL);
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        break;
      }
      struct timespec ts;
      ts.tv_sec = static_cast<time_t>(left);
      ts.tv_nsec = static_cast<long>((left - static_cast<double>(ts.tv_sec)) * 1e9);
      sigtimedwait(&chld, NULL, &ts);
    }
    // Nothing started by the launch may outlive it.
    kill(-pid, SIGKILL);
    sigprocmask(SIG_SETMASK, &old, NULL);
    struct rusage after;
    getrusage(RUSAGE_CHILDREN, &after);
    program_wall_seconds_ += Now() - launched;
    program_cpu_seconds_ +=
        static_cast<double>((after.ru_utime.tv_sec - before.ru_utime.tv_sec) +
                            (after.ru_stime.tv_sec - before.ru_stime.tv_sec)) +
        1e-6 * static_cast<double>((after.ru_utime.tv_usec - before.ru_utime.tv_usec) +
                                   (after.ru_stime.tv_usec - before.ru_stime.tv_usec));
  }

  // Returns false when the program left no readable trace (killed, crashed).
  // A launch always costs one iteration.
  bool RunProgram(const vector<value_t>& input, SymbolicExecution* ex) {
    LaunchProgram(input);
    num_iters_++;
    ifstream in("szd_execution", ios::in | ios::binary);
    if (in && ex->Parse(in)) return true;
    incomplete_++;
    SymbolicExecution empty;
    ex->Swap(empty);
    return false;
  }

  void UpdateCoverage(const SymbolicExecution& ex) {
    const vector<branch_id_t>& path = ex.path().branches();
    for (size_t i = 0; i < path.size(); i++) {
      branch_id_t b = path[i];
      if (b > 0 && b < max_branch_ && !total_covered_[b]) {
        total_covered_[b] = true;
        total_num_covered_++;
      }
    }
    for (size_t i = 0; i < sizeof(kCheckpoints) / sizeof(kCheckpoints[0]); i++)
      if (kCheckpoints[i] == num_iters_)
        curve_.push_back(std::make_pair(num_iters_, total_num_covered_));
    if (num_iters_ % 100 == 0) WriteResult("running");
  }

  // CREST's Search::SolveAtBranch, unchanged in meaning.
  bool SolveAtBranch(const SymbolicExecution& ex, size_t branch_idx, vector<value_t>* input) {
    const vector<SymbolicPred*>& constraints = ex.path().constraints();
    // If an earlier constraint is identical, negating this one is infeasible.
    for (int i = static_cast<int>(branch_idx) - 1; i >= 0; i--) {
      if (constraints[branch_idx]->Equal(*constraints[i])) return false;
    }
    vector<const SymbolicPred*> cs(constraints.begin(), constraints.begin() + branch_idx + 1);
    map<var_t, value_t> soln;
    solver_calls_++;
    constraints[branch_idx]->Negate();
    bool success = false;
    double solve_started = ThreadCpu();
    try {
      success = YicesSolver::IncrementalSolve(ex.inputs(), ex.vars(), cs, &soln);
    } catch (...) {
      success = false;  // solver gave up: treated as infeasible
    }
    constraints[branch_idx]->Negate();
    double spent = ThreadCpu() - solve_started;
    solver_cpu_seconds_ += spent;
    if (!success) {
      solver_infeasible_++;
      solver_infeasible_cpu_seconds_ += spent;
      return false;
    }
    *input = ex.inputs();
    for (map<var_t, value_t>::const_iterator i = soln.begin(); i != soln.end(); ++i) {
      if (i->first < input->size()) (*input)[i->first] = i->second;
    }
    return true;
  }

  void WriteResult(const string& status) {
    string tmp = result_file_ + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f, "{\"status\": \"%s\", \"iterations\": %d, \"budget\": %d, \"covered\": %u, "
               "\"negate_requests\": %lld, \"solver_attempts\": %lld, \"solver_infeasible_attempts\": %lld, "
               "\"solver_infeasible_cpu_seconds\": %.3f, \"infeasible\": %lld, "
               "\"incomplete_executions\": %lld, "
               "\"peak_live_executions\": %zu, \"strategy_cpu_seconds\": %.3f, "
               "\"solver_cpu_seconds\": %.3f, \"program_cpu_seconds\": %.3f, "
               "\"program_wall_seconds\": %.3f, "
               "\"strategy_peak_kb\": %ld, \"wall_seconds\": %.3f, \"curve\": [",
            status.c_str(), num_iters_, budget_, total_num_covered_, negate_requests_, solver_calls_,
            solver_infeasible_,
            solver_infeasible_cpu_seconds_, infeasible_,
            incomplete_, peak_live_, StrategyCpu(), solver_cpu_seconds_, program_cpu_seconds_,
            program_wall_seconds_, strategy_peak_kb_, Now() - start_);
    for (size_t i = 0; i < curve_.size(); i++)
      fprintf(f, "%s[%d, %u]", i ? ", " : "", curve_[i].first, curve_[i].second);
    fprintf(f, "]}\n");
    fclose(f);
    rename(tmp.c_str(), result_file_.c_str());
  }

  string Finish(const string& status) {
    if (child_ > 0) Kill(status);
    const char* dump = getenv("CREST_COVERAGE_FILE");  // debugging aid, original ids
    if (dump && *dump) {
      FILE* f = fopen(dump, "w");
      if (f) {
        for (size_t i = 0; i < branches_.size(); i++)
          if (total_covered_[branches_[i]]) fprintf(f, "%d\n", branches_[i]);
        fclose(f);
      }
    }
    WriteResult(status);
    return status;
  }
};

}  // namespace crest


int main(int argc, char* argv[]) {
  if (argc != 6) {
    fprintf(stderr, "usage: search_harness PROGRAM_COMMAND BUDGET SEED STRATEGY_HOST RESULT_FILE\n");
    return 2;
  }
  signal(SIGPIPE, SIG_IGN);
  crest::SetLimit(RLIMIT_CORE, 0);
  // The harness draws no random numbers itself; libcrest inside the program
  // under test is seeded per launch from SEED and the iteration number.
  crest::Harness harness(argv[1], atoi(argv[2]), atoll(argv[3]), argv[4], argv[5]);
  std::string status = harness.Run();
  fprintf(stderr, "search_harness: %s\n", status.c_str());
  return 0;
}
