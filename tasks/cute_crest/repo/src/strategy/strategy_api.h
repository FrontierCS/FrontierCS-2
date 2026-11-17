// Search-strategy interface of the concolic tester.
//
// This header is part of the fixed harness.  A solution implements
// frontier::MakeStrategy() in src/frontier_solution/ and nothing else.
//
// The strategy runs in its own sandboxed process.  It sees executions only
// as sequences of branch ids and decides which symbolic branch of which
// earlier execution to negate next, or asks for a restart on fresh random
// inputs.  Constraint solving, input generation, launching the program under
// test, and counting coverage and iterations all happen in the harness
// process, on the other side of a pipe.

#ifndef STRATEGY_STRATEGY_API_H__
#define STRATEGY_STRATEGY_API_H__

#include <stddef.h>
#include <stdint.h>
#include <vector>

namespace frontier {

// Pseudo branch ids marking a function call and a function return on a path.
static const int kCallId = -1;
static const int kReturnId = -2;

// One concrete+symbolic execution of the program under test.
struct Execution {
  // Handle, unique within the run.
  int id;
  // False when the program was killed (time or memory limit) or crashed
  // before it could report a trace; `path` is then empty.
  bool completed;
  // Every branch taken, in order.  Entries are branch ids (> 0) or the
  // kCallId / kReturnId markers.
  std::vector<int> path;
  // Positions in `path` of the branches whose condition depends on a
  // symbolic input, in increasing order.  constraint_pos[k] is the k-th
  // constraint of the execution; only these can be negated.
  std::vector<uint32_t> constraint_pos;
};

// Static description of the program under test.  Branch and function ids
// are relabelled with a fresh secret permutation in every run: an id means
// nothing outside the run that produced it.
struct Program {
  // Branch ids are in [1, num_branch_ids).  Index 0 is unused.
  int num_branch_ids;
  // All branch ids, sorted.  Each conditional contributes two ids.
  std::vector<int> branches;
  // paired[b] is the other arm of the conditional that owns branch b.
  std::vector<int> paired;
  // function_of[b] is the id of the function containing branch b.
  std::vector<int> function_of;
  // Branch-level control-flow graph: successors[b] lists, sorted, the
  // branches that can be the next branch executed after b.  Calls are
  // followed into the callee; returns are not followed back to the caller.
  std::vector<std::vector<int> > successors;
};

class HostImpl;

// Services of the harness, as seen from the strategy process.
class Host {
 public:
  const Program& program() const;

  // Iteration budget of the run and the part of it already spent.  One
  // iteration is one launch of the program under test.
  int budget() const;
  int iterations_used() const;

  // Coverage accumulated by the whole run so far.
  bool covered(int branch) const;
  int num_covered() const;

  // Deterministic pseudo-random numbers, seeded by the harness per run.
  // Random() is uniform on [0, 2^32); Uniform(n) is uniform on [0, n), n > 0.
  uint32_t Random();
  uint32_t Uniform(uint32_t n);

  // Runs the program on fresh random inputs.  Costs one iteration.
  const Execution* Restart();

  // Asks the harness to negate constraint k of execution e, keeping the
  // constraints before it, and to run the program on the solved inputs.
  //   - If the solver finds the negated prefix infeasible, returns NULL and
  //     no iteration is spent.
  //   - Otherwise the program is launched (one iteration) and the resulting
  //     execution is returned.  The new execution need not follow the
  //     predicted path; compare the paths to find out.
  const Execution* Negate(const Execution* e, size_t k);

  // Gives an execution back.  At most kMaxLiveExecutions may be held.
  void Release(const Execution* e);

  static const int kMaxLiveExecutions = 64;

  explicit Host(HostImpl* impl) : impl_(impl) {}

 private:
  HostImpl* impl_;
  Host(const Host&);
  Host& operator=(const Host&);
};

// When the budget is exhausted, Restart() and Negate() do not return: the
// process ends there.  A strategy that returns from Run() early ends the run
// with the coverage reached so far.
class Strategy {
 public:
  virtual ~Strategy() {}
  virtual void Run(Host& host) = 0;
};

// Implemented by the solution in src/frontier_solution/.
Strategy* MakeStrategy();

}  // namespace frontier

#endif  // STRATEGY_STRATEGY_API_H__
