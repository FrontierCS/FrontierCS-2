// Your search strategy goes here.
//
// This starter only restarts the program on fresh random inputs until the
// budget is spent.  It compiles and runs, and it is a poor search: replace
// the body of Run().  You may add files to this directory and include them
// from here; see the task README for the rules on what this code may use.

#include "strategy/strategy_api.h"

namespace frontier {

namespace {

class RandomRestartSearch : public Strategy {
 public:
  virtual void Run(Host& host) {
    while (true) {
      const Execution* ex = host.Restart();
      host.Release(ex);
    }
  }
};

}  // namespace

Strategy* MakeStrategy() { return new RandomRestartSearch(); }

}  // namespace frontier
