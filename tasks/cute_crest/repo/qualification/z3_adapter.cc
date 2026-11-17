#include <queue>
#include <vector>
#include <stdexcept>
#include <z3++.h>
#include "base/yices_solver.h"
namespace crest {
using std::queue;
using std::make_pair;
typedef vector<const SymbolicPred*>::const_iterator PredIt;
bool YicesSolver::IncrementalSolve(const vector<value_t>& old_soln,
				   const map<var_t,type_t>& vars,
				   const vector<const SymbolicPred*>& constraints,
				   map<var_t,value_t>* soln) {
  set<var_t> tmp;
  typedef set<var_t>::const_iterator VarIt;

  // Build a graph on the variables, indicating a dependence when two
  // variables co-occur in a symbolic predicate.
  vector< set<var_t> > depends(vars.size());
  for (PredIt i = constraints.begin(); i != constraints.end(); ++i) {
    tmp.clear();
    (*i)->AppendVars(&tmp);
    for (VarIt j = tmp.begin(); j != tmp.end(); ++j) {
      depends[*j].insert(tmp.begin(), tmp.end());
    }
  }

  // Initialize the set of dependent variables to those in the constraints.
  // (Assumption: Last element of constraints is the only new constraint.)
  // Also, initialize the queue for the BFS.
  map<var_t,type_t> dependent_vars;
  queue<var_t> Q;
  tmp.clear();
  constraints.back()->AppendVars(&tmp);
  for (VarIt j = tmp.begin(); j != tmp.end(); ++j) {
    dependent_vars.insert(*vars.find(*j));
    Q.push(*j);
  }

  // Run the BFS.
  while (!Q.empty()) {
    var_t i = Q.front();
    Q.pop();
    for (VarIt j = depends[i].begin(); j != depends[i].end(); ++j) {
      if (dependent_vars.find(*j) == dependent_vars.end()) {
	Q.push(*j);
	dependent_vars.insert(*vars.find(*j));
      }
    }
  }

  // Generate the list of dependent constraints.
  vector<const SymbolicPred*> dependent_constraints;
  for (PredIt i = constraints.begin(); i != constraints.end(); ++i) {
    if ((*i)->DependsOn(dependent_vars))
      dependent_constraints.push_back(*i);
  }

  soln->clear();
  if (Solve(dependent_vars, dependent_constraints, soln)) {
    // Merge in the constrained variables.
    for (PredIt i = constraints.begin(); i != constraints.end(); ++i) {
      (*i)->AppendVars(&tmp);
    }
    for (set<var_t>::const_iterator i = tmp.begin(); i != tmp.end(); ++i) {
      if (soln->find(*i) == soln->end()) {
	soln->insert(make_pair(*i, old_soln[*i]));
      }
    }
    return true;
  }

  return false;
}


// Compatibility adapter: preserve CREST's linear integer formula and inclusive
// C type bounds. This does not add the crest-z3 fork's nonlinear expression tree.
bool YicesSolver::Solve(const map<var_t,type_t>& vars,
                       const vector<const SymbolicPred*>& constraints,
                       map<var_t,value_t>* soln) {
  z3::context ctx;
  z3::solver solver(ctx);
  // Bound every query: a deterministic resource limit first, a wall-clock
  // limit behind it.  A query that hits either is reported as unknown.
  { z3::params limits(ctx);
    limits.set("rlimit", 20000000u);
    limits.set("timeout", 10000u);
    solver.set(limits); }
  map<var_t,z3::expr> symbols;
  for (map<var_t,type_t>::const_iterator i=vars.begin();i!=vars.end();++i) {
    std::string name="x"+std::to_string(i->first);
    z3::expr x=ctx.int_const(name.c_str());
    symbols.insert(std::make_pair(i->first,x));
    solver.add(x>=ctx.int_val(kMinValueStr[i->second]));
    solver.add(x<=ctx.int_val(kMaxValueStr[i->second]));
  }
  for (size_t i=0;i<constraints.size();++i) {
    const SymbolicExpr& e=constraints[i]->expr();
    z3::expr term=ctx.int_val(std::to_string(e.const_term()).c_str());
    for (SymbolicExpr::TermIt j=e.terms().begin();j!=e.terms().end();++j)
      term=term+symbols.at(j->first)*ctx.int_val(std::to_string(j->second).c_str());
    switch(constraints[i]->op()) {
      case ops::EQ: solver.add(term==0);break;
      case ops::NEQ: solver.add(term!=0);break;
      case ops::GT: solver.add(term>0);break;
      case ops::LE: solver.add(term<=0);break;
      case ops::LT: solver.add(term<0);break;
      case ops::GE: solver.add(term>=0);break;
      default: throw std::runtime_error("unknown CREST comparison");
    }
  }
  z3::check_result r=solver.check();
  if(r==z3::unsat)return false;
  if(r!=z3::sat)throw std::runtime_error("Z3 unknown: "+solver.reason_unknown());
  z3::model model=solver.get_model();
  soln->clear();
  for (map<var_t,type_t>::const_iterator i=vars.begin();i!=vars.end();++i) {
    // Fail explicitly when the mathematical model does not fit CREST value_t.
    // The task's admissible signed-int/byte fixtures never need larger values.
    int64_t value;
    if(!model.eval(symbols.at(i->first),true).is_numeral_i64(value))
      throw std::runtime_error("model outside CREST value_t");
    soln->insert(std::make_pair(i->first,static_cast<value_t>(value)));
  }
  return true;
}

}
