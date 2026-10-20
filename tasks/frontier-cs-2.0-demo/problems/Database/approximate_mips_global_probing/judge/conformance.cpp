#include "query_core.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv) {
  bool ordered=argc>1;
  std::mt19937 rng(81239);std::normal_distribution<double> normal;
  auto core=make_query_core();int cases=0;
  for(int K=1;K<=9;++K) for(int L: {1,2,5}) for(int mode=0;mode<4;++mode) {
    std::vector<double> p(K*L);for(auto&x:p)x=mode==0?0:mode==1?1:normal(rng);
    core->reset(L,K,p.data());
    std::set<std::pair<int,int>> seen;Probe b;double prev=-1;int count=0;
    while(core->next(b)) {
      if(++count>L*(1<<K))throw std::runtime_error("nontermination");
      if(b.table<0||b.table>=L||b.key<0||b.key>=(1<<K))throw std::runtime_error("range");
      if(!seen.emplace(b.table,b.key).second)throw std::runtime_error("duplicate");
      double want=0;
      for(int j=0;j<K;++j){double z=p[b.table*K+j];if(((b.key>>(K-1-j))&1)!=(z>0))want+=z*z;}
      if(!std::isfinite(b.score)||std::abs(want-b.score)>1e-9*(1+want))throw std::runtime_error("QD");
      if(ordered&&b.score<prev-1e-9)throw std::runtime_error("non-global order");
      prev=b.score;
    }
    if(count!=L*(1<<K))throw std::runtime_error("incomplete");
    // Stop must never activate without k results or at a negative inner product.
    SearchState s={0,10,0,100,0,0,.8,1,L,"unused"};
    if(core->stop(s))throw std::runtime_error("empty-stop");
    s.result_count=10;s.kth_score=-.2;
    if(core->stop(s))throw std::runtime_error("negative-stop");
    ++cases;
  }
  std::cout<<"conformance cases="<<cases<<" ordered="<<ordered<<" PASS\n";
}
