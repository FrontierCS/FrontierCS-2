#define HYPER 1
#include "hygra.h"
#include "index_map.h"
#include "bucket.h"
#include "edgeMapReduce.h"

template <class vertex>
array_imap<uintE> KCore(hypergraph<vertex>& GA, size_t num_buckets=128) {
  // Implement the required computation here.
  return array_imap<uintE>();
}

template <class vertex>
void Compute(hypergraph<vertex>& GA, commandLine P) {
  size_t num_buckets = P.getOptionLongValue("-nb", 128);
  if (num_buckets != (1 << pbbs::log2_up(num_buckets))) {
    cout << "Number of buckets must be a power of two." << endl;
    exit(-1);
  }
  //cout << "### Application: k-core" << endl;
  //cout << "### Graph: " << P.getArgument(0) << endl;
  //cout << "### Workers: " << getWorkers() << endl;
  //cout << "### Buckets: " << num_buckets << endl;

  auto cores = KCore(GA, num_buckets);
  //cout << "### Max core: " << sequence::reduce(cores.s,GA.nv,maxF<uintE>()) << endl;
}
