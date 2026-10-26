#pragma once
#include <cstdint>
#include <string>

namespace spectrum {

// Checkpoint metrics measured by the harness around the editable checkpoint
// backend during real Spectrum execution (CHECKPOINTED transactions only):
// create cost (save_checkpoint() plus storing the checkpoint), restore cost
// (load_checkpoint() alone), and bytes retained per live checkpoint, i.e. the
// net heap bytes the executing thread allocated since the transaction's
// execution state was created, divided by its live checkpoints. Heap bytes are
// jemalloc's per-thread allocated/deallocated counters.
void frontier_checkpoint_record_create(uint64_t nanoseconds);
void frontier_checkpoint_record_restore(uint64_t nanoseconds);
void frontier_checkpoint_record_retained(uint64_t bytes_per_checkpoint);
int64_t frontier_checkpoint_thread_net_bytes();
std::string frontier_checkpoint_print();

} // namespace spectrum
