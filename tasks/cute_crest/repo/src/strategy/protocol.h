// Wire protocol between the search harness and the sandboxed strategy
// process.  Fixed harness code; not part of the solution surface.
//
// All integers are 32-bit little-endian.  The strategy process reads from
// file descriptor 0 and writes to file descriptor 1.

#ifndef STRATEGY_PROTOCOL_H__
#define STRATEGY_PROTOCOL_H__

namespace frontier {
namespace wire {

static const int kMagic = 0x43535431;  // "CST1"

// Requests (strategy -> harness).
static const int kReqRestart = 1;
static const int kReqNegate = 2;   // + execution id, constraint index
static const int kReqRelease = 3;  // + execution id (no reply)
static const int kReqDone = 4;     // strategy returned from Run() (no reply)

// Replies (harness -> strategy).
static const int kReplyInfeasible = 0;
static const int kReplyExecution = 1;  // + execution record
static const int kReplyBudgetExhausted = 2;

}  // namespace wire
}  // namespace frontier

#endif  // STRATEGY_PROTOCOL_H__
