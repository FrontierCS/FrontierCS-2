#include <memory>
#include <stdint.h>

using s8 = int8_t;
using u8 = uint8_t;
using u16 = uint16_t;
using s32 = int32_t;
using u64 = uint64_t;

namespace nRUNLTS {
class RUNLTS;
}

class CBP2025_RUNLTS {
    std::shared_ptr<nRUNLTS::RUNLTS> p_impl;

public:
    void setup();
    bool predict(u64 seq_no, u8 piece, u64 PC);
    void update(u64 seq_no, u8 piece, u64 PC, bool resolveDir, bool predDir, u64 nextPC);
    void history_update(u64 seq_no, u8 piece, u64 PC, int brtype, bool pred_dir, bool resolve_dir, u64 nextPC);
    void TrackOtherInst(u64 PC, int brtype, bool pred_dir, bool resolve_dir, u64 nextPC);
    void terminate();
    void decode_notify(u64 seq_no, u8 piece, u64 dst_reg);
    void execute_notify(u64 seq_no, u8 piece, u64 dst_reg, u64 value);
} inline cbp2025_RUNLTS;
