#pragma once
/*

Copyright (c) 2019, North Carolina State University
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
simulation thereof owned by North Carolina State University must not be used to endorse or promote products derived from this software without prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

// Author: Eric Rotenberg (ericro@ncsu.edu)


////////////////////////////////////////////////////////////////////////////////
//
// Interface for the 1st Championship on Value Prediction (CVP-1).
//
////////////////////////////////////////////////////////////////////////////////

#include "sim_common_structs.h"

struct PredictionRequest
{
    // Instruction sequence number
    uint64_t seq_no = -1;
    // Instruction Program Counter
    uint64_t pc = 0xdeadbeef;
    // Instruction piece number (e.g., for SIMD instructions)
    uint8_t piece = 0;
    // Is candidate for VP in the current track
    bool is_candidate = false;
    // Data Cache hit/miss information
    HitMissInfo cache_hit = HitMissInfo::Invalid;
};

struct PredictionResult
{
    uint64_t predicted_value = 0x0;
    bool speculate = false;
};

//
// getPrediction(const PredictionRequest& req)
//
// Return value: PredictionResult
// .speculate:
//         "true": if microarch. simulator should speculate based on the prediction for this instruction.
//         "false": if it should not speculate based on the prediction for this instruction.
// .predicted_value: The prediction
// This allows contestants to decide between the potential speedup of speculation vs.
// the potential penalty of a squash from ROB-head due to misspeculation.
//
// Input arguments: PredictionRequest
// 1. sequence number: the dynamic micro-instruction number
// 2. program counter (pc) of instruction
// 3. piece: Some instructions operate on values that are wider than 64 bits.
//           These are split into multiple 64-bit pieces and getPrediction() is called
//           for pieces 0, 1, ..., n, consecutively, for the same instruction.
//           Note that sequence number is incremented whether for an instruction or piece
//           of an instruction (hence sequence number is dynamic micro-instruction number).
// 4. is_candidate: Is candidate instruction for the active track (e.g. is_load)
// 5. cache_hit: is hit in the data cache. valid only for LoadsOnlyHitMiss track
extern 
PredictionResult getPrediction(const PredictionRequest& req);

//
// speculativeUpdate()
//
// This function is called immediately after getPrediction() for the just-predicted instruction.
//
// A key argument to this function is "prediction_result".
// * If getPrediction() instructed the simulator to speculate, then "prediction_result" will reveal whether or not
//   the predicted value is correct, immediately after getPrediction().
//   In other words, if the contestant took the risk of speculating, he/she has the privilege of knowing the outcome of
//   speculation immediately.  This privilege is justified whether or not the prediction was correct.  If the prediction
//   was incorrect, all instructions prior to the mispredicted one will be retired before the next call to getPrediction(),
//   including draining the window of all pending calls to updatePredictor().  Effectively, architectural state becomes
//   visible after a value misprediction and before the next value prediction.
//   On the other hand, if the prediction was correct, exposing it as correct supports a speculative-update policy
//   without the need for value predictor fix-ups, for all contestants.
// * On the other hand, if getPrediction() instructed the simulator to NOT speculate, then the contestant forfeits the
//   privilege described above.  "prediction_result" will signify that the outcome of the prediction will not be
//   revealed immediately.  The contestant must wait until the non-speculative updatePredictor() function to see
//   whether or not the prediction was correct.
//
extern
void speculativeUpdate(uint64_t seq_no,         // dynamic micro-instruction # (starts at 0 and increments indefinitely)
                       bool eligible,           // true: instruction is eligible for value prediction. false: not eligible.
               uint8_t prediction_result,   // 0: incorrect, 1: correct, 2: unknown (not revealed)
               // Note: can assemble local and global branch history using pc, next_pc, and insn.
               uint64_t pc,         
               uint64_t next_pc,
               uint8_t insn_class_uint8,
               uint8_t piece,
               // Note: up to 3 logical source register specifiers, up to 1 logical destination register specifier.
               // 0xdeadbeef means that logical register does not exist.
               // May use this information to reconstruct architectural register file state (using log. reg. and value at updatePredictor()).
               uint64_t src1,
               uint64_t src2,
               uint64_t src3,
               uint64_t dst);

//
// updatePredictor()
//
// This is called for a micro-instruction when it is retired.
//
// Generally there is a delay between the getPrediction()/speculativeUpdate() calls and the corresponding updatePredictor() call,
// which is the delay between fetch and retire.  This delay manifests to contestants as multiple unrelated
// getPrediction(y,z,...)/speculativeUpdate(y,z,...) calls between an instruction x's getPrediction(x)/speculativeUpdate(x) calls and
// its updatePredictor(x) call.  The delay goes away when there is a value misprediction because of the complete-squash recovery model.
// After a value misprediction, the window is drained of all pending updatePredictor() calls before the next call to getPrediction()/speculativeUpdate().
//
extern
void updatePredictor(uint64_t seq_no,       // dynamic micro-instruction #
             uint64_t actual_addr,  // load or store address (0xdeadbeef if not a load or store instruction)
             uint64_t actual_value, // value of destination register (0xdeadbeef if instr. is not eligible for value prediction)
             uint64_t actual_latency);  // actual execution latency of instruction

//
// beginPredictor()
// 
// This function is called by the simulator before the start of simulation.
// It can be used for arbitrary initialization steps for the contestant's code.
// Note that the contestant may also parse additional contestant-specific command-line arguments that come
// after the final required argument, i.e., after the .gz trace file.
//
extern
void beginPredictor(int argc_other, char **argv_other);

//
// endPredictor()
//
// This function is called by the simulator at the end of simulation.
// It can be used by the contestant to print out other contestant-specific measurements.
//
extern
void endPredictor();
