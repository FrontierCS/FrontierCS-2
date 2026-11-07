#include <inttypes.h>


#include "value_predictor_interface.h"
#include "my_value_predictor.h"
#include <iostream>
int seq_commit;


#define NOTLLCMISS (actual_latency < 150)
#define NOTL2MISS (actual_latency < 60)
#define NOTL1MISS (actual_latency < 12)
#define FASTINST (actual_latency ==1)
#define MFASTINST (actual_latency <3)

#define aluInstClass  0
#define loadInstClass  1
#define storeInstClass  2
#define condBranchInstClass  3
#define uncondDirectBranchInstClass  4
#define uncondIndirectBranchInstClass  5
#define fpInstClass  6
#define slowAluInstClass  7
#define undefInstClass  8 
#define callDirectInstClass  9
#define callIndirectInstClass  10
#define ReturnInstClass  11

void getPredVtage (ForUpdate * U, uint64_t & predicted_value)
{
}

void
getPredStride (ForUpdate * U, uint64_t & predicted_value, uint64_t seq_no)
{
}


PredictionResult getPrediction(const PredictionRequest& req)
{
    PredictionResult result;
    return result;
}

// Update of the Stride predictor
// function determining whether to  update or not confidence on a correct prediction
bool
strideupdateconf (ForUpdate * U, uint64_t actual_value, int actual_latency,
          int stride)
{
    return false;
}

//Allocate or not if instruction absent from the predictor
bool
StrideAllocateOrNot (ForUpdate * U, uint64_t actual_value, int actual_latency)
{
  return false;
}

void
UpdateStridePred (ForUpdate * U, uint64_t actual_value, int actual_latency)
{}




/////////Update of  VTAGE
// function determining whether to  update or not confidence on a correct prediction

bool
vtageupdateconf (ForUpdate * U, uint64_t actual_value, int actual_latency)
{
    return false;
}

// Update of the U counter or not 
bool
VtageUpdateU (ForUpdate * U, uint64_t actual_value, int actual_latency)
{
    return false;
}

bool
VtageAllocateOrNot (ForUpdate * U, uint64_t actual_value, int actual_latency,
            bool MedConf)
{
    return false;
}



void
UpdateVtagePred (ForUpdate * U, uint64_t actual_value, int actual_latency)
{}

void
updatePredictor (uint64_t
         seq_no,
         uint64_t
         actual_addr, uint64_t actual_value, uint64_t actual_latency)
{
}




void
speculativeUpdate (uint64_t seq_no, // dynamic micro-instruction # (starts at 0 and increments indefinitely)
           bool eligible,   // true: instruction is eligible for value prediction. false: not eligible.
           uint8_t prediction_result,   // 0: incorrect, 1: correct, 2: unknown (not revealed)
           // Note: can assemble local and global branch history using pc, next_pc, and insn.
           uint64_t
           pc, uint64_t next_pc, uint8_t insn_class_uint8, uint8_t piece,
           // Note: up to 3 logical source register specifiers, up to 1 logical destination register specifier.
           // 0xdeadbeef means that logical register does not exist.
           // May use this information to reconstruct architectural register file state (using log. reg. and value at updatePredictor()).
           uint64_t src1, uint64_t src2, uint64_t src3, uint64_t dst)
{}

void
beginPredictor (int argc_other, char **argv_other)
{
}

void
endPredictor ()
{
}
