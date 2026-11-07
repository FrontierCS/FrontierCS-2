// CBP Trace Reader
// Author: Arthur Perais (arthur.perais@gmail.com) for CVP
//         Saransh Jain/Rami Sheikh updated for CBP

/*
   This is free and unencumbered software released into the public domain.

   Anyone is free to copy, modify, publish, use, compile, sell, or
   distribute this software, either in source code form or as a compiled
   binary, for any purpose, commercial or non-commercial, and by any
   means.

   In jurisdictions that recognize copyright laws, the author or authors
   of this software dedicate any and all copyright interest in the
   software to the public domain. We make this dedication for the benefit
   of the public at large and to the detriment of our heirs and
   successors. We intend this dedication to be an overt act of
   relinquishment in perpetuity of all present and future rights to this
   software under copyright law.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
   OTHER DEALINGS IN THE SOFTWARE.

   For more information, please refer to <http://unlicense.org>

   This project relies on https://www.cs.unc.edu/Research/compgeom/gzstream/ (LGPL) to stream from compressed traces.
   For ease of use, gzstream.h and gzstream.C are provided with minor modifications (include paths) with the reader.
   */

// Compilation : Don't forget to add gzstream.C in the source list and to link with zlib (-lz).
//
// Usage : TraceReader reader("./my_trace.tar.gz")
//         while(reader.readInstr())
//           reader.mInstr.printInstr();
//            ...
// Note that this is an exemple file (that was used for CVP).
// Given the trace format below, you can write your own trace reader that
// populates the instruction format in the way you like.
//
// Trace Format :
// Inst PC                  - 8 bytes
// Inst Type                - 1 byte
// If load/storeInst
//   Effective Address      - 8 bytes
//   Access Size (total)    - 1 byte
//   Involves Base Update   - 1 byte
//   If Store:
//      Involves Reg Offset - 1 byte
// If branch
//   Taken                  - 1 byte
//   If Taken:
//      Target              - 8 bytes
// Num Input Regs           - 1 byte
// Input Reg Names          - 1 byte each
// Num Output Regs          - 1 byte
// Output Reg Names         - 1 byte each
// Output Reg Values
//   If INT                 - 8 bytes each
//   If SIMD                - 16 bytes each
//
// Int registers are encoded 0-30(GPRs), 31(Stack Pointer Register), 64(Flag Register), 65(Zero Register)
// SIMD registers are encoded 32-63

#include <fstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include "sim_common_structs.h"
#include "./gzstream.h"

// This structure is used by CBP's simulator.
// Adapt for your own needs.
struct db_operand_t
{
    bool valid;
    bool is_int;
    uint64_t log_reg;
    uint64_t value;

    void print() const
    {
        std::cout << *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const db_operand_t& entry)
    {
        os << " (int: " << entry.is_int << ", idx: " << entry.log_reg << " val: " << std::hex << entry.value << std::dec << ") ";
        return os;
    }
};

// This structure is used by CBP's simulator.
// Adapt for your own needs.
struct db_t
{
    InstClass insn_class;
    uint64_t pc;
    bool is_taken;
    uint64_t next_pc;

    db_operand_t A;
    db_operand_t B;
    db_operand_t C;
    db_operand_t D;

    bool is_load;
    bool is_store;
    uint64_t addr;
    uint64_t size;

    bool is_last_piece;

    friend std::ostream& operator<<(std::ostream& os, const db_t& entry)
    {
        os << "[PC: 0x" << std::hex << entry.pc <<std::dec << " type: "  << cInfo[static_cast<uint8_t>(entry.insn_class)];
        if(entry.insn_class == InstClass::loadInstClass || entry.insn_class == InstClass::storeInstClass)
        {
            assert(entry.is_load || entry.is_store);
            os << " ea: 0x" << std::hex << entry.addr << std::dec << " size: " << entry.size;
        }
        //if(insn_class == InstClass::condBranchInstClass || insn_class == InstClass::uncondDirectBranchInstClass || insn_class == InstClass::uncondIndirectBranchInstClass)
        if(is_br(entry.insn_class))
            os << " ( tkn:" << (entry.next_pc != entry.pc + 4) << " tar: 0x" << std::hex << entry.next_pc << ") " << std::dec;

        if(entry.A.valid)
        {
            os << " 1st input: " << entry.A;
        }

        if(entry.B.valid)
        {
            os << "2nd input: " << entry.B;
        }

        if(entry.C.valid)
        {
            os << "3rd input: " << entry.C;
        }

        if(entry.D.valid)
        {
            os << " output: " << entry.D;
        }

        os << " ]" ;
        return os;
    }

    void printInst(uint64_t fetch_cycle) const
    {
        std::cout << fetch_cycle<<"::uOP:: "<<*this<<std::endl;
    }
};

// INT registers are registers 0 to 31. SIMD/FP registers are registers 32 to 63. Flag register is register 64
enum Offset
{
    vecOffset = 32,
    ccOffset = 64,
    ZeroOffset = 65
};

inline bool reg_is_int(uint8_t reg_offset)
{
    return ( (reg_offset < Offset::vecOffset) || (reg_offset == Offset::ccOffset) || (reg_offset == Offset::ZeroOffset) );
}


// Trace reader class.
// Format assumes that instructions have at most three inputs and at most one input.
// If the trace contains an instruction that has more than three inputs, they are ignored.
// If the trace contains an instruction that has more than one outputs, one instruction object will be created for each output,
// For instance, for a multiply, two objects with same PC will be created through two subsequent calls to get_inst(). Each will have the same
// inputs, but one will have the low part of the product as output, and one will have the high part of the product as output.
// Other instructions subject to this are (list not exhaustive): load pair and vector inststructions.
struct TraceReader
{
    struct Instr
    {
        uint64_t mPc;
        InstClass mType; // Type is InstClass
        bool mTaken;
        uint64_t mNextPc;
        uint64_t mEffAddr;
        uint8_t mMemSize; // In bytes
        uint8_t mBaseUpd;
        uint8_t mHasRegOffset;
        uint8_t mNumInRegs;
        std::vector<uint8_t> mInRegs;
        uint8_t mNumOutRegs;
        std::vector<uint8_t> mOutRegs;
        std::optional<uint8_t> mBaseUpdReg;
        std::vector<uint64_t> mOutRegsValues;

        Instr()
        {
            reset();
        }

        void reset()
        {
            mPc = mNextPc = 0xdeadbeef;
            mEffAddr = 0xdeadbeef;
            mMemSize = 0;
            mBaseUpdReg = 0;
            mHasRegOffset = 0;
            mType = InstClass::undefInstClass;
            mTaken = false;
            mNumInRegs = mNumOutRegs = 0;
            mInRegs.clear();
            mOutRegs.clear();
            mBaseUpdReg.reset();
            mOutRegsValues.clear();
        }

        bool capture_base_update_log_reg()
        {
            assert(mType != InstClass::undefInstClass);
            assert(!mBaseUpdReg.has_value());
            
            if(!is_mem(mType))
            {
                return false;
            }

            if(is_store(mType))
            {
                if(mNumOutRegs > 1)
                {
                    std::cout<<"Store with >1 base upd! src_regs: [";
                    for(auto i:mInRegs)
                    {
                        std::cout<<", "<<(uint64_t)i;
                    }
                    std::cout<<"], dst_regs: [";
                    for(auto i:mOutRegs)
                    {
                        std::cout<<", "<<(uint64_t)i;
                    }
                    std::cout<<"]"<<std::endl;
                }

                assert(mNumOutRegs <= 1 && "Store unexpected to have more than 1 output register");
                assert((mNumOutRegs == 1) == (mBaseUpd == 1) && "Mismatch between trace writer/reader for base update logic");
                if(mNumOutRegs == 1)
                {
                    mBaseUpdReg.emplace(mOutRegs.at(0));
                    return true;
                }
                return false;
            }

            // Load
            assert(is_load(mType));
            if(mOutRegs.size() == 0)
            {
                std::cout<<"Expected load to have a destination. PC:0x"<<std::hex<<mPc<<" NextPC:0x"<<mNextPc<<std::dec<<std::endl;
            }
            assert(mOutRegs.size() > 0);
            if(mOutRegs.size() <= 1)
            {
                return false;
            }
            auto src_reg_ids = mInRegs;
            std::sort(src_reg_ids.begin(), src_reg_ids.end());
            auto src_it = std::lower_bound(src_reg_ids.begin(), src_reg_ids.end(),  Offset::vecOffset);
            if(src_it != src_reg_ids.end())
            {
                src_reg_ids.erase(src_it, src_reg_ids.end());
            }

            auto dst_reg_ids = mOutRegs;
            std::sort(dst_reg_ids.begin(), dst_reg_ids.end());
            auto dst_it = std::lower_bound(dst_reg_ids.begin(), dst_reg_ids.end(),  Offset::vecOffset);
            if(dst_it != dst_reg_ids.end())
            {
                dst_reg_ids.erase(dst_it, dst_reg_ids.end());
            }

            std::vector<uint8_t> overlap_vec;
            std::set_intersection(src_reg_ids.begin(), src_reg_ids.end(), dst_reg_ids.begin(), dst_reg_ids.end(), std::back_inserter(overlap_vec));

            if(overlap_vec.size() > 1)
            {
                std::cout<<"Load with >1 base upd! src_regs: [";
                for(auto i:src_reg_ids)
                {
                    std::cout<<", "<<(uint64_t)i;
                }
                std::cout<<"], dst_regs: [";
                for(auto i:dst_reg_ids)
                {
                    std::cout<<", "<<(uint64_t)i;
                }
                std::cout<<"], overlap_vec: [";
                for(auto i:overlap_vec)
                {
                    std::cout<<", "<<(uint64_t)i;
                }
                std::cout<<"]"<<std::endl;
            }
            assert(overlap_vec.size() <= 1);
            const bool base_update = overlap_vec.size() == 1;
            if(mBaseUpd == 1)
            {
                assert(base_update);
            }
            const bool true_base_update = (mBaseUpd == 1) && base_update;
            if(true_base_update)
            {
                mBaseUpdReg.emplace(overlap_vec.at(0));
            }
            return true_base_update;
        }

        friend std::ostream& operator<<(std::ostream& os, const Instr& instr)
        {
            os << "mOP:: [PC: 0x" << std::hex << instr.mPc << std::dec  << " type: "  << cInfo[static_cast<uint8_t>(instr.mType)];
            if(instr.mType == InstClass::loadInstClass || instr.mType == InstClass::storeInstClass)
                os << " ea: 0x" << std::hex << instr.mEffAddr << std::dec << " size: " << (uint64_t) instr.mMemSize << " baseupdreg: " << (uint64_t)instr.mBaseUpdReg.value_or(UINT8_MAX);

            //if(mType == InstClass::condBranchInstClass || mType == InstClass::uncondDirectBranchInstClass || mType == InstClass::uncondIndirectBranchInstClass)
            if(is_br(instr.mType))
                os << " ( tkn:" << instr.mTaken << " tar: 0x" << std::hex << instr.mNextPc << ") "<<std::dec;

            os << std::dec << " InRegs : { ";
            for(unsigned elt : instr.mInRegs)
            {
                os << elt << " ";
            }
            os << " } OutRegs : { ";
            for(unsigned i = 0, j = 0; i < instr.mOutRegs.size(); i++)
            {
                if(!reg_is_int(instr.mOutRegs[i])) //mOutRegs[i] >= Offset::vecOffset && mOutRegs[i] != Offset::ccOffset
                {
                    assert(j+1 < instr.mOutRegsValues.size());
                    os << std::dec << (unsigned) instr.mOutRegs[i] << std::hex << " (hi:" << instr.mOutRegsValues[j+1] << " lo:" << instr.mOutRegsValues[j] << ") "<<std::dec;
                    j += 2;
                }
                else
                {
                    assert(j < instr.mOutRegsValues.size());
                    os << std::dec << (unsigned) instr.mOutRegs[i] << std::hex << " (" << instr.mOutRegsValues[j++] << ") "<<std::dec;
                }
            }
            os << "} ]";
            return os;
        }

        void printInstr()
        {
            std::cout<<*this<<std::endl;
        }
    };

    gz::igzstream * dpressed_input;

    // Buffer to hold trace instruction information
    Instr mInstr;

    // Expected total pieces of an instr
    uint8_t mTotalPieces;
    // Expected MemPieces
    uint8_t mMemPieces;
    // Pieces processed so far 
    uint8_t mProcessedPieces;

    // There are bookkeeping variables for trace instructions that necessitate the creation of several objects.
    // Specifically, load pair yields two objects with different register outputs, but SIMD instructions also yield
    // multiple objects but they have the same register output, although one has the low 64-bit and one has the high 64-bit.
    // Thus, for load pair, first piece will have mCrackRegIdx 0 (first output of trace instruction) and second will have mCrackRegIdx 1
    // (second output of trace instruction). However, in both cases, since outputs are 64-bit, mCrackValIdx will remain 0.
    // Conversely, both pieces of a SIMD instruction will have mCrackRegIdx 0 (since SIMD instruction has a single 128-bit output), but first piece
    // will have mCrackValIdx 0 (low 64-bit) and second piece will have mCrackValIdx 1 (high 64-bit).
    uint8_t mCrackRegIdx;
    uint8_t mCrackValIdx;

    // Inverse of memory size multiplier for each piece. E.g., a 128-bit access will have size 16, and each piece will have size 8 (16 * 1/2).
    uint8_t mSizeFactor;

    // Number of instructions processed so far.
    uint64_t nInstr;

    // This simply tracks how many lanes one SIMD register have been processed.
    // In this case, since SIMD is 128 bits and pieces output 64 bits, if it is pair and we are creating an instruction object from a trace instruction, this means that
    // the output of the instruction object will contain the low order bits of the SIMD register.
    // If it is odd, it means that it will contain the high order bits of the SIMD register.
    uint8_t start_fp_reg;

    // Note that there is no check for trace existence, so modify to suit your needs.
    TraceReader(const char * trace_name)
    {
        dpressed_input = new gz::igzstream();
        dpressed_input->open(trace_name, std::ios_base::in | std::ios_base::binary);

        mTotalPieces = 0;
        mMemPieces = 0;
        mCrackRegIdx = 0;
        mCrackValIdx = 0;
        mProcessedPieces = 0;
        mSizeFactor = 0;
        nInstr = 0;
        start_fp_reg = 0;
    }

    ~TraceReader()
    {
        if(dpressed_input)
            delete dpressed_input;

        std::cout  << " Read " << nInstr << " instrs " << std::endl;
    }

    // This is the main API function
    // There is no specific reason to call the other functions from without this file.
    // Idiom is : while(instr = get_inst())
    //              ... process instr
    db_t  *get_inst()
    {
        // If we are creating several pieces from a single trace instructions and some are left to create,
        // mProcessedPieces != mTotalPieces
        if(mProcessedPieces != mTotalPieces)
        {
            //std::cout<<"Continuing with the same MacroOP"<<std::endl;
            return populateNewInstr();
        }
        // If there is a single piece to create
        else if(readInstr())
        {
            //std::cout<<"Read New MacroOp"<<std::endl;
            return populateNewInstr();
        }
        else
        {
            // If the trace is done
            //std::cout<<"End of sim"<<std::endl;
            return nullptr;
        }

    }

    // Creates a new object and populate it with trace information.
    // Subsequent calls to populateNewInstr() will take care of creating multiple pieces for a trace instruction
    // that has several outputs or 128-bit output.
    // Number of calls is decided by mProcessedPieces from get_inst().
    db_t *populateNewInstr()
    {
        db_t * inst = new db_t();

        //std::cout<<"Processing piece:"<<(uint64_t)(1+mProcessedPieces)<<" from:"<<(uint64_t)mTotalPieces<<std::endl;
        assert(mProcessedPieces < mTotalPieces);
        assert(mMemPieces <= mTotalPieces);

        const bool is_macro_op_mem = is_mem(mInstr.mType);
        if(is_macro_op_mem)
        {
            assert(mMemPieces >=1);
        }
        else
        {
            assert(mMemPieces ==0);
        }

        //                                
        const bool create_base_update_op = is_macro_op_mem && (mProcessedPieces>=1) && (mMemPieces == mProcessedPieces) && (mMemPieces == (mTotalPieces -1));
        // verify only 1 piece remaining
        if(create_base_update_op)
        {
            assert((1 + mMemPieces) == mTotalPieces && "Only 1 create_base_update_op piece expected");
            assert((1 + mProcessedPieces) == mTotalPieces && "Base-update piece expected to be the last one");
        }

        inst->insn_class = create_base_update_op ? InstClass::aluInstClass : mInstr.mType;
        inst->pc = mInstr.mPc;
        inst->is_taken = mInstr.mTaken;
        if(is_uncond_br(inst->insn_class))
        {
            assert(inst->is_taken);
        }
        else if(!is_cond_br(inst->insn_class))
        {
            assert(!inst->is_taken);
        }

        inst->next_pc = mInstr.mNextPc;
      
        const bool base_update_reg_present = mInstr.mBaseUpdReg.has_value();;
        const uint8_t base_upd_reg = mInstr.mBaseUpdReg.value_or(UINT8_MAX);

        // Process all the inputs for this uop
        uint8_t inp_reg_processed_this_instr = 0;
        if(create_base_update_op)
        {
            assert(base_update_reg_present);
            inp_reg_processed_this_instr++;
            assert(base_upd_reg <= Offset::vecOffset);
            inst->A.valid = true;
            inst->A.is_int = reg_is_int(base_upd_reg);
            assert(inst->A.is_int);
            inst->A.log_reg = base_upd_reg;
            inst->A.value = 0xdeadbeef;

            inst->B.valid = false;
            inst->C.valid = false;
        }
        else if(is_store(mInstr.mType))
        {
            const uint8_t max_str_val_regs_per_instr = 1;
            // str addr reg
            {
                inp_reg_processed_this_instr++;
                inst->A.valid = true;
                inst->A.is_int = reg_is_int(mInstr.mInRegs.at(0));
                inst->A.log_reg = mInstr.mInRegs[0];
                inst->A.value = 0xdeadbeef;
            }

            const uint8_t str_value_reg_offset = 1 + mInstr.mHasRegOffset + mProcessedPieces*max_str_val_regs_per_instr;
            if(mInstr.mHasRegOffset)
            {
                assert(mInstr.mNumInRegs >= 2);
                // str offset
                inp_reg_processed_this_instr++;
                inst->B.valid = true;
                inst->B.is_int = reg_is_int(mInstr.mInRegs.at(1));
                inst->B.log_reg = mInstr.mInRegs[1];
                inst->B.value = 0xdeadbeef;

                // str value
                if( str_value_reg_offset < mInstr.mNumInRegs )
                {
                    inp_reg_processed_this_instr++;
                    inst->C.valid = true;
                    inst->C.is_int = reg_is_int(mInstr.mInRegs.at(str_value_reg_offset));
                    inst->C.log_reg = mInstr.mInRegs[str_value_reg_offset];
                    inst->C.value = 0xdeadbeef;
                }
                else
                {
                    inst->C.valid = false;
                }
            }
            else
            {
                assert(mInstr.mNumInRegs >= 1);
                if( str_value_reg_offset < mInstr.mNumInRegs )
                {
                    // str value
                    inp_reg_processed_this_instr++;
                    inst->B.valid = true;
                    inst->B.is_int = reg_is_int(mInstr.mInRegs.at(str_value_reg_offset));
                    inst->B.log_reg = mInstr.mInRegs[str_value_reg_offset];
                    inst->B.value = 0xdeadbeef;

                    inst->C.valid = false;
                }
                else
                {
                    inst->B.valid = false;
                    inst->C.valid = false;
                }
            }
        }
        else
        {
            if(mInstr.mNumInRegs >= 1)
            {
                inp_reg_processed_this_instr++;
                inst->A.valid = true;
                inst->A.is_int = reg_is_int(mInstr.mInRegs.at(0));
                inst->A.log_reg = mInstr.mInRegs[0];
                inst->A.value = 0xdeadbeef;
            }
            else
                inst->A.valid = false;

            if(mInstr.mNumInRegs >= 2)
            {
                inp_reg_processed_this_instr++;
                inst->B.valid = true;
                inst->B.is_int = reg_is_int(mInstr.mInRegs.at(1));
                inst->B.log_reg = mInstr.mInRegs[1];
                inst->B.value = 0xdeadbeef;
            }
            else
                inst->B.valid = false;

            if(mInstr.mNumInRegs >= 3)
            {
                inp_reg_processed_this_instr++;
                inst->C.valid = true;
                inst->C.is_int = reg_is_int(mInstr.mInRegs.at(2));
                inst->C.log_reg = mInstr.mInRegs[2];
                inst->C.value = 0xdeadbeef;
            }
            else
                inst->C.valid = false;
        }

        assert(inp_reg_processed_this_instr<= 3);

        if(create_base_update_op)   // store output handled here
        {
            assert(base_update_reg_present);
            assert(base_upd_reg == *mInstr.mOutRegs.rbegin());
            inst->D.valid = true;
            inst->D.is_int = reg_is_int(base_upd_reg);
            assert(inst->D.is_int);
            inst->D.log_reg = base_upd_reg;
            inst->D.value = *mInstr.mOutRegsValues.rbegin();
        }
        else if(!is_store(mInstr.mType) && mInstr.mNumOutRegs >= 1)
        {
            inst->D.valid = true;
            // Flag register is considered to be INT
            inst->D.is_int = reg_is_int(mInstr.mOutRegs.at(mCrackRegIdx));
            inst->D.log_reg = mInstr.mOutRegs[mCrackRegIdx];
            inst->D.value = mInstr.mOutRegsValues.at(mCrackValIdx);
            // if SIMD register, we processed one more 64-bit lane.
            if(!inst->D.is_int)
                start_fp_reg++;
            else
                start_fp_reg = 0;
        }
        else
        {
            inst->D.valid = false;
            start_fp_reg = 0;
        }

        inst->is_load = create_base_update_op ? false : mInstr.mType == InstClass::loadInstClass;
        inst->is_store = create_base_update_op ? false : mInstr.mType == InstClass::storeInstClass;
        inst->addr = mInstr.mEffAddr + (mProcessedPieces * mSizeFactor);
        inst->size = std::max((uint64_t)1, (uint64_t)mSizeFactor);

        assert(inst->size || !(inst->is_load || inst->is_store));
        assert(mProcessedPieces < mTotalPieces);

        // At this point, if mProcessedPieces is 0, the next statements will have no effect.
        mProcessedPieces++;
        inst->is_last_piece = mProcessedPieces == mTotalPieces;

        // If there are more output registers to be processed and they are SIMD
        if(mInstr.mNumOutRegs > mCrackRegIdx && !reg_is_int(mInstr.mOutRegs.at(mCrackRegIdx)))
        {
            // Next output value is in the next 64-bit lane
            mCrackValIdx++;
            // If we processed the high-order bits of the current SIMD register, the next output is a different register
            if(start_fp_reg % 2 == 0)
                mCrackRegIdx++;
        }
        // If there are more output INT registers, go to next value and next register name
        else
        {
            mCrackValIdx++;
            mCrackRegIdx++;
        }

        return inst;
    }

    // Read bytes from the trace and populate a buffer object.
    // Returns true if something was read from the trace, false if we the trace is over.
    bool readInstr()
    {
        // Trace Format :
        // Inst PC                  - 8 bytes
        // Inst Type                - 1 byte
        // If load/storeInst
        //   Effective Address      - 8 bytes
        //   Access Size (total)    - 1 byte
        //   Involves Base Update   - 1 byte
        //   If Store:
        //      Involves Reg Offset - 1 byte
        // If branch
        //   Taken                  - 1 byte
        //   If Taken:
        //      Target              - 8 bytes
        // Num Input Regs           - 1 byte
        // Input Reg Names          - 1 byte each
        // Num Output Regs          - 1 byte
        // Output Reg Names         - 1 byte each
        // Output Reg Values
        //   If INT                 - 8 bytes each
        //   If SIMD                - 16 bytes each
        //
        // Int registers are encoded 0-30(GPRs), 31(Stack Pointer Register), 64(Flag Register), 65(Zero Register)
        // SIMD registers are encoded 32-63
        mInstr.reset();
        start_fp_reg = 0;

        dpressed_input->read((char*) &mInstr.mPc, sizeof(mInstr.mPc));

        if(dpressed_input->eof())
        {
            std::cout<<"EOF"<<std::endl;
            return false;
        }

        // reset bookkeeping variables
        mTotalPieces = 0;
        mMemPieces = 0;
        mProcessedPieces = 0;
        mSizeFactor = 1;
        mCrackRegIdx = 0;
        mCrackValIdx = 0;

        // default NextPc
        mInstr.mNextPc = mInstr.mPc + 4;

        dpressed_input->read((char*) &mInstr.mType, sizeof(mInstr.mType));

        assert(mInstr.mType != InstClass::undefInstClass);

        //EffAddr is the base address
        if(mInstr.mType == InstClass::loadInstClass || mInstr.mType == InstClass::storeInstClass)
        {
            dpressed_input->read((char*) &mInstr.mEffAddr, sizeof(mInstr.mEffAddr));
            dpressed_input->read((char*) &mInstr.mMemSize, sizeof(mInstr.mMemSize));
            dpressed_input->read((char*) &mInstr.mBaseUpd, sizeof(mInstr.mBaseUpd));
            if(mInstr.mType == InstClass::storeInstClass)
            {
                dpressed_input->read((char*) &mInstr.mHasRegOffset, sizeof(mInstr.mHasRegOffset));
            }
        }

        if(is_br(mInstr.mType))
        {
            dpressed_input->read((char*) &mInstr.mTaken, sizeof(mInstr.mTaken));
            if(!is_cond_br(mInstr.mType))
            {
                assert(mInstr.mTaken);
            }
            if(mInstr.mTaken)
            {
                dpressed_input->read((char*) &mInstr.mNextPc, sizeof(mInstr.mNextPc));
            }
        }

        dpressed_input->read((char*) &mInstr.mNumInRegs, sizeof(mInstr.mNumInRegs));

        // capture logical src reg
        for(auto i = 0; i != mInstr.mNumInRegs; i++)
        {
            uint8_t inReg;
            dpressed_input->read((char*) &inReg, sizeof(inReg));
            mInstr.mInRegs.push_back(inReg);
        }

        dpressed_input->read((char*) &mInstr.mNumOutRegs, sizeof(mInstr.mNumOutRegs));

        // capture logical dst reg
        for(auto i = 0; i != mInstr.mNumOutRegs; i++)
        {
            uint8_t outReg;
            dpressed_input->read((char*) &outReg, sizeof(outReg));
            mInstr.mOutRegs.push_back(outReg);
        }

        // assumes 1 piece per logical register output
        mTotalPieces =  (mInstr.mNumOutRegs > 0) ? mInstr.mNumOutRegs : 1;

        const bool base_update_present = mInstr.capture_base_update_log_reg();

        uint8_t base_upd_pos_in_out_regs = UINT8_MAX;
        uint64_t base_upd_val = UINT64_MAX;

        for(auto i = 0; i != mInstr.mNumOutRegs; i++)
        {
            uint64_t val;

            dpressed_input->read((char*) &val, sizeof(val));

            const bool matching_base_upd = base_update_present && mInstr.mBaseUpdReg.value() == mInstr.mOutRegs[i];
            if(matching_base_upd) // capture base_upd_val and skip pushing it to OutRegVal
            {
                assert(base_upd_pos_in_out_regs == UINT8_MAX);
                base_upd_pos_in_out_regs = i;
                base_upd_val = val;
            }
            else
            {
                mInstr.mOutRegsValues.push_back(val);

                // if  writing to some FP/SIMD/SVE reg and upper half is non-zero
                if(!reg_is_int(mInstr.mOutRegs[i]))
                {
                    assert(!is_store(mInstr.mType) && "Stores don't expect base updates for FP/SIMD/SVE regs");
                    dpressed_input->read((char*) &val, sizeof(val));
                    mInstr.mOutRegsValues.push_back(val);
                    if(val != 0)
                    {
                        mTotalPieces++;
                    }
                }
            }
        }

        const bool is_macro_op_mem = is_mem(mInstr.mType);
        // move dst/val to end of dst reg/vals 
        if(base_update_present)
        {
            assert(is_macro_op_mem);
            assert(base_upd_pos_in_out_regs != UINT8_MAX);
            assert(mInstr.mOutRegs[base_upd_pos_in_out_regs] == mInstr.mBaseUpdReg.value());
            if(mInstr.mOutRegs.size() > 1)
            {
                mInstr.mOutRegs.erase(mInstr.mOutRegs.begin() + base_upd_pos_in_out_regs);
                mInstr.mOutRegs.push_back(mInstr.mBaseUpdReg.value());
            }
            mInstr.mOutRegsValues.push_back(base_upd_val);
        }
        else
        {
            assert(base_upd_pos_in_out_regs == UINT8_MAX);
        }


        if(is_store(mInstr.mType) ) // special handling of stores
        {
            // allow 1 addr reg, 1 offset reg and 1 value sources for store operations
            // assumes first input register is the address register in populateNewInstr
            const uint8_t str_val_regs = mInstr.mNumInRegs - (1 + mInstr.mHasRegOffset); // accounting for store address reg + offset
            uint8_t true_str_val_regs = (str_val_regs  == 0) ? 1 : str_val_regs;
            if(mInstr.mMemSize%true_str_val_regs != 0)
            {
                std::cout<<"Store! Size:"<<(uint64_t)mInstr.mMemSize<<" Expected value registers"<<(uint64_t)true_str_val_regs<<std::endl;
                std::cout<<"str_val_regs:"<<(uint64_t)str_val_regs<<" InputRegCount:"<<(uint64_t)mInstr.mNumInRegs<<" REgOffset:"<<(uint64_t)mInstr.mHasRegOffset<<std::endl;
            }
            assert(mInstr.mMemSize%true_str_val_regs == 0); // all pieces should be of same size
            mMemPieces = true_str_val_regs;

            assert(mMemPieces > 0);

            mTotalPieces = mMemPieces + base_update_present;
            mSizeFactor = mInstr.mMemSize/mMemPieces;
        }
        else if(is_load(mInstr.mType))
        {
            // for loads, total_pieces included base_update reg already
            mMemPieces = mTotalPieces - base_update_present;
            assert(mMemPieces > 0);
            mSizeFactor = mInstr.mMemSize/mMemPieces;
        }
        else
        {
            mMemPieces = 0;
            mSizeFactor = mInstr.mMemSize/mTotalPieces;
            assert(mSizeFactor == 0);
        }

        assert(mTotalPieces >= 1);
        if(base_update_present)
        {
            assert(is_macro_op_mem);
            assert(mMemPieces >= 1);
            assert(mTotalPieces == (mMemPieces+ base_update_present));
        }

        if(!is_store(mInstr.mType) )
        {
            assert(mInstr.mNumInRegs <= 3);
        }

        nInstr++;

        if(nInstr % 5000000 == 0)
            std::cout << nInstr << " instrs " << std::endl;

        return true;
    }
};
