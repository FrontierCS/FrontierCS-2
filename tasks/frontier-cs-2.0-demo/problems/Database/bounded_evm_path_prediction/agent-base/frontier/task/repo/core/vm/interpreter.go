// Copyright 2014 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package vm

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/math"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/log"
	"github.com/holiman/uint256"
)

// Config are the configuration options for the Interpreter
type Config struct {
	Debug                   bool      // Enables debugging
	Tracer                  EVMLogger // Opcode logger
	NoBaseFee               bool      // Forces the EIP-1559 baseFee to 0 (needed for 0 price calls)
	EnablePreimageRecording bool      // Enables recording of SHA3/keccak preimages
	ExtraEips               []int     // Additional EIPS that are to be enabled
}

// ScopeContext contains the things that are per-call, such as stack and memory,
// but not transients like pc and gas
type ScopeContext struct {
	Memory    *Memory
	Stack     StackInterface
	Contract  *Contract
	TmpState  *TmpState
	Signature string
}

// EVMInterpreter represents an EVM interpreter
type EVMInterpreter struct {
	evm   *EVM
	table *JumpTable

	hasher    crypto.KeccakState // Keccak256 hasher instance shared across opcodes
	hasherBuf common.Hash        // Keccak256 hasher result array shared aross opcodes

	isPreExecution bool   // Whether to conduct pre-execution
	isFastEnabled  bool   // Whether to initiate fast path execution or repair
	isPredictor    bool   // Whether to enable predictor-based prediction
	checkpoint     bool   // Whether to store checkpoint snapshots
	readOnly       bool   // Whether to throw on stateful modifications
	repair         bool   // Whether the repair needs to be conducted at the end of a transaction execution (only enabled during pre-execution)
	fullStorage    bool   // Whether to store all state variable versions in the mvcache
	returnData     []byte // Last CALL's return data for subsequent reuse

	callMap           map[int]*Snapshot                                      // records the call stack mapping from the call depth to the snapshot
	repairedLoc       map[common.Address]map[common.Hash]map[string]struct{} // records write location leading to pre-execution repair
	txGasDistribution map[common.Hash]int
}

// NewEVMInterpreter returns a new instance of the Interpreter.
func NewEVMInterpreter(evm *EVM, isPreExecution, isPredictor, checkpoint, fullStorage bool, txGasDistribution map[common.Hash]int) *EVMInterpreter {
	// If jump table was not initialised we set the default one.
	var table *JumpTable
	switch {
	case evm.chainRules.IsShanghai:
		t := newShanghaiInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsMerge:
		t := newMergeInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsLondon:
		t := newLondonInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsBerlin:
		t := newBerlinInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsIstanbul:
		t := newIstanbulInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsConstantinople:
		t := newConstantinopleInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsByzantium:
		t := newByzantiumInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsEIP158:
		t := newSpuriousDragonInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsEIP150:
		t := newTangerineWhistleInstructionSet(isPreExecution)
		table = &t
	case evm.chainRules.IsHomestead:
		t := newHomesteadInstructionSet(isPreExecution)
		table = &t
	default:
		t := newFrontierInstructionSet(isPreExecution)
		table = &t
	}
	var extraEips []int
	if len(evm.Config.ExtraEips) > 0 {
		// Deep-copy jumptable to prevent modification of opcodes in other tables
		table = copyJumpTable(table)
	}
	for _, eip := range evm.Config.ExtraEips {
		if err := EnableEIP(eip, table, isPreExecution); err != nil {
			// Disable it, so caller can check if it's activated or not
			log.Error("EIP activation failed", "eip", eip, "error", err)
		} else {
			extraEips = append(extraEips, eip)
		}
	}
	evm.Config.ExtraEips = extraEips
	return &EVMInterpreter{
		evm:               evm,
		table:             table,
		isPreExecution:    isPreExecution,
		isFastEnabled:     false,
		isPredictor:       isPredictor,
		checkpoint:        checkpoint,
		repair:            false,
		fullStorage:       fullStorage,
		repairedLoc:       make(map[common.Address]map[common.Hash]map[string]struct{}),
		txGasDistribution: txGasDistribution,
	}
}

// Run loops and evaluates the contract's code with the given input data and returns
// the return byte-slice and an error if one occurred.
//
// It's important to note that any errors returned by the interpreter should be
// considered a revert-and-consume-all-gas operation except for
// ErrExecutionReverted which means revert-and-keep-gas-left.
func (in *EVMInterpreter) Run(contract *Contract, input []byte, readOnly bool) (ret []byte, err error) {
	// Increment the call depth which is restricted to 1024
	in.evm.depth++
	defer func() { in.evm.depth-- }()

	// Make sure the readOnly is only set if we aren't in readOnly yet.
	// This also makes sure that the readOnly flag isn't removed for child calls.
	if readOnly && !in.readOnly {
		in.readOnly = true
		defer func() { in.readOnly = false }()
	}

	// Reset the previous call's return data. It's unimportant to preserve the old buffer
	// as every returning call will return new data anyway.
	in.returnData = nil

	// Don't bother with the execution if there's no code.
	if len(contract.Code) == 0 {
		return nil, nil
	}

	var (
		op          OpCode        // current opcode
		mem         = NewMemory() // bound memory
		tmp         *TmpState
		callContext *ScopeContext
		stack       StackInterface
		// For optimisation reason we're using uint64 as the program counter.
		// It's theoretically possible to go above 2^64. The YP defines the PC
		// to be uint256. Practically much less so feasible.
		pc   = uint64(0) // program counter
		cost uint64
		// copies used by tracer
		pcCopy  uint64 // needed for the deferred EVMLogger
		gasCopy uint64 // for EVMLogger to log gas remaining before execution
		logged  bool   // deferred EVMLogger should ignore already logged steps
		res     []byte // result of the opcode execution function
	)

	if in.isPreExecution {
		stack = newPreStack()
		tmp = NewTmpState()
		callContext = &ScopeContext{
			Memory:   mem,
			Stack:    stack,
			Contract: contract,
			TmpState: tmp,
		}
	} else {
		stack = newstack()
		tmp = nil
		callContext = &ScopeContext{
			Memory:   mem,
			Stack:    stack,
			Contract: contract,
			TmpState: tmp,
		}
	}

	// use the cached snapshot
	if in.isFastEnabled {
		if snapshot, ok := in.callMap[in.evm.depth]; ok {
			mem = snapshot.GetMemory()
			pc = snapshot.GetPC()
			if in.isPreExecution {
				stack = snapshot.GetTracingStack()
			} else {
				stack = snapshot.GetStack()
			}
			callContext.Memory = mem
			callContext.Stack = stack
		}
	}

	// Don't move this deferred function, it's placed before the capturestate-deferred method,
	// so that it get's executed _after_: the capturestate needs the stacks before
	// they are returned to the pools
	defer func() {
		callContext.Stack.returnStack()
	}()
	contract.Input = input

	if in.evm.Config.Debug {
		defer func() {
			if err != nil {
				if !logged {
					in.evm.Config.Tracer.CaptureState(pcCopy, op, gasCopy, cost, callContext, in.returnData, in.evm.depth, err)
				} else {
					in.evm.Config.Tracer.CaptureFault(pcCopy, op, gasCopy, cost, callContext, in.evm.depth, err)
				}
			}
		}()
	}
	// The Interpreter main run loop (contextual). This loop runs until either an
	// explicit STOP, RETURN or SELFDESTRUCT is executed, an error occurred during
	// the execution of one of the operations or until the done flag is set by the
	// parent context.
	for {
		if in.evm.Config.Debug {
			// Capture pre-execution values for tracing.
			logged, pcCopy, gasCopy = false, pc, contract.Gas
		}
		// Get the operation from the jump table and validate the stack to ensure there are
		// enough stack items available to perform the operation.
		op = contract.GetOp(pc)
		operation := in.table[op]
		cost = operation.constantGas // For tracing
		// Validate stack
		if sLen := callContext.Stack.len(true); sLen < operation.minStack {
			return nil, &ErrStackUnderflow{stackLen: sLen, required: operation.minStack}
		} else if sLen > operation.maxStack {
			return nil, &ErrStackOverflow{stackLen: sLen, limit: operation.maxStack}
		}
		if !contract.UseGas(cost) {
			return nil, ErrOutOfGas
		}
		if operation.dynamicGas != nil {
			// All ops with a dynamic memory usage also has a dynamic gas cost.
			var memorySize uint64
			// calculate the new memory size and expand the memory to fit
			// the operation
			// Memory check needs to be done prior to evaluating the dynamic gas portion,
			// to detect calculation overflows
			if operation.memorySize != nil {
				memSize, overflow := operation.memorySize(callContext.Stack)
				if overflow {
					return nil, ErrGasUintOverflow
				}
				// memory is expanded in words of 32 bytes. Gas
				// is also calculated in words.
				if memorySize, overflow = math.SafeMul(toWordSize(memSize), 32); overflow {
					return nil, ErrGasUintOverflow
				}
			}
			// Consume the gas and return an error if not enough gas is available.
			// cost is explicitly set so that the capture state defer method can get the proper cost
			var dynamicCost uint64
			dynamicCost, err = operation.dynamicGas(in.evm, contract, callContext.Stack, mem, memorySize)
			cost += dynamicCost // for tracing
			if err != nil || !contract.UseGas(dynamicCost) {
				return nil, ErrOutOfGas
			}
			// Do tracing before memory expansion
			if in.evm.Config.Debug {
				in.evm.Config.Tracer.CaptureState(pc, op, gasCopy, cost, callContext, in.returnData, in.evm.depth, err)
				logged = true
			}
			if memorySize > 0 {
				mem.Resize(memorySize)
			}
		} else if in.evm.Config.Debug {
			in.evm.Config.Tracer.CaptureState(pc, op, gasCopy, cost, callContext, in.returnData, in.evm.depth, err)
			logged = true
		}

		// estimate 提前返回
		if in.evm.StateDB.GetDBError() != nil {
			return []byte(""), nil
		}

		// execute the operation
		res, err = operation.execute(&pc, in, callContext)
		if err != nil || in.evm.StateDB.GetDBError() != nil {
			break
		}
		pc++
	}

	if err == errStopToken {
		err = nil // clear stop token error
	}

	return res, err
}

func (in *EVMInterpreter) GetRepair() bool { return in.repair }
func (in *EVMInterpreter) GetRepairedLoc() map[common.Address]map[common.Hash]map[string]struct{} {
	return in.repairedLoc
}
func (in *EVMInterpreter) SetCallMap(callMap map[int]*Snapshot) { in.callMap = callMap }
func (in *EVMInterpreter) SetFastEnabled()                      { in.isFastEnabled = true }

//func (in *EVMInterpreter) ClearRepair() {
//	in.repair = false
//	in.repairedLoc = make(map[common.Address]map[common.Hash]map[int]struct{})
//}

func (in *EVMInterpreter) InsertRepairedLoc(contractAddr common.Address, slot common.Hash, offset uint256.Int) {
	slotMap, ok := in.repairedLoc[contractAddr]
	if ok {
		offsetMap, ok2 := slotMap[slot]
		if ok2 {
			offsetMap[offset.String()] = struct{}{}
		} else {
			offsetMap = make(map[string]struct{})
			offsetMap[offset.String()] = struct{}{}
			slotMap[slot] = offsetMap
		}
	} else {
		slotMap = make(map[common.Hash]map[string]struct{})
		offsetMap := make(map[string]struct{})
		offsetMap[offset.String()] = struct{}{}
		slotMap[slot] = offsetMap
		in.repairedLoc[contractAddr] = slotMap
	}
}
