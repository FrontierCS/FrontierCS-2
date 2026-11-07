// Copyright 2015 The go-ethereum Authors
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
	"fmt"
	"github.com/ethereum/go-ethereum/params"
)

type (
	executionFunc func(pc *uint64, interpreter *EVMInterpreter, callContext *ScopeContext) ([]byte, error)
	gasFunc       func(*EVM, *Contract, StackInterface, *Memory, uint64) (uint64, error) // last parameter is the requested memory size as a uint64
	// memorySizeFunc returns the required size, and whether the operation overflowed a uint64
	memorySizeFunc func(stackInterface StackInterface) (size uint64, overflow bool)
)

type operation struct {
	// execute is the operation function
	execute     executionFunc
	constantGas uint64
	dynamicGas  gasFunc
	// minStack tells how many stack items are required
	minStack int
	// maxStack specifies the max length the stack can have for this operation
	// to not overflow the stack.
	maxStack int

	// memorySize returns the memory size required for the operation
	memorySize memorySizeFunc
}

//var (
//	frontierInstructionSet         = newFrontierInstructionSet()
//	homesteadInstructionSet        = newHomesteadInstructionSet()
//	tangerineWhistleInstructionSet = newTangerineWhistleInstructionSet()
//	spuriousDragonInstructionSet   = newSpuriousDragonInstructionSet()
//	byzantiumInstructionSet        = newByzantiumInstructionSet()
//	constantinopleInstructionSet   = newConstantinopleInstructionSet()
//	istanbulInstructionSet         = newIstanbulInstructionSet()
//	berlinInstructionSet           = newBerlinInstructionSet()
//	londonInstructionSet           = newLondonInstructionSet()
//	mergeInstructionSet            = newMergeInstructionSet()
//	shanghaiInstructionSet         = newShanghaiInstructionSet()
//)

// JumpTable contains the EVM opcodes supported at a given fork.
type JumpTable [256]*operation

func validate(jt JumpTable) JumpTable {
	for i, op := range jt {
		if op == nil {
			panic(fmt.Sprintf("op %#x is not set", i))
		}
		// The interpreter has an assumption that if the memorySize function is
		// set, then the dynamicGas function is also set. This is a somewhat
		// arbitrary assumption, and can be removed if we need to -- but it
		// allows us to avoid a condition check. As long as we have that assumption
		// in there, this little sanity check prevents us from merging in a
		// change which violates it.
		if op.memorySize != nil && op.dynamicGas == nil {
			panic(fmt.Sprintf("op %v has dynamic memory but not dynamic gas", OpCode(i).String()))
		}
	}
	return jt
}

func newShanghaiInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newMergeInstructionSet(isPreExecution)
	enable3855(&instructionSet, isPreExecution) // PUSH0 instruction
	enable3860(&instructionSet, isPreExecution) // Limit and meter initcode
	return validate(instructionSet)
}

func newMergeInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newLondonInstructionSet(isPreExecution)
	var execute executionFunc
	if isPreExecution {
		execute = popRandom
	} else {
		execute = opRandom
	}
	instructionSet[PREVRANDAO] = &operation{
		execute:     execute,
		constantGas: GasQuickStep,
		minStack:    minStack(0, 1),
		maxStack:    maxStack(0, 1),
	}
	return validate(instructionSet)
}

// newLondonInstructionSet returns the frontier, homestead, byzantium,
// constantinople, istanbul, petersburg, berlin and london instructions.
func newLondonInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newBerlinInstructionSet(isPreExecution)
	enable3529(&instructionSet, isPreExecution) // EIP-3529: Reduction in refunds https://eips.ethereum.org/EIPS/eip-3529
	enable3198(&instructionSet, isPreExecution) // Base fee opcode https://eips.ethereum.org/EIPS/eip-3198
	return validate(instructionSet)
}

// newBerlinInstructionSet returns the frontier, homestead, byzantium,
// constantinople, istanbul, petersburg and berlin instructions.
func newBerlinInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newIstanbulInstructionSet(isPreExecution)
	enable2929(&instructionSet, isPreExecution) // Access lists for trie accesses https://eips.ethereum.org/EIPS/eip-2929
	return validate(instructionSet)
}

// newIstanbulInstructionSet returns the frontier, homestead, byzantium,
// constantinople, istanbul and petersburg instructions.
func newIstanbulInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newConstantinopleInstructionSet(isPreExecution)

	enable1344(&instructionSet, isPreExecution) // ChainID opcode - https://eips.ethereum.org/EIPS/eip-1344
	enable1884(&instructionSet, isPreExecution) // Reprice reader opcodes - https://eips.ethereum.org/EIPS/eip-1884
	enable2200(&instructionSet, isPreExecution) // Net metered SSTORE - https://eips.ethereum.org/EIPS/eip-2200

	return validate(instructionSet)
}

// newConstantinopleInstructionSet returns the frontier, homestead,
// byzantium and constantinople instructions.
func newConstantinopleInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newByzantiumInstructionSet(isPreExecution)
	var op []executionFunc
	if isPreExecution {
		op = []executionFunc{popSHL, popSHR, popSAR, popExtCodeHash, popCreate2}
	} else {
		op = []executionFunc{opSHL, opSHR, opSAR, opExtCodeHash, opCreate2}
	}
	instructionSet[SHL] = &operation{
		execute:     op[0],
		constantGas: GasFastestStep,
		minStack:    minStack(2, 1),
		maxStack:    maxStack(2, 1),
	}
	instructionSet[SHR] = &operation{
		execute:     op[1],
		constantGas: GasFastestStep,
		minStack:    minStack(2, 1),
		maxStack:    maxStack(2, 1),
	}
	instructionSet[SAR] = &operation{
		execute:     op[2],
		constantGas: GasFastestStep,
		minStack:    minStack(2, 1),
		maxStack:    maxStack(2, 1),
	}
	instructionSet[EXTCODEHASH] = &operation{
		execute:     op[3],
		constantGas: params.ExtcodeHashGasConstantinople,
		minStack:    minStack(1, 1),
		maxStack:    maxStack(1, 1),
	}
	instructionSet[CREATE2] = &operation{
		execute:     op[4],
		constantGas: params.Create2Gas,
		dynamicGas:  gasCreate2,
		minStack:    minStack(4, 1),
		maxStack:    maxStack(4, 1),
		memorySize:  memoryCreate2,
	}
	return validate(instructionSet)
}

// newByzantiumInstructionSet returns the frontier, homestead and
// byzantium instructions.
func newByzantiumInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newSpuriousDragonInstructionSet(isPreExecution)
	var op []executionFunc
	if isPreExecution {
		op = []executionFunc{popStaticCall, popReturnDataSize, popReturnDataCopy, popRevert}
	} else {
		op = []executionFunc{opStaticCall, opReturnDataSize, opReturnDataCopy, opRevert}
	}
	instructionSet[STATICCALL] = &operation{
		execute:     op[0],
		constantGas: params.CallGasEIP150,
		dynamicGas:  gasStaticCall,
		minStack:    minStack(6, 1),
		maxStack:    maxStack(6, 1),
		memorySize:  memoryStaticCall,
	}
	instructionSet[RETURNDATASIZE] = &operation{
		execute:     op[1],
		constantGas: GasQuickStep,
		minStack:    minStack(0, 1),
		maxStack:    maxStack(0, 1),
	}
	instructionSet[RETURNDATACOPY] = &operation{
		execute:     op[2],
		constantGas: GasFastestStep,
		dynamicGas:  gasReturnDataCopy,
		minStack:    minStack(3, 0),
		maxStack:    maxStack(3, 0),
		memorySize:  memoryReturnDataCopy,
	}
	instructionSet[REVERT] = &operation{
		execute:    op[3],
		dynamicGas: gasRevert,
		minStack:   minStack(2, 0),
		maxStack:   maxStack(2, 0),
		memorySize: memoryRevert,
	}
	return validate(instructionSet)
}

// EIP 158 a.k.a Spurious Dragon
func newSpuriousDragonInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newTangerineWhistleInstructionSet(isPreExecution)
	instructionSet[EXP].dynamicGas = gasExpEIP158
	return validate(instructionSet)
}

// EIP 150 a.k.a Tangerine Whistle
func newTangerineWhistleInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newHomesteadInstructionSet(isPreExecution)
	instructionSet[BALANCE].constantGas = params.BalanceGasEIP150
	instructionSet[EXTCODESIZE].constantGas = params.ExtcodeSizeGasEIP150
	instructionSet[SLOAD].constantGas = params.SloadGasEIP150
	instructionSet[EXTCODECOPY].constantGas = params.ExtcodeCopyBaseEIP150
	instructionSet[CALL].constantGas = params.CallGasEIP150
	instructionSet[CALLCODE].constantGas = params.CallGasEIP150
	instructionSet[DELEGATECALL].constantGas = params.CallGasEIP150
	return validate(instructionSet)
}

// newHomesteadInstructionSet returns the frontier and homestead
// instructions that can be executed during the homestead phase.
func newHomesteadInstructionSet(isPreExecution bool) JumpTable {
	instructionSet := newFrontierInstructionSet(isPreExecution)
	var callOp executionFunc
	if isPreExecution {
		callOp = popDelegateCall
	} else {
		callOp = opDelegateCall
	}
	instructionSet[DELEGATECALL] = &operation{
		execute:     callOp,
		dynamicGas:  gasDelegateCall,
		constantGas: params.CallGasFrontier,
		minStack:    minStack(6, 1),
		maxStack:    maxStack(6, 1),
		memorySize:  memoryDelegateCall,
	}
	return validate(instructionSet)
}

// newFrontierInstructionSet returns the frontier instructions
// that can be executed during the frontier phase.
func newFrontierInstructionSet(isPreExecution bool) JumpTable {
	var op []executionFunc
	if isPreExecution {
		op = []executionFunc{popStop, popAdd, popMul, popSub, popDiv, popSdiv, popMod, popSmod, popAddmod, popMulmod,
			popExp, popSignExtend, popLt, popGt, popSlt, popSgt, popEq, popIszero, popAnd, popXor, popOr, popNot,
			popByte, popKeccak256, popAddress, popBalance, popOrigin, popCaller, popCallValue, popCallDataLoad,
			popCallDataSize, popCallDataCopy, popCodeSize, popCodeCopy, popGasprice, popExtCodeSize, popExtCodeCopy,
			popBlockhash, popCoinbase, popTimestamp, popNumber, popDifficulty, popGasLimit, popPop, popMload, popMstore,
			popMstore8, popSload, popSstore, popJump, popJumpi, popPc, popMsize, popGas, popJumpdest, popPush1,
			makePush2(2, 2), makePush2(3, 3), makePush2(4, 4),
			makePush2(5, 5), makePush2(6, 6), makePush2(7, 7),
			makePush2(8, 8), makePush2(9, 9), makePush2(10, 10),
			makePush2(11, 11), makePush2(12, 12), makePush2(13, 13),
			makePush2(14, 14), makePush2(15, 15), makePush2(16, 16),
			makePush2(17, 17), makePush2(18, 18), makePush2(19, 19),
			makePush2(20, 20), makePush2(21, 21), makePush2(22, 22),
			makePush2(23, 23), makePush2(24, 24), makePush2(25, 25),
			makePush2(26, 26), makePush2(27, 27), makePush2(28, 28),
			makePush2(29, 29), makePush2(30, 30), makePush2(31, 31),
			makePush2(32, 32), makeDup2(1), makeDup2(2), makeDup2(3), makeDup2(4),
			makeDup2(5), makeDup2(6), makeDup2(7), makeDup2(8), makeDup2(9), makeDup2(10),
			makeDup2(11), makeDup2(12), makeDup2(13), makeDup2(14), makeDup2(15), makeDup2(16),
			makeSwap2(1), makeSwap2(2), makeSwap2(3), makeSwap2(4), makeSwap2(5), makeSwap2(6),
			makeSwap2(7), makeSwap2(8), makeSwap2(9), makeSwap2(10), makeSwap2(11), makeSwap2(12),
			makeSwap2(13), makeSwap2(14), makeSwap2(15), makeSwap2(16), makeLog2(0), makeLog2(1),
			makeLog2(2), makeLog2(3), makeLog2(4), popCreate, popCall, popCallCode, popReturn, popSelfdestruct,
		}
	} else {
		op = []executionFunc{opStop, opAdd, opMul, opSub, opDiv, opSdiv, opMod, opSmod, opAddmod, opMulmod,
			opExp, opSignExtend, opLt, opGt, opSlt, opSgt, opEq, opIszero, opAnd, opXor, opOr, opNot,
			opByte, opKeccak256, opAddress, opBalance, opOrigin, opCaller, opCallValue, opCallDataLoad,
			opCallDataSize, opCallDataCopy, opCodeSize, opCodeCopy, opGasprice, opExtCodeSize, opExtCodeCopy,
			opBlockhash, opCoinbase, opTimestamp, opNumber, opDifficulty, opGasLimit, opPop, opMload, opMstore,
			opMstore8, opSload, opSstore, opJump, opJumpi, opPc, opMsize, opGas, opJumpdest, opPush1,
			makePush(2, 2), makePush(3, 3), makePush(4, 4),
			makePush(5, 5), makePush(6, 6), makePush(7, 7),
			makePush(8, 8), makePush(9, 9), makePush(10, 10),
			makePush(11, 11), makePush(12, 12), makePush(13, 13),
			makePush(14, 14), makePush(15, 15), makePush(16, 16),
			makePush(17, 17), makePush(18, 18), makePush(19, 19),
			makePush(20, 20), makePush(21, 21), makePush(22, 22),
			makePush(23, 23), makePush(24, 24), makePush(25, 25),
			makePush(26, 26), makePush(27, 27), makePush(28, 28),
			makePush(29, 29), makePush(30, 30), makePush(31, 31),
			makePush(32, 32), makeDup(1), makeDup(2), makeDup(3), makeDup(4),
			makeDup(5), makeDup(6), makeDup(7), makeDup(8), makeDup(9), makeDup(10),
			makeDup(11), makeDup(12), makeDup(13), makeDup(14), makeDup(15), makeDup(16),
			makeSwap(1), makeSwap(2), makeSwap(3), makeSwap(4), makeSwap(5), makeSwap(6),
			makeSwap(7), makeSwap(8), makeSwap(9), makeSwap(10), makeSwap(11), makeSwap(12),
			makeSwap(13), makeSwap(14), makeSwap(15), makeSwap(16), makeLog(0), makeLog(1),
			makeLog(2), makeLog(3), makeLog(4), opCreate, opCall, opCallCode, opReturn, opSelfdestruct,
		}
	}

	tbl := JumpTable{
		STOP: {
			execute:     op[0],
			constantGas: 0,
			minStack:    minStack(0, 0),
			maxStack:    maxStack(0, 0),
		},
		ADD: {
			execute:     op[1],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		MUL: {
			execute:     op[2],
			constantGas: GasFastStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		SUB: {
			execute:     op[3],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		DIV: {
			execute:     op[4],
			constantGas: GasFastStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		SDIV: {
			execute:     op[5],
			constantGas: GasFastStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		MOD: {
			execute:     op[6],
			constantGas: GasFastStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		SMOD: {
			execute:     op[7],
			constantGas: GasFastStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		ADDMOD: {
			execute:     op[8],
			constantGas: GasMidStep,
			minStack:    minStack(3, 1),
			maxStack:    maxStack(3, 1),
		},
		MULMOD: {
			execute:     op[9],
			constantGas: GasMidStep,
			minStack:    minStack(3, 1),
			maxStack:    maxStack(3, 1),
		},
		EXP: {
			execute:    op[10],
			dynamicGas: gasExpFrontier,
			minStack:   minStack(2, 1),
			maxStack:   maxStack(2, 1),
		},
		SIGNEXTEND: {
			execute:     op[11],
			constantGas: GasFastStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		LT: {
			execute:     op[12],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		GT: {
			execute:     op[13],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		SLT: {
			execute:     op[14],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		SGT: {
			execute:     op[15],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		EQ: {
			execute:     op[16],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		ISZERO: {
			execute:     op[17],
			constantGas: GasFastestStep,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
		},
		AND: {
			execute:     op[18],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		XOR: {
			execute:     op[19],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		OR: {
			execute:     op[20],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		NOT: {
			execute:     op[21],
			constantGas: GasFastestStep,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
		},
		BYTE: {
			execute:     op[22],
			constantGas: GasFastestStep,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
		},
		KECCAK256: {
			execute:     op[23],
			constantGas: params.Keccak256Gas,
			dynamicGas:  gasKeccak256,
			minStack:    minStack(2, 1),
			maxStack:    maxStack(2, 1),
			memorySize:  memoryKeccak256,
		},
		ADDRESS: {
			execute:     op[24],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		BALANCE: {
			execute:     op[25],
			constantGas: params.BalanceGasFrontier,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
		},
		ORIGIN: {
			execute:     op[26],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		CALLER: {
			execute:     op[27],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		CALLVALUE: {
			execute:     op[28],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		CALLDATALOAD: {
			execute:     op[29],
			constantGas: GasFastestStep,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
		},
		CALLDATASIZE: {
			execute:     op[30],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		CALLDATACOPY: {
			execute:     op[31],
			constantGas: GasFastestStep,
			dynamicGas:  gasCallDataCopy,
			minStack:    minStack(3, 0),
			maxStack:    maxStack(3, 0),
			memorySize:  memoryCallDataCopy,
		},
		CODESIZE: {
			execute:     op[32],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		CODECOPY: {
			execute:     op[33],
			constantGas: GasFastestStep,
			dynamicGas:  gasCodeCopy,
			minStack:    minStack(3, 0),
			maxStack:    maxStack(3, 0),
			memorySize:  memoryCodeCopy,
		},
		GASPRICE: {
			execute:     op[34],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		EXTCODESIZE: {
			execute:     op[35],
			constantGas: params.ExtcodeSizeGasFrontier,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
		},
		EXTCODECOPY: {
			execute:     op[36],
			constantGas: params.ExtcodeCopyBaseFrontier,
			dynamicGas:  gasExtCodeCopy,
			minStack:    minStack(4, 0),
			maxStack:    maxStack(4, 0),
			memorySize:  memoryExtCodeCopy,
		},
		BLOCKHASH: {
			execute:     op[37],
			constantGas: GasExtStep,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
		},
		COINBASE: {
			execute:     op[38],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		TIMESTAMP: {
			execute:     op[39],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		NUMBER: {
			execute:     op[40],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		DIFFICULTY: {
			execute:     op[41],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		GASLIMIT: {
			execute:     op[42],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		POP: {
			execute:     op[43],
			constantGas: GasQuickStep,
			minStack:    minStack(1, 0),
			maxStack:    maxStack(1, 0),
		},
		MLOAD: {
			execute:     op[44],
			constantGas: GasFastestStep,
			dynamicGas:  gasMLoad,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
			memorySize:  memoryMLoad,
		},
		MSTORE: {
			execute:     op[45],
			constantGas: GasFastestStep,
			dynamicGas:  gasMStore,
			minStack:    minStack(2, 0),
			maxStack:    maxStack(2, 0),
			memorySize:  memoryMStore,
		},
		MSTORE8: {
			execute:     op[46],
			constantGas: GasFastestStep,
			dynamicGas:  gasMStore8,
			memorySize:  memoryMStore8,
			minStack:    minStack(2, 0),
			maxStack:    maxStack(2, 0),
		},
		SLOAD: {
			execute:     op[47],
			constantGas: params.SloadGasFrontier,
			minStack:    minStack(1, 1),
			maxStack:    maxStack(1, 1),
		},
		SSTORE: {
			execute:    op[48],
			dynamicGas: gasSStore,
			minStack:   minStack(2, 0),
			maxStack:   maxStack(2, 0),
		},
		JUMP: {
			execute:     op[49],
			constantGas: GasMidStep,
			minStack:    minStack(1, 0),
			maxStack:    maxStack(1, 0),
		},
		JUMPI: {
			execute:     op[50],
			constantGas: GasSlowStep,
			minStack:    minStack(2, 0),
			maxStack:    maxStack(2, 0),
		},
		PC: {
			execute:     op[51],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		MSIZE: {
			execute:     op[52],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		GAS: {
			execute:     op[53],
			constantGas: GasQuickStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		JUMPDEST: {
			execute:     op[54],
			constantGas: params.JumpdestGas,
			minStack:    minStack(0, 0),
			maxStack:    maxStack(0, 0),
		},
		PUSH1: {
			execute:     op[55],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH2: {
			execute:     op[56],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH3: {
			execute:     op[57],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH4: {
			execute:     op[58],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH5: {
			execute:     op[59],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH6: {
			execute:     op[60],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH7: {
			execute:     op[61],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH8: {
			execute:     op[62],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH9: {
			execute:     op[63],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH10: {
			execute:     op[64],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH11: {
			execute:     op[65],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH12: {
			execute:     op[66],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH13: {
			execute:     op[67],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH14: {
			execute:     op[68],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH15: {
			execute:     op[69],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH16: {
			execute:     op[70],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH17: {
			execute:     op[71],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH18: {
			execute:     op[72],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH19: {
			execute:     op[73],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH20: {
			execute:     op[74],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH21: {
			execute:     op[75],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH22: {
			execute:     op[76],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH23: {
			execute:     op[77],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH24: {
			execute:     op[78],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH25: {
			execute:     op[79],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH26: {
			execute:     op[80],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH27: {
			execute:     op[81],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH28: {
			execute:     op[82],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH29: {
			execute:     op[83],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH30: {
			execute:     op[84],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH31: {
			execute:     op[85],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		PUSH32: {
			execute:     op[86],
			constantGas: GasFastestStep,
			minStack:    minStack(0, 1),
			maxStack:    maxStack(0, 1),
		},
		DUP1: {
			execute:     op[87],
			constantGas: GasFastestStep,
			minStack:    minDupStack(1),
			maxStack:    maxDupStack(1),
		},
		DUP2: {
			execute:     op[88],
			constantGas: GasFastestStep,
			minStack:    minDupStack(2),
			maxStack:    maxDupStack(2),
		},
		DUP3: {
			execute:     op[89],
			constantGas: GasFastestStep,
			minStack:    minDupStack(3),
			maxStack:    maxDupStack(3),
		},
		DUP4: {
			execute:     op[90],
			constantGas: GasFastestStep,
			minStack:    minDupStack(4),
			maxStack:    maxDupStack(4),
		},
		DUP5: {
			execute:     op[91],
			constantGas: GasFastestStep,
			minStack:    minDupStack(5),
			maxStack:    maxDupStack(5),
		},
		DUP6: {
			execute:     op[92],
			constantGas: GasFastestStep,
			minStack:    minDupStack(6),
			maxStack:    maxDupStack(6),
		},
		DUP7: {
			execute:     op[93],
			constantGas: GasFastestStep,
			minStack:    minDupStack(7),
			maxStack:    maxDupStack(7),
		},
		DUP8: {
			execute:     op[94],
			constantGas: GasFastestStep,
			minStack:    minDupStack(8),
			maxStack:    maxDupStack(8),
		},
		DUP9: {
			execute:     op[95],
			constantGas: GasFastestStep,
			minStack:    minDupStack(9),
			maxStack:    maxDupStack(9),
		},
		DUP10: {
			execute:     op[96],
			constantGas: GasFastestStep,
			minStack:    minDupStack(10),
			maxStack:    maxDupStack(10),
		},
		DUP11: {
			execute:     op[97],
			constantGas: GasFastestStep,
			minStack:    minDupStack(11),
			maxStack:    maxDupStack(11),
		},
		DUP12: {
			execute:     op[98],
			constantGas: GasFastestStep,
			minStack:    minDupStack(12),
			maxStack:    maxDupStack(12),
		},
		DUP13: {
			execute:     op[99],
			constantGas: GasFastestStep,
			minStack:    minDupStack(13),
			maxStack:    maxDupStack(13),
		},
		DUP14: {
			execute:     op[100],
			constantGas: GasFastestStep,
			minStack:    minDupStack(14),
			maxStack:    maxDupStack(14),
		},
		DUP15: {
			execute:     op[101],
			constantGas: GasFastestStep,
			minStack:    minDupStack(15),
			maxStack:    maxDupStack(15),
		},
		DUP16: {
			execute:     op[102],
			constantGas: GasFastestStep,
			minStack:    minDupStack(16),
			maxStack:    maxDupStack(16),
		},
		SWAP1: {
			execute:     op[103],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(2),
			maxStack:    maxSwapStack(2),
		},
		SWAP2: {
			execute:     op[104],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(3),
			maxStack:    maxSwapStack(3),
		},
		SWAP3: {
			execute:     op[105],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(4),
			maxStack:    maxSwapStack(4),
		},
		SWAP4: {
			execute:     op[106],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(5),
			maxStack:    maxSwapStack(5),
		},
		SWAP5: {
			execute:     op[107],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(6),
			maxStack:    maxSwapStack(6),
		},
		SWAP6: {
			execute:     op[108],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(7),
			maxStack:    maxSwapStack(7),
		},
		SWAP7: {
			execute:     op[109],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(8),
			maxStack:    maxSwapStack(8),
		},
		SWAP8: {
			execute:     op[110],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(9),
			maxStack:    maxSwapStack(9),
		},
		SWAP9: {
			execute:     op[111],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(10),
			maxStack:    maxSwapStack(10),
		},
		SWAP10: {
			execute:     op[112],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(11),
			maxStack:    maxSwapStack(11),
		},
		SWAP11: {
			execute:     op[113],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(12),
			maxStack:    maxSwapStack(12),
		},
		SWAP12: {
			execute:     op[114],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(13),
			maxStack:    maxSwapStack(13),
		},
		SWAP13: {
			execute:     op[115],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(14),
			maxStack:    maxSwapStack(14),
		},
		SWAP14: {
			execute:     op[116],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(15),
			maxStack:    maxSwapStack(15),
		},
		SWAP15: {
			execute:     op[117],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(16),
			maxStack:    maxSwapStack(16),
		},
		SWAP16: {
			execute:     op[118],
			constantGas: GasFastestStep,
			minStack:    minSwapStack(17),
			maxStack:    maxSwapStack(17),
		},
		LOG0: {
			execute:    op[119],
			dynamicGas: makeGasLog(0),
			minStack:   minStack(2, 0),
			maxStack:   maxStack(2, 0),
			memorySize: memoryLog,
		},
		LOG1: {
			execute:    op[120],
			dynamicGas: makeGasLog(1),
			minStack:   minStack(3, 0),
			maxStack:   maxStack(3, 0),
			memorySize: memoryLog,
		},
		LOG2: {
			execute:    op[121],
			dynamicGas: makeGasLog(2),
			minStack:   minStack(4, 0),
			maxStack:   maxStack(4, 0),
			memorySize: memoryLog,
		},
		LOG3: {
			execute:    op[122],
			dynamicGas: makeGasLog(3),
			minStack:   minStack(5, 0),
			maxStack:   maxStack(5, 0),
			memorySize: memoryLog,
		},
		LOG4: {
			execute:    op[123],
			dynamicGas: makeGasLog(4),
			minStack:   minStack(6, 0),
			maxStack:   maxStack(6, 0),
			memorySize: memoryLog,
		},
		CREATE: {
			execute:     op[124],
			constantGas: params.CreateGas,
			dynamicGas:  gasCreate,
			minStack:    minStack(3, 1),
			maxStack:    maxStack(3, 1),
			memorySize:  memoryCreate,
		},
		CALL: {
			execute:     op[125],
			constantGas: params.CallGasFrontier,
			dynamicGas:  gasCall,
			minStack:    minStack(7, 1),
			maxStack:    maxStack(7, 1),
			memorySize:  memoryCall,
		},
		CALLCODE: {
			execute:     op[126],
			constantGas: params.CallGasFrontier,
			dynamicGas:  gasCallCode,
			minStack:    minStack(7, 1),
			maxStack:    maxStack(7, 1),
			memorySize:  memoryCall,
		},
		RETURN: {
			execute:    op[127],
			dynamicGas: gasReturn,
			minStack:   minStack(2, 0),
			maxStack:   maxStack(2, 0),
			memorySize: memoryReturn,
		},
		SELFDESTRUCT: {
			execute:    op[128],
			dynamicGas: gasSelfdestruct,
			minStack:   minStack(1, 0),
			maxStack:   maxStack(1, 0),
		},
	}

	// Fill all unassigned slots with opUndefined.
	for i, entry := range tbl {
		if entry == nil {
			if isPreExecution {
				tbl[i] = &operation{execute: popUndefined, maxStack: maxStack(0, 0)}
			} else {
				tbl[i] = &operation{execute: opUndefined, maxStack: maxStack(0, 0)}
			}
		}
	}

	return validate(tbl)
}

func copyJumpTable(source *JumpTable) *JumpTable {
	dest := *source
	for i, op := range source {
		if op != nil {
			opCopy := *op
			dest[i] = &opCopy
		}
	}
	return &dest
}
