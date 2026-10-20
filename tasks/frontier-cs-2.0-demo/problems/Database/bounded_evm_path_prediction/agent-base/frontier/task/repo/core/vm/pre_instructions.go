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
	"errors"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/params"
	"github.com/holiman/uint256"
	"seerEVM/core/types"
	"strings"
	"sync/atomic"
)

const (
	skip = iota
	uncertain
	taken
	notTaken
)

// merge updates tracking units on the stack when performing some computation operations (merges top two units on the stack)
func merge(topVal, originVal, curVal uint256.Int, operation string, topUnit, changedUnit TracingUnit, scope *ScopeContext, record bool) {
	var label int
	if unit, ok := changedUnit.(*StateUnit); ok {
		unit.SetValue(curVal)
		if record {
			switch u := topUnit.(type) {
			case *NormalUnit:
				label = NORMAL
				unit.Record(operation, topVal, true, label, u.Copy())
			case *CallDataUnit:
				label = INPUT
				unit.Record(operation, topVal, true, label, u.Copy())
			case *StateUnit:
				label = STATE
				notEnvTop := strings.Compare(u.GetBlockEnv(), "nil") == 0
				notEnvSecond := strings.Compare(unit.GetBlockEnv(), "nil") == 0
				if notEnvTop && !notEnvSecond {
					u.SetValue(curVal)
					unit.SetValue(originVal)
					u.Record(operation, originVal, false, label, unit.Copy())
					scope.Stack.override(u)
				} else {
					unit.Record(operation, topVal, true, label, u.Copy())
				}
			}
		}
	} else {
		// 反向合并
		if unit2, ok2 := topUnit.(*StateUnit); ok2 {
			// in case that storage compact has happened
			unit2.SetValue(curVal)
			if record {
				switch u := changedUnit.(type) {
				case *NormalUnit:
					label = NORMAL
					unit2.Record(operation, originVal, false, label, u.Copy())
				case *CallDataUnit:
					label = INPUT
					unit2.Record(operation, originVal, false, label, u.Copy())
				}
			}
			scope.Stack.override(unit2)
		} else {
			if unit3, ok3 := topUnit.(*CallDataUnit); ok3 {
				unit3.SetValue(curVal)
				scope.Stack.override(unit3)
			} else {
				changedUnit.SetValue(curVal)
			}
		}
	}
}

// branchRecord records the relevant branch (related to state variables) info into the state variable table
func branchRecord(pc, jumpPc uint64, interpreter *EVMInterpreter, scope *ScopeContext, topUnit, secondUnit TracingUnit, judgement string) (uint256.Int, uint256.Int, int, error) {
	var branchID string
	var compact bool
	interpreter.evm.VarTable.BranchNum++
	tunit, ok1 := topUnit.(*StateUnit)
	sunit, ok2 := secondUnit.(*StateUnit)
	if ok1 {
		interpreter.evm.VarTable.StateRelated++
		var entry *Entry
		txID := interpreter.evm.TxContext.ID
		ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
		if interpreter.txGasDistribution != nil {
			switch interpreter.txGasDistribution[txID] {
			case 0:
				interpreter.evm.VarTable.StateRelatedForCTxs0++
			case 1:
				interpreter.evm.VarTable.StateRelatedForCTxs1++
			case 2:
				interpreter.evm.VarTable.StateRelatedForCTxs2++
			case 3:
				interpreter.evm.VarTable.StateRelatedForCTxs3++
			}
		}
		// cache the current snapshot before branch
		if interpreter.checkpoint {
			ret.CacheSnapshot(scope.Stack, scope.Memory, pc, jumpPc, scope.Contract, interpreter.evm.depth)
		} else {
			ret.InitialBranch()
		}

		slot := tunit.GetSlot()
		offset := tunit.GetOffset()
		bits := tunit.GetBits()
		if offset.Uint64() > 0 && bits < 256 {
			compact = true
		}
		signature := scope.Signature
		contractAddr := scope.Contract.Address()

		if ok2 {
			slot2 := sunit.GetSlot()
			offset2 := sunit.GetOffset()
			bits2 := sunit.GetBits()
			value := sunit.GetStorageValue()
			newVar := NewVarInfo(slot2, offset2, bits2)
			branchID = GenerateBranchID(signature, true, newVar, value)
			// The branch including the variable relevant to the block environment will not be stored in the branch info table
			// Instead, it is only stored in the pre-execution table
			if strings.Compare(tunit.GetBlockEnv(), "nil") == 0 {
				entry = branchQuery(interpreter, contractAddr, common.Hash(slot.Bytes32()), branchID, compact, true, offset, value, newVar)
			}
			// store the branch info into the pre-execution table
			ret.UpdateBranchInfo(contractAddr, tunit.Copy(), sunit.Copy(), true, true, branchID, judgement)
		} else {
			value := secondUnit.GetValue()
			branchID = GenerateBranchID(signature, false, nil, value)
			if strings.Compare(tunit.GetBlockEnv(), "nil") == 0 {
				entry = branchQuery(interpreter, contractAddr, common.Hash(slot.Bytes32()), branchID, compact, false, offset, value, nil)
			}
			// store the branch info into the pre-execution table
			ret.UpdateBranchInfo(contractAddr, tunit.Copy(), secondUnit.Copy(), false, true, branchID, judgement)
		}

		// We do not need to perform prediction of branches including the variables relevant to the block environment
		if strings.Compare(tunit.GetBlockEnv(), "nil") == 0 {
			// utilize the predictor model to perform prediction
			if interpreter.isPredictor {
				res := entry.Predict(offset, branchID)
				if res == taken || res == notTaken {
					// obtain relatively certain prediction result, directly output it
					return uint256.Int{}, uint256.Int{}, res, nil
				}
			}

			// utilize the ordering-based prediction to fetch the latest value from the multi-version cache
			firstVal, err := GetComparedVal(interpreter.evm, interpreter.evm.TxContext.ID, contractAddr, slot, offset, bits, tunit, compact, true)
			if err != nil {
				return uint256.Int{}, uint256.Int{}, uncertain, err
			}
			if ok2 {
				slot2 := sunit.GetSlot()
				offset2 := sunit.GetOffset()
				bits2 := sunit.GetBits()
				compact2 := offset2.Uint64() > 0 && bits2 < 256
				updatedVal, err2 := GetComparedVal(interpreter.evm, interpreter.evm.TxContext.ID, contractAddr, slot2, offset2, bits2, sunit, compact2, true)
				if err2 != nil {
					return uint256.Int{}, uint256.Int{}, uncertain, err2
				}
				entry.UpdateJudgementVal(sunit.GetValue(), updatedVal, offset, branchID)
			}
			_, secVal := entry.GetJudgementVal(offset, branchID)
			return firstVal, secVal, uncertain, nil
		}
	} else if !ok1 && ok2 {
		interpreter.evm.VarTable.StateRelated++
		txID := interpreter.evm.TxContext.ID
		ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
		if interpreter.txGasDistribution != nil {
			switch interpreter.txGasDistribution[txID] {
			case 0:
				interpreter.evm.VarTable.StateRelatedForCTxs0++
			case 1:
				interpreter.evm.VarTable.StateRelatedForCTxs1++
			case 2:
				interpreter.evm.VarTable.StateRelatedForCTxs2++
			case 3:
				interpreter.evm.VarTable.StateRelatedForCTxs3++
			}
		}
		// cache the current snapshot before branch
		if interpreter.checkpoint {
			ret.CacheSnapshot(scope.Stack, scope.Memory, pc, jumpPc, scope.Contract, interpreter.evm.depth)
		} else {
			ret.InitialBranch()
		}

		slot := sunit.GetSlot()
		offset := sunit.GetOffset()
		bits := sunit.GetBits()
		if offset.Uint64() > 0 && bits < 256 {
			compact = true
		}
		signature := scope.Signature
		contractAddr := scope.Contract.Address()
		value := topUnit.GetValue()
		branchID = GenerateBranchID(signature, false, nil, value)
		// store the branch info into the pre-execution table
		ret.UpdateBranchInfo(contractAddr, sunit.Copy(), topUnit.Copy(), false, false, branchID, judgement)

		if strings.Compare(sunit.GetBlockEnv(), "nil") == 0 {
			entry := branchQuery(interpreter, contractAddr, common.Hash(slot.Bytes32()), branchID, compact, false, offset, value, nil)
			// utilize the predictor model to perform prediction
			if interpreter.isPredictor {
				res := entry.Predict(offset, branchID)
				if res == taken || res == notTaken {
					// obtain relatively certain prediction result, directly output it
					return uint256.Int{}, uint256.Int{}, res, nil
				}
			}

			// utilize the ordering-based prediction to fetch the latest value from the multi-version cache
			secVal, err := GetComparedVal(interpreter.evm, interpreter.evm.TxContext.ID, contractAddr, slot, offset, bits, sunit, compact, true)
			if err != nil {
				return uint256.Int{}, uint256.Int{}, uncertain, err
			}
			_, firstVal := entry.GetJudgementVal(offset, branchID)
			return firstVal, secVal, uncertain, nil
		}
	}
	return uint256.Int{}, uint256.Int{}, skip, nil
}

// branchQuery queries if the branch exists, if not, creates a new one
func branchQuery(interpreter *EVMInterpreter, contractAddr common.Address, slot common.Hash, branchID string, compact, isVar bool, offset, value uint256.Int, varInfo *VarInfo) *Entry {
	st, err := interpreter.evm.VarTable.GetSubTable(contractAddr)
	if err != nil {
		// the table does not exist
		st = interpreter.evm.VarTable.InsertSubTable(contractAddr)
		entry := st.InsertEntry(slot, compact)
		entry.GenerateBranchInfo(branchID, offset, value, isVar, varInfo, interpreter.evm.VarTable.GetEpoch())
		return entry
	}
	entry, err2 := st.GetEntry(slot)
	if err2 != nil {
		// the entry does not exist
		entry = st.InsertEntry(slot, compact)
		entry.GenerateBranchInfo(branchID, offset, value, isVar, varInfo, interpreter.evm.VarTable.GetEpoch())
		return entry
	}
	exist := entry.BranchExist(offset, branchID)
	if !exist {
		// the branch does not exist
		entry.GenerateBranchInfo(branchID, offset, value, isVar, varInfo, interpreter.evm.VarTable.GetEpoch())
		return entry
	}
	return entry
}

// fetchInMVCache fetches the latest state version in multi-version cache
func fetchInMVCache(evm *EVM, txID common.Hash, contractAddr common.Address, slot, offset uint256.Int, compact bool) (uint256.Int, error) {
	tip := evm.TxContext.GasTip
	if compact {
		writeVersion, err := evm.MVCache.GetCompactedStorageVersion(contractAddr, slot, offset, tip)
		if err != nil {
			return uint256.Int{}, err
		}
		// record the read operation
		err2 := evm.MVCache.SetCompactedStorageForRead(contractAddr, slot, offset, txID, tip)
		if err2 != nil {
			return uint256.Int{}, err2
		}
		return writeVersion.GetVal(), nil
	} else {
		writeVersion, err := evm.MVCache.GetStorageVersion(contractAddr, slot, tip)
		if err != nil {
			return uint256.Int{}, err
		}
		// record the read operation
		err2 := evm.MVCache.SetStorageForRead(contractAddr, slot, txID, tip)
		if err2 != nil {
			return uint256.Int{}, err2
		}
		return writeVersion.GetVal(), nil
	}
}

// fetchInStateDB fetches the latest state version in stateDB
func fetchInStateDB(evm *EVM, contractAddr common.Address, slot, offset uint256.Int, bits int, compact bool, unit *StateUnit) uint256.Int {
	var newVal uint256.Int
	slotVal := evm.StateDB.GetState(contractAddr, slot.Bytes32())
	newVal.SetBytes(slotVal.Bytes())
	// obtain the state value from the compacted storage
	if compact {
		signExtend := unit.GetSignExtend()
		newVal = fetchStorageVal(newVal, offset, bits, signExtend)
	}
	return newVal
}

// computeTmpVar computes the latest temp variable value based on the state variable related to the branch
func computeTmpVar(evm *EVM, txID common.Hash, contractAddr common.Address, newVal uint256.Int, su *StateUnit, isMultiVersion bool, depth int) (uint256.Int, error) {
	depth++
	// prevent stack overflow
	if depth > 10 {
		return su.GetValue(), errors.New("stack overflow")
	}

	for _, t := range su.opTracer {
		var (
			tmpVal, newVal2 uint256.Int
			noLoop          bool
			err             error
		)
		op := StringToOp(t.op)
		unit := t.GetAttaching()
		su2, ok := unit.(*StateUnit)
		if ok {
			var compact bool
			slot := su2.GetSlot()
			offset := su2.GetOffset()
			bits := su2.GetBits()
			if offset.Uint64() > 0 && bits < 256 {
				compact = true
			}
			notEnv := strings.Compare(su2.GetBlockEnv(), "nil") == 0
			if isMultiVersion {
				if notEnv {
					newVal2, err = fetchInMVCache(evm, txID, contractAddr, slot, offset, compact)
					if err != nil {
						if err.Error() == "not found" {
							newVal2 = su2.GetStorageValue()
						} else {
							return uint256.Int{}, err
						}
					}
				} else {
					noLoop = true
					tmpVal = t.GetVal()
				}
			} else {
				if notEnv {
					newVal2 = fetchInStateDB(evm, contractAddr, slot, offset, bits, compact, su2)
				} else {
					// directly fetch the current blockchain environment
					newVal2 = GetEnvValue(su2.GetBlockEnv(), evm, su2.GetBlockNum(), su2.GetBalAddr())
				}
			}
			if !su2.Compare() && !noLoop {
				tmpVal, err = computeTmpVar(evm, txID, contractAddr, newVal2, su2, isMultiVersion, depth)
				if err != nil {
					if err.Error() == "stack overflow" {
						return tmpVal, err
					}
					return uint256.Int{}, err
				}
			}
		} else {
			tmpVal = t.GetVal()
		}
		direction := t.GetDirection()
		if direction {
			err = Compute(&tmpVal, &newVal, op)
			if err != nil {
				return uint256.Int{}, err
			}
		} else {
			err = Compute(&newVal, &tmpVal, op)
			if err != nil {
				return uint256.Int{}, err
			}
			newVal = tmpVal
		}
	}

	return newVal, nil
}

// GetComparedVal obtains the latest compared value based on whether the current value is not equal to the storage value
func GetComparedVal(evm *EVM, txID common.Hash, contractAddr common.Address, slot, offset uint256.Int, bits int, unit *StateUnit, compact, isMultiVersion bool) (uint256.Int, error) {
	var (
		newVal uint256.Int
		err    error
	)

	if isMultiVersion {
		newVal, err = fetchInMVCache(evm, txID, contractAddr, slot, offset, compact)
		if err != nil {
			if err.Error() == "not found" {
				newVal = unit.GetStorageValue()
			} else {
				return uint256.Int{}, err
			}
		}
	} else {
		notEnv := strings.Compare(unit.GetBlockEnv(), "nil") == 0
		if notEnv {
			newVal = fetchInStateDB(evm, contractAddr, slot, offset, bits, compact, unit)
		} else {
			// directly fetch the current blockchain environment
			newVal = GetEnvValue(unit.GetBlockEnv(), evm, unit.GetBlockNum(), unit.GetBalAddr())
		}
	}

	if !unit.Compare() { // In case of storing a compact variable, its storage value must be different from the current value
		updatedVal, err := computeTmpVar(evm, txID, contractAddr, newVal, unit, isMultiVersion, 0)
		if err != nil && err.Error() != "stack overflow" {
			return uint256.Int{}, err
		}
		return updatedVal, nil
	}
	return newVal, nil
}

// fetchStorageVal fetches the value of state variable that is stored in the slot compactly
func fetchStorageVal(slotVal, offset uint256.Int, bits int, signExtend bool) uint256.Int {
	result := new(uint256.Int)
	if signExtend {
		// 有符号的变量需要符号扩展，取值操作略有不同
		opVal := new(uint256.Int)
		opVal.SetUint64(uint64(bits/8 - 1))
		result.Div(&slotVal, &offset)
		result.ExtendSign(result, opVal)
	} else {
		result.Div(&slotVal, &offset)
		mask := MakeMask(bits)
		result.And(mask, result)
	}
	return *result
}

// MakeMask creates a mask code for storage compact
func MakeMask(bits int) *uint256.Int {
	var buf []byte
	for i := 0; i < bits/8; i++ {
		buf = append(buf, 255)
	}
	integer := new(uint256.Int)
	mask := integer.SetBytes(buf)
	return mask
}

// isMask identifies if a stack value is a mask code
func isMask(x uint256.Int) bool {
	var not bool
	for _, str := range x.Hex()[2:] {
		if strings.Compare(string(str), "f") != 0 && strings.Compare(string(str), "0") != 0 {
			not = true
			break
		}
	}
	return !not
}

func popAdd(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Add(&x, y)
	merge(x, originVal, *y, "ADD", xUnit, yUnit, scope, true)
	return nil, nil
}

func popSub(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Sub(&x, y)
	merge(x, originVal, *y, "SUB", xUnit, yUnit, scope, true)
	return nil, nil
}

func popMul(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Mul(&x, y)
	// update the bit info of the state variable
	normalUnit, ok := yUnit.(*NormalUnit)
	isOffset := ok && normalUnit.GetFlag()
	if isMask(x) && isOffset {
		bits := 4 * (len(x.Hex()) - 2)
		normalUnit.SetBits(bits)
	}
	merge(x, originVal, *y, "MUL", xUnit, yUnit, scope, true)
	return nil, nil
}

func popDiv(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Div(&x, y)
	// update the offset info of the state variable
	//stateUnit, ok1 := xUnit.(*StateUnit)
	//normalUnit, ok2 := yUnit.(*NormalUnit)
	//isOffset := ok2 && normalUnit.GetFlag()
	//if ok1 && isOffset {
	//	// 存在误报
	//	offset := normalUnit.GetOffset()
	//	stateUnit.SetOffset(offset)
	//}
	merge(x, originVal, *y, "DIV", xUnit, yUnit, scope, true)
	return nil, nil
}

func popSdiv(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.SDiv(&x, y)
	merge(x, originVal, *y, "SDIV", xUnit, yUnit, scope, true)
	return nil, nil
}

func popMod(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Mod(&x, y)
	merge(x, originVal, *y, "MOD", xUnit, yUnit, scope, true)
	return nil, nil
}

func popSmod(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.SMod(&x, y)
	merge(x, originVal, *y, "SMOD", xUnit, yUnit, scope, true)
	return nil, nil
}

func popExp(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	base, baseUnit := scope.Stack.pop()
	exponent, exponentUnit := scope.Stack.peek()
	originVal := *exponent
	exponent.Exp(&base, exponent)
	merge(base, originVal, *exponent, "EXP", baseUnit, exponentUnit, scope, true)
	// the storage offset is usually obtained by EXP operation
	if offsetUnit, ok := exponentUnit.(*NormalUnit); ok {
		offsetUnit.SetFlag()
		offsetUnit.SetOffset(*exponent)
	}
	return nil, nil
}

func popSignExtend(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	back, _ := scope.Stack.pop()
	num, numUnit := scope.Stack.peek()
	num.ExtendSign(num, &back)

	if sUnit, ok := numUnit.(*StateUnit); ok {
		ops := sUnit.GetTracer()
		if len(ops) > 0 {
			latestTr := ops[len(ops)-1]
			if strings.Compare(latestTr.GetOps(), "DIV") == 0 && !sUnit.GetSignExtend() {
				// back的值是需要符号扩展数的字节数-1
				offset := latestTr.GetVal()
				bits := 8 * (int(back.Uint64()) + 1)
				sUnit.SetOffset(offset)
				sUnit.SetBits(bits)
				sUnit.UpdateStorageValue(*num)
				sUnit.ClearTracer()
				// 生成新的tmp state进行记录 (防止跟踪丢失)
				fragment, _ := scope.TmpState.GetFragment(scope.Contract.Address(), sUnit.GetSlot())
				fragment.GenerateVar(sUnit)
			}
		}
		sUnit.SetSignExtend()
	}
	numUnit.SetValue(*num)
	return nil, nil
}

func popNot(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.peek()
	x.Not(x)
	xUnit.SetValue(*x)
	unit, ok := xUnit.(*StateUnit)
	if ok {
		unit.Record("NOT", uint256.Int{0}, true, NORMAL, nil)
	}
	return nil, nil
}

func popLt(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	xVal, yVal, res, err := branchRecord(*pc, 0, interpreter, scope, xUnit, yUnit, "LT")
	if err != nil {
		return nil, err
	}
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	switch res {
	case taken:
		y.SetOne()
		ret.UpdateDirection(1)
	case notTaken:
		y.Clear()
		ret.UpdateDirection(0)
	case uncertain:
		if xVal.Lt(&yVal) {
			y.SetOne()
		} else {
			y.Clear()
		}
		ret.UpdateDirection(int(y.Uint64()))
	case skip:
		if x.Lt(y) {
			y.SetOne()
		} else {
			y.Clear()
		}
	}
	merge(x, uint256.Int{}, *y, "LT", xUnit, yUnit, scope, false)
	return nil, nil
}

func popGt(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	xVal, yVal, res, err := branchRecord(*pc, 0, interpreter, scope, xUnit, yUnit, "GT")
	if err != nil {
		return nil, err
	}
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	switch res {
	case taken:
		y.SetOne()
		ret.UpdateDirection(1)
	case notTaken:
		y.Clear()
		ret.UpdateDirection(0)
	case uncertain:
		if xVal.Gt(&yVal) {
			y.SetOne()
		} else {
			y.Clear()
		}
		ret.UpdateDirection(int(y.Uint64()))
	case skip:
		if x.Gt(y) {
			y.SetOne()
		} else {
			y.Clear()
		}
	}
	merge(x, uint256.Int{}, *y, "GT", xUnit, yUnit, scope, false)
	return nil, nil
}

func popSlt(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	xVal, yVal, res, err := branchRecord(*pc, 0, interpreter, scope, xUnit, yUnit, "SLT")
	if err != nil {
		return nil, err
	}
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	switch res {
	case taken:
		y.SetOne()
		ret.UpdateDirection(1)
	case notTaken:
		y.Clear()
		ret.UpdateDirection(0)
	case uncertain:
		if xVal.Slt(&yVal) {
			y.SetOne()
		} else {
			y.Clear()
		}
		ret.UpdateDirection(int(y.Uint64()))
	case skip:
		if x.Slt(y) {
			y.SetOne()
		} else {
			y.Clear()
		}
	}
	merge(x, uint256.Int{}, *y, "SLT", xUnit, yUnit, scope, false)
	return nil, nil
}

func popSgt(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	xVal, yVal, res, err := branchRecord(*pc, 0, interpreter, scope, xUnit, yUnit, "SGT")
	if err != nil {
		return nil, err
	}
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	switch res {
	case taken:
		y.SetOne()
		ret.UpdateDirection(1)
	case notTaken:
		y.Clear()
		ret.UpdateDirection(0)
	case uncertain:
		if xVal.Sgt(&yVal) {
			y.SetOne()
		} else {
			y.Clear()
		}
		ret.UpdateDirection(int(y.Uint64()))
	case skip:
		if x.Sgt(y) {
			y.SetOne()
		} else {
			y.Clear()
		}
	}
	merge(x, uint256.Int{}, *y, "SGT", xUnit, yUnit, scope, false)
	return nil, nil
}

func popEq(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	if x.Eq(y) {
		y.SetOne()
	} else {
		y.Clear()
	}
	merge(x, originVal, *y, "EQ", xUnit, yUnit, scope, false)
	return nil, nil
}

func popIszero(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.peek()
	if x.IsZero() {
		x.SetOne()
	} else {
		x.Clear()
	}
	xUnit.SetValue(*x)
	return nil, nil
}

func popAnd(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.And(&x, y)

	normalUnit, ok := xUnit.(*NormalUnit)
	unit, ok2 := yUnit.(*StateUnit)
	// 每个slot有64位十六进制，加上0x字符串，一共是66位
	if ok && ok2 && isMask(x) {
		if len(x.Hex()) < 66 {
			ops := unit.GetTracer()
			if len(ops) > 0 {
				latestTr := ops[len(ops)-1]
				if strings.Compare(latestTr.GetOps(), "DIV") == 0 {
					// sload操作取得变量的值
					offset := latestTr.val
					bits := 4 * (len(x.Hex()) - 2)
					unit.SetOffset(offset)
					unit.SetBits(bits)
					unit.UpdateStorageValue(*y)
					unit.ClearTracer()
					// 生成新的tmp state进行记录 (防止跟踪丢失)
					fragment, exist := scope.TmpState.GetFragment(scope.Contract.Address(), unit.GetSlot())
					if !exist {
						fragment = scope.TmpState.InsertUnit(scope.Contract.Address(), unit.GetSlot())
					}
					fragment.GenerateVar(unit)
					unit.SetValue(*y)
					return nil, nil
				}
			}
		} else {
			// sstore操作取当前slot除了更新部分剩余的值
			bits := normalUnit.GetBits()
			offset := normalUnit.GetOffset()
			if offset.Uint64() > 0 && bits < 256 {
				unit.SetBits(bits)
				unit.SetOffset(offset)
				// 生成新的tmp state进行记录 (防止跟踪丢失)
				fragment, exist := scope.TmpState.GetFragment(scope.Contract.Address(), unit.GetSlot())
				if !exist {
					fragment = scope.TmpState.InsertUnit(scope.Contract.Address(), unit.GetSlot())
				}
				fragment.GenerateVar(unit)
				unit.SetValue(*y)
				return nil, nil
			}
		}
	}
	merge(x, originVal, *y, "AND", xUnit, yUnit, scope, true)
	return nil, nil
}

func popOr(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Or(&x, y)
	merge(x, originVal, *y, "OR", xUnit, yUnit, scope, true)
	return nil, nil
}

func popXor(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Xor(&x, y)
	merge(x, originVal, *y, "XOR", xUnit, yUnit, scope, true)
	return nil, nil
}

func popByte(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	th, thUnit := scope.Stack.pop()
	val, valUnit := scope.Stack.peek()
	originVal := *val
	val.Byte(&th)
	merge(th, originVal, *val, "BYTE", thUnit, valUnit, scope, true)
	return nil, nil
}

func popAddmod(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	//x, y, z := scope.Stack.pop(), scope.Stack.pop(), scope.Stack.peek()
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Add(&x, y)
	merge(x, originVal, *y, "ADD", xUnit, yUnit, scope, true)

	y2, yUnit2 := scope.Stack.pop()
	z, zUnit := scope.Stack.peek()
	originVal2 := *z
	if z.IsZero() {
		z.Clear()
	} else {
		//z.AddMod(&x, &y, z)
		z.Mod(&y2, z)
	}
	merge(y2, originVal2, *z, "MOD", yUnit2, zUnit, scope, true)
	return nil, nil
}

func popMulmod(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	//x, y, z := scope.Stack.pop(), scope.Stack.pop(), scope.Stack.peek()
	x, xUnit := scope.Stack.pop()
	y, yUnit := scope.Stack.peek()
	originVal := *y
	y.Mul(&x, y)
	merge(x, originVal, *y, "MUL", xUnit, yUnit, scope, true)

	y2, yUnit2 := scope.Stack.pop()
	z, zUnit := scope.Stack.peek()
	originVal2 := *z
	//z.MulMod(&x, &y, z)
	z.Mod(&y2, z)
	merge(y2, originVal2, *z, "MOD", yUnit2, zUnit, scope, true)
	return nil, nil
}

// opSHL implements Shift Left
// The SHL instruction (shift left) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the left by arg1 number of bits.
func popSHL(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// Note, second operand is left in the stack; accumulate result into it, and no need to push it afterwards
	shift, shiftUnit := scope.Stack.pop()
	value, valueUnit := scope.Stack.peek()
	originVal := *value
	if shift.LtUint64(256) {
		value.Lsh(value, uint(shift.Uint64()))
	} else {
		value.Clear()
	}
	merge(shift, originVal, *value, "SHL", shiftUnit, valueUnit, scope, true)
	return nil, nil
}

// opSHR implements Logical Shift Right
// The SHR instruction (logical shift right) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the right by arg1 number of bits with zero fill.
func popSHR(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// Note, second operand is left in the stack; accumulate result into it, and no need to push it afterwards
	shift, shiftUnit := scope.Stack.pop()
	value, valueUnit := scope.Stack.peek()
	originVal := *value
	if shift.LtUint64(256) {
		value.Rsh(value, uint(shift.Uint64()))
	} else {
		value.Clear()
	}
	merge(shift, originVal, *value, "SHR", shiftUnit, valueUnit, scope, true)
	return nil, nil
}

// opSAR implements Arithmetic Shift Right
// The SAR instruction (arithmetic shift right) pops 2 values from the stack, first arg1 and then arg2,
// and pushes on the stack arg2 shifted to the right by arg1 number of bits with sign extension.
func popSAR(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	shift, shiftUnit := scope.Stack.pop()
	value, valueUnit := scope.Stack.peek()
	originVal := *value
	if shift.GtUint64(256) {
		if value.Sign() >= 0 {
			value.Clear()
		} else {
			// Max negative shift: all bits set
			value.SetAllOne()
		}
		merge(shift, originVal, *value, "SAR", shiftUnit, valueUnit, scope, true)
		return nil, nil
	}
	n := uint(shift.Uint64())
	value.SRsh(value, n)
	merge(shift, originVal, *value, "SAR", shiftUnit, valueUnit, scope, true)
	return nil, nil
}

func popKeccak256(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	offset, offsetUnit := scope.Stack.pop()
	size, sizeUnit := scope.Stack.peek()
	originVal := *size
	// TODO: identify state variables in memory
	data := scope.Memory.GetPtr(int64(offset.Uint64()), int64(size.Uint64()))

	if interpreter.hasher == nil {
		interpreter.hasher = crypto.NewKeccakState()
	} else {
		interpreter.hasher.Reset()
	}
	interpreter.hasher.Write(data)
	interpreter.hasher.Read(interpreter.hasherBuf[:])

	evm := interpreter.evm
	if evm.Config.EnablePreimageRecording {
		evm.StateDB.AddPreimage(interpreter.hasherBuf, data)
	}

	size.SetBytes(interpreter.hasherBuf[:])
	merge(offset, originVal, *size, "KECCAK256", offsetUnit, sizeUnit, scope, false)
	return nil, nil
}

func popAddress(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetBytes(scope.Contract.Address().Bytes()))
	return nil, nil
}

func popBalance(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	slot, _ := scope.Stack.peek()
	address := common.Address(slot.Bytes20())
	slot.SetFromBig(interpreter.evm.StateDB.GetBalance(address))
	s := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *s, uint256.Int{}, *slot, uint256.Int{}, "BALANCE", address)
	// cache the read operation into the pre-execution table
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	ret.CacheReadSet(scope.Contract.Address(), nil)
	return nil, nil
}

func popOrigin(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetBytes(interpreter.evm.Origin.Bytes()))
	return nil, nil
}

func popCaller(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetBytes(scope.Contract.Caller().Bytes()))
	return nil, nil
}

func popCallValue(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v, _ := uint256.FromBig(scope.Contract.value)
	scope.Stack.push(v)
	return nil, nil
}

func popCallDataLoad(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	x, _ := scope.Stack.peek()
	if offset, overflow := x.Uint64WithOverflow(); !overflow {
		data := getData(scope.Contract.Input, offset, 32)
		x.SetBytes(data)
		// set the function signature
		sig := x.Hex()[:]
		scope.Signature = sig
		off := uint256.Int{}
		off.SetUint64(offset)
		scope.Stack.updateUnit(INPUT, uint256.Int{}, off, *x, uint256.Int{}, "nil", common.Address{})
	} else {
		x.Clear()
		off := uint256.Int{}
		off.SetUint64(0)
		scope.Stack.updateUnit(INPUT, uint256.Int{}, off, *x, uint256.Int{}, "nil", common.Address{})
	}
	return nil, nil
}

func popCallDataSize(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetUint64(uint64(len(scope.Contract.Input))))
	return nil, nil
}

func popCallDataCopy(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	var (
		memOffset, _  = scope.Stack.pop()
		dataOffset, _ = scope.Stack.pop()
		length, _     = scope.Stack.pop()
	)
	dataOffset64, overflow := dataOffset.Uint64WithOverflow()
	if overflow {
		dataOffset64 = 0xffffffffffffffff
	}
	// These values are checked for overflow during gas cost calculation
	memOffset64 := memOffset.Uint64()
	length64 := length.Uint64()
	scope.Memory.Set(memOffset64, length64, getData(scope.Contract.Input, dataOffset64, length64))

	return nil, nil
}

func popReturnDataSize(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetUint64(uint64(len(interpreter.returnData))))
	return nil, nil
}

func popReturnDataCopy(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	var (
		memOffset, _  = scope.Stack.pop()
		dataOffset, _ = scope.Stack.pop()
		length, _     = scope.Stack.pop()
	)

	offset64, overflow := dataOffset.Uint64WithOverflow()
	if overflow {
		return nil, ErrReturnDataOutOfBounds
	}
	// we can reuse dataOffset now (aliasing it for clarity)
	var end = dataOffset
	end.Add(&dataOffset, &length)
	end64, overflow := end.Uint64WithOverflow()
	if overflow || uint64(len(interpreter.returnData)) < end64 {
		return nil, ErrReturnDataOutOfBounds
	}
	scope.Memory.Set(memOffset.Uint64(), length.Uint64(), interpreter.returnData[offset64:end64])
	return nil, nil
}

func popExtCodeSize(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	slot, slotUnit := scope.Stack.peek()
	slot.SetUint64(uint64(interpreter.evm.StateDB.GetCodeSize(slot.Bytes20())))
	slotUnit.SetValue(*slot)
	// cache the read operation into the pre-execution table
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	ret.CacheReadSet(scope.Contract.Address(), nil)
	return nil, nil
}

func popCodeSize(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	l := new(uint256.Int)
	l.SetUint64(uint64(len(scope.Contract.Code)))
	scope.Stack.push(l)
	return nil, nil
}

func popCodeCopy(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	var (
		memOffset, _  = scope.Stack.pop()
		codeOffset, _ = scope.Stack.pop()
		length, _     = scope.Stack.pop()
	)
	uint64CodeOffset, overflow := codeOffset.Uint64WithOverflow()
	if overflow {
		uint64CodeOffset = 0xffffffffffffffff
	}
	codeCopy := getData(scope.Contract.Code, uint64CodeOffset, length.Uint64())
	scope.Memory.Set(memOffset.Uint64(), length.Uint64(), codeCopy)

	return nil, nil
}

func popExtCodeCopy(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	var (
		stack         = scope.Stack
		a, _          = stack.pop()
		memOffset, _  = stack.pop()
		codeOffset, _ = stack.pop()
		length, _     = stack.pop()
	)
	uint64CodeOffset, overflow := codeOffset.Uint64WithOverflow()
	if overflow {
		uint64CodeOffset = 0xffffffffffffffff
	}
	addr := common.Address(a.Bytes20())
	codeCopy := getData(interpreter.evm.StateDB.GetCode(addr), uint64CodeOffset, length.Uint64())
	scope.Memory.Set(memOffset.Uint64(), length.Uint64(), codeCopy)
	// cache the read operation into the pre-execution table
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	ret.CacheReadSet(scope.Contract.Address(), nil)

	return nil, nil
}

// opExtCodeHash returns the code hash of a specified account.
// There are several cases when the function is called, while we can relay everything
// to `state.GetCodeHash` function to ensure the correctness.
//
//	(1) Caller tries to get the code hash of a normal contract account, state
//
// should return the relative code hash and set it as the result.
//
//	(2) Caller tries to get the code hash of a non-existent account, state should
//
// return common.Hash{} and zero will be set as the result.
//
//	(3) Caller tries to get the code hash for an account without contract code,
//
// state should return emptyCodeHash(0xc5d246...) as the result.
//
//	(4) Caller tries to get the code hash of a precompiled account, the result
//
// should be zero or emptyCodeHash.
//
// It is worth noting that in order to avoid unnecessary create and clean,
// all precompile accounts on mainnet have been transferred 1 wei, so the return
// here should be emptyCodeHash.
// If the precompile account is not transferred any amount on a private or
// customized chain, the return value will be zero.
//
//	(5) Caller tries to get the code hash for an account which is marked as suicided
//
// in the current transaction, the code hash of this account should be returned.
//
//	(6) Caller tries to get the code hash for an account which is marked as deleted,
//
// this account should be regarded as a non-existent account and zero should be returned.
func popExtCodeHash(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	slot, slotUnit := scope.Stack.peek()
	address := common.Address(slot.Bytes20())
	if interpreter.evm.StateDB.Empty(address) {
		slot.Clear()
	} else {
		slot.SetBytes(interpreter.evm.StateDB.GetCodeHash(address).Bytes())
	}
	slotUnit.SetValue(*slot)
	// cache the read operation into the pre-execution table
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	ret.CacheReadSet(scope.Contract.Address(), nil)
	return nil, nil
}

// We regard all variables relevant to blockchain environments as state units, e.g., gasPrice, blockhash, coinbase, timestamp, etc.
// We use the storage slot of '0' as the storage location for these variables.
// It is safe since we will not interact with StateDB when encountering these variables.
func popGasprice(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v, _ := uint256.FromBig(interpreter.evm.GasPrice)
	scope.Stack.push(v)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *v, uint256.Int{}, "GASPRICE", common.Address{})
	return nil, nil
}

func popBlockhash(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	num, numUnit := scope.Stack.peek()
	originNum := *num
	num64, overflow := num.Uint64WithOverflow()
	if overflow {
		num.Clear()
		return nil, nil
	}
	var upper, lower uint64
	upper = interpreter.evm.Context.BlockNumber.Uint64()
	if upper < 257 {
		lower = 0
	} else {
		lower = upper - 256
	}
	if num64 >= lower && num64 < upper {
		num.SetBytes(interpreter.evm.Context.GetHash(num64).Bytes())
	} else {
		num.Clear()
	}
	numUnit.SetValue(*num)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *num, originNum, "BLOCKHASH", common.Address{})
	return nil, nil
}

func popCoinbase(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v := new(uint256.Int).SetBytes(interpreter.evm.Context.Coinbase.Bytes())
	scope.Stack.push(v)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *v, uint256.Int{}, "COINBASE", common.Address{})
	return nil, nil
}

func popTimestamp(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v := new(uint256.Int).SetUint64(interpreter.evm.Context.Time)
	scope.Stack.push(v)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *v, uint256.Int{}, "TIMESTAMP", common.Address{})
	return nil, nil
}

func popNumber(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v, _ := uint256.FromBig(interpreter.evm.Context.BlockNumber)
	scope.Stack.push(v)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *v, uint256.Int{}, "NUMBER", common.Address{})
	return nil, nil
}

func popDifficulty(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v, _ := uint256.FromBig(interpreter.evm.Context.Difficulty)
	scope.Stack.push(v)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *v, uint256.Int{}, "DIFFICULTY", common.Address{})
	return nil, nil
}

func popRandom(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v := new(uint256.Int).SetBytes(interpreter.evm.Context.Random.Bytes())
	scope.Stack.push(v)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *v, uint256.Int{}, "RANDOM", common.Address{})
	return nil, nil
}

func popGasLimit(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v := new(uint256.Int).SetUint64(interpreter.evm.Context.GasLimit)
	scope.Stack.push(v)
	slotInt := new(uint256.Int).SetUint64(0)
	scope.Stack.updateUnit(STATE, *slotInt, uint256.Int{}, *v, uint256.Int{}, "GASLIMIT", common.Address{})
	return nil, nil
}

func popPop(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.pop()
	return nil, nil
}

func popMload(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	v, unit := scope.Stack.peek()
	offset := int64(v.Uint64())
	v.SetBytes(scope.Memory.GetPtr(offset, 32))
	unit.SetValue(*v)
	return nil, nil
}

func popMstore(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// pop value of the stack
	mStart, _ := scope.Stack.pop()
	val, _ := scope.Stack.pop()
	scope.Memory.Set32(mStart.Uint64(), &val)
	return nil, nil
}

func popMstore8(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	off, _ := scope.Stack.pop()
	val, _ := scope.Stack.pop()
	scope.Memory.store[off.Uint64()] = byte(val.Uint64())
	return nil, nil
}

func popSload(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	loc, _ := scope.Stack.peek()
	tmpLoc := *loc
	hash := common.Hash(loc.Bytes32())
	val := interpreter.evm.StateDB.GetState(scope.Contract.Address(), hash)
	loc.SetBytes(val.Bytes())
	offset := uint256.Int{}
	offset.SetUint64(0)
	scope.Stack.updateUnit(STATE, tmpLoc, offset, *loc, uint256.Int{}, "nil", common.Address{})
	// insert the state unit into the temporary state space during a pre-execution
	scope.TmpState.InsertUnit(scope.Contract.Address(), tmpLoc)
	// cache the read operation into the pre-execution table
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	ret.CacheReadSet(scope.Contract.Address(), &hash)
	if interpreter.fullStorage {
		tip := interpreter.evm.TxContext.GasTip
		_, err := interpreter.evm.MVCache.SetStorageForWrite(scope.Contract.Address(), tmpLoc,
			*loc, txID, tip)
		if err != nil {
			return nil, err
		}
		_, err2 := interpreter.evm.MVCache.SetCompactedStorageForWrite(scope.Contract.Address(), tmpLoc,
			*loc, offset, txID, tip)
		if err2 != nil {
			return nil, err
		}
	}
	return nil, nil
}

func popSstore(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// sstore会有两种情况:
	// 1. loc位置对应的变量存在compact存储:
	// 赋值的val的unit一定为stateUnit，且stateUnit存储了此loc位置的相关信息，此时val（为整个slot的值）需要通过掩码操作得到实际更新的变量值
	// 2. loc位置对应的变量满足256位，未发生compact:
	// 无论赋值的val的unit为stateUnit还是其他，此时val就是实际更新的变量值
	if interpreter.readOnly {
		return nil, ErrWriteProtection
	}
	loc, locUnit := scope.Stack.pop()
	val, valUnit := scope.Stack.pop()
	locHash := common.Hash(loc.Bytes32())
	// 模拟执行时不会向statedb中写入更新值
	//interpreter.evm.StateDB.SetState(scope.Contract.Address(),
	//	loc.Bytes32(), val.Bytes32())
	var (
		updatedVal uint256.Int
		offset     uint256.Int
		isCompact  bool
		signExtend bool
		repair     bool
		err        error
	)
	txID := interpreter.evm.TxContext.ID
	ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	// 通过tmpState获取到slot位置上的变量信息
	fragment, ok := scope.TmpState.GetFragment(scope.Contract.Address(), loc)
	if ok {
		// sload之后再赋值
		stateUnit, ok2 := valUnit.(*StateUnit)
		// 发生了storage compaction
		if ok2 && fragment.GetCompact() {
			offset = stateUnit.GetOffset()
			bits := stateUnit.GetBits()
			stateVar := fragment.GetVar(offset)
			// 取真实存储值 (operation最后保存的操作是与存储值or)
			ops := stateUnit.GetTracer()
			if offset.Uint64() > 0 && bits < 256 {
				isCompact = true
				if len(ops) > 0 {
					attaching := ops[len(ops)-1].GetAttaching()
					if opUnit, ok3 := attaching.(*StateUnit); ok3 {
						signExtend = opUnit.GetSignExtend()
						if signExtend {
							stateUnit.SetSignExtend()
						}
					}
				}
				updatedVal = fetchStorageVal(val, offset, bits, signExtend)
			} else {
				isCompact = false
				updatedVal = val
			}
			stateVar.SetUpdatedVal(updatedVal)
		} else {
			isCompact = false
			offset.SetUint64(0)
			updatedVal = val
			stateVar := fragment.GetVar(offset)
			stateVar.SetUpdatedVal(updatedVal)
		}
	} else {
		// 直接赋值
		isCompact = false
		offset.SetUint64(0)
		updatedVal = val
		fragment = NewFragment(loc)
		fragmentMap := scope.TmpState.Space[scope.Contract.Address()]
		fragmentMap[common.Hash(loc.Bytes32())] = fragment
		stateVar := fragment.GetVar(offset)
		stateVar.SetUpdatedVal(updatedVal)
	}
	// store the sstore info into the pre-execution table
	ret.CacheSStoreInfo(scope.Contract.Address(), updatedVal, locUnit.Copy(), valUnit.Copy(), isCompact)

	// 只有已经记录的与分支相关的变量才会被多版本缓存存储
	if interpreter.fullStorage || interpreter.evm.VarTable.VarExist(scope.Contract.Address(), loc.Bytes32(), offset) {
		tip := interpreter.evm.TxContext.GasTip
		if isCompact {
			repair, err = interpreter.evm.MVCache.SetCompactedStorageForWrite(scope.Contract.Address(), loc,
				updatedVal, offset, txID, tip)
		} else {
			repair, err = interpreter.evm.MVCache.SetStorageForWrite(scope.Contract.Address(), loc,
				updatedVal, txID, tip)
		}
		if err != nil {
			return nil, err
		}
		interpreter.repair = repair
		if interpreter.repair {
			interpreter.InsertRepairedLoc(scope.Contract.Address(), loc.Bytes32(), offset)
		}
	}
	// store the write operation into the pre-execution table
	ret.CacheWriteSet(scope.Contract.Address(), &locHash)
	return nil, nil
}

func popJump(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	if atomic.LoadInt32(&interpreter.evm.abort) != 0 {
		return nil, errStopToken
	}
	pos, _ := scope.Stack.pop()
	if !scope.Contract.validJumpdest(&pos) {
		return nil, ErrInvalidJump
	}
	*pc = pos.Uint64() - 1 // pc will be increased by the interpreter loop
	return nil, nil
}

func popJumpi(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	if atomic.LoadInt32(&interpreter.evm.abort) != 0 {
		return nil, errStopToken
	}
	pos, _ := scope.Stack.pop()
	cond, condUnit := scope.Stack.pop()

	if sUnit, ok := condUnit.(*StateUnit); ok {
		ops := sUnit.GetTracer()
		if len(ops) > 0 {
			latestTr := ops[len(ops)-1]
			// 在智能合约中sub用来判断两个数是否相等
			if strings.Compare(latestTr.GetOps(), "SUB") == 0 {
				opUnit := latestTr.GetAttaching()
				firstVal, secondVal, res, err := branchRecord(*pc, pos.Uint64(), interpreter, scope, sUnit, opUnit, "EQ")
				if err != nil {
					return nil, err
				}
				txID := interpreter.evm.TxContext.ID
				ret, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
				switch res {
				case taken:
					ret.UpdateDirection(1)
					return nil, nil
				case notTaken:
					if !scope.Contract.validJumpdest(&pos) {
						return nil, ErrInvalidJump
					}
					*pc = pos.Uint64() - 1 // pc will be increased by the interpreter loop
					ret.UpdateDirection(0)
					return nil, nil
				case uncertain:
					firstVal.Sub(&firstVal, &secondVal)
					if !firstVal.IsZero() {
						if !scope.Contract.validJumpdest(&pos) {
							return nil, ErrInvalidJump
						}
						*pc = pos.Uint64() - 1 // pc will be increased by the interpreter loop
						ret.UpdateDirection(0)
					} else {
						ret.UpdateDirection(1)
					}
					return nil, nil
				}
			}
		}
	}

	if !cond.IsZero() {
		if !scope.Contract.validJumpdest(&pos) {
			return nil, ErrInvalidJump
		}
		*pc = pos.Uint64() - 1 // pc will be increased by the interpreter loop
	}
	return nil, nil
}

func popJumpdest(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	return nil, nil
}

func popPc(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetUint64(*pc))
	return nil, nil
}

func popMsize(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetUint64(uint64(scope.Memory.Len())))
	return nil, nil
}

func popGas(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	scope.Stack.push(new(uint256.Int).SetUint64(scope.Contract.Gas))
	return nil, nil
}

func popCreate(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	if interpreter.readOnly {
		return nil, ErrWriteProtection
	}
	var (
		value, _  = scope.Stack.pop()
		offset, _ = scope.Stack.pop()
		size, _   = scope.Stack.pop()
		input     = scope.Memory.GetCopy(int64(offset.Uint64()), int64(size.Uint64()))
		gas       = scope.Contract.Gas
	)
	if interpreter.evm.chainRules.IsEIP150 {
		gas -= gas / 64
	}
	// reuse size int for stackvalue
	stackvalue := size

	scope.Contract.UseGas(gas)
	//TODO: use uint256.Int instead of converting with toBig()
	var bigVal = big0
	if !value.IsZero() {
		bigVal = value.ToBig()
	}

	res, addr, returnGas, suberr := interpreter.evm.Create(scope.Contract, input, gas, bigVal)
	// Push item on the stack based on the returned error. If the ruleset is
	// homestead we must check for CodeStoreOutOfGasError (homestead only
	// rule) and treat as an error, if the ruleset is frontier we must
	// ignore this error and pretend the operation was successful.
	if interpreter.evm.chainRules.IsHomestead && suberr == ErrCodeStoreOutOfGas {
		stackvalue.Clear()
	} else if suberr != nil && suberr != ErrCodeStoreOutOfGas {
		stackvalue.Clear()
	} else {
		stackvalue.SetBytes(addr.Bytes())
	}
	scope.Stack.push(&stackvalue)
	scope.Contract.Gas += returnGas

	if suberr == ErrExecutionReverted {
		interpreter.returnData = res // set REVERT data to return data buffer
		return res, nil
	}
	interpreter.returnData = nil // clear dirty return data buffer
	return nil, nil
}

func popCreate2(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	if interpreter.readOnly {
		return nil, ErrWriteProtection
	}
	var (
		endowment, _ = scope.Stack.pop()
		offset, _    = scope.Stack.pop()
		size, _      = scope.Stack.pop()
		salt, _      = scope.Stack.pop()
		input        = scope.Memory.GetCopy(int64(offset.Uint64()), int64(size.Uint64()))
		gas          = scope.Contract.Gas
	)

	// Apply EIP150
	gas -= gas / 64
	scope.Contract.UseGas(gas)
	// reuse size int for stackvalue
	stackvalue := size
	//TODO: use uint256.Int instead of converting with toBig()
	bigEndowment := big0
	if !endowment.IsZero() {
		bigEndowment = endowment.ToBig()
	}
	res, addr, returnGas, suberr := interpreter.evm.Create2(scope.Contract, input, gas,
		bigEndowment, &salt)
	// Push item on the stack based on the returned error.
	if suberr != nil {
		stackvalue.Clear()
	} else {
		stackvalue.SetBytes(addr.Bytes())
	}
	scope.Stack.push(&stackvalue)
	scope.Contract.Gas += returnGas

	if suberr == ErrExecutionReverted {
		interpreter.returnData = res // set REVERT data to return data buffer
		return res, nil
	}
	interpreter.returnData = nil // clear dirty return data buffer
	return nil, nil
}

func popCall(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// cache the current caller snapshot before internal call
	stack := scope.Stack
	txID := interpreter.evm.TxContext.ID
	res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	if interpreter.checkpoint {
		res.CacheSnapshot(stack, scope.Memory, *pc, 0, scope.Contract, interpreter.evm.depth)
	}
	// Pop gas. The actual gas in interpreter.evm.callGasTemp.
	// We can use this as a temporary value
	temp, _ := stack.pop()
	gas := interpreter.evm.callGasTemp
	// Pop other call parameters.
	addr, _ := stack.pop()
	value, _ := stack.pop()
	inOffset, _ := stack.pop()
	inSize, _ := stack.pop()
	retOffset, _ := stack.pop()
	retSize, _ := stack.pop()

	toAddr := common.Address(addr.Bytes20())
	// Get the arguments from the memory.
	args := scope.Memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	if interpreter.readOnly && !value.IsZero() {
		return nil, ErrWriteProtection
	}
	var bigVal = big0
	//TODO: use uint256.Int instead of converting with toBig()
	// By using big0 here, we save an alloc for the most common case (non-ether-transferring contract calls),
	// but it would make more sense to extend the usage of uint256.Int
	if !value.IsZero() {
		gas += params.CallStipend
		bigVal = value.ToBig()
	}

	ret, returnGas, err := interpreter.evm.Call(scope.Contract, toAddr, args, gas, bigVal)

	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		ret = common.CopyBytes(ret)
		scope.Memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	scope.Contract.Gas += returnGas

	interpreter.returnData = ret
	return ret, nil
}

func popCallCode(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// cache the current caller snapshot before internal call
	stack := scope.Stack
	txID := interpreter.evm.TxContext.ID
	res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	if interpreter.checkpoint {
		res.CacheSnapshot(stack, scope.Memory, *pc, 0, scope.Contract, interpreter.evm.depth)
	}
	// Pop gas. The actual gas is in interpreter.evm.callGasTemp.
	// We use it as a temporary value
	temp, _ := stack.pop()
	gas := interpreter.evm.callGasTemp
	// Pop other call parameters.
	addr, _ := stack.pop()
	value, _ := stack.pop()
	inOffset, _ := stack.pop()
	inSize, _ := stack.pop()
	retOffset, _ := stack.pop()
	retSize, _ := stack.pop()

	toAddr := common.Address(addr.Bytes20())
	// Get arguments from the memory.
	args := scope.Memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	//TODO: use uint256.Int instead of converting with toBig()
	var bigVal = big0
	if !value.IsZero() {
		gas += params.CallStipend
		bigVal = value.ToBig()
	}

	ret, returnGas, err := interpreter.evm.CallCode(scope.Contract, toAddr, args, gas, bigVal)
	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		ret = common.CopyBytes(ret)
		scope.Memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	scope.Contract.Gas += returnGas

	interpreter.returnData = ret
	return ret, nil
}

func popDelegateCall(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// cache the current caller snapshot before internal call
	stack := scope.Stack
	txID := interpreter.evm.TxContext.ID
	res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	if interpreter.checkpoint {
		res.CacheSnapshot(stack, scope.Memory, *pc, 0, scope.Contract, interpreter.evm.depth)
	}
	// Pop gas. The actual gas is in interpreter.evm.callGasTemp.
	// We use it as a temporary value
	temp, _ := stack.pop()
	gas := interpreter.evm.callGasTemp
	// Pop other call parameters.
	addr, _ := stack.pop()
	inOffset, _ := stack.pop()
	inSize, _ := stack.pop()
	retOffset, _ := stack.pop()
	retSize, _ := stack.pop()

	toAddr := common.Address(addr.Bytes20())
	// Get arguments from the memory.
	args := scope.Memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	ret, returnGas, err := interpreter.evm.DelegateCall(scope.Contract, toAddr, args, gas)
	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		ret = common.CopyBytes(ret)
		scope.Memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	scope.Contract.Gas += returnGas

	interpreter.returnData = ret
	return ret, nil
}

func popStaticCall(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// cache the current caller snapshot before internal call
	stack := scope.Stack
	txID := interpreter.evm.TxContext.ID
	res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	if interpreter.checkpoint {
		res.CacheSnapshot(stack, scope.Memory, *pc, 0, scope.Contract, interpreter.evm.depth)
	}
	// Pop gas. The actual gas is in interpreter.evm.callGasTemp.
	// We use it as a temporary value
	temp, _ := stack.pop()
	gas := interpreter.evm.callGasTemp
	// Pop other call parameters.
	addr, _ := stack.pop()
	inOffset, _ := stack.pop()
	inSize, _ := stack.pop()
	retOffset, _ := stack.pop()
	retSize, _ := stack.pop()

	toAddr := common.Address(addr.Bytes20())
	// Get arguments from the memory.
	args := scope.Memory.GetPtr(int64(inOffset.Uint64()), int64(inSize.Uint64()))

	ret, returnGas, err := interpreter.evm.StaticCall(scope.Contract, toAddr, args, gas)
	if err != nil {
		temp.Clear()
	} else {
		temp.SetOne()
	}
	stack.push(&temp)
	if err == nil || err == ErrExecutionReverted {
		ret = common.CopyBytes(ret)
		scope.Memory.Set(retOffset.Uint64(), retSize.Uint64(), ret)
	}
	scope.Contract.Gas += returnGas

	interpreter.returnData = ret
	return ret, nil
}

func popReturn(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	offset, _ := scope.Stack.pop()
	size, _ := scope.Stack.pop()
	ret := scope.Memory.GetPtr(int64(offset.Uint64()), int64(size.Uint64()))
	// cache the result of execution and the relevant read/write set
	if interpreter.evm.depth == 1 {
		txID := interpreter.evm.TxContext.ID
		res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
		res.GenerateFinalSnapshot(ret, scope.Contract.Gas, errStopToken)
	}
	return ret, errStopToken
}

func popRevert(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	offset, _ := scope.Stack.pop()
	size, _ := scope.Stack.pop()
	ret := scope.Memory.GetPtr(int64(offset.Uint64()), int64(size.Uint64()))
	interpreter.returnData = ret
	// cache the result of execution and the relevant read/write set
	if interpreter.evm.depth == 1 {
		txID := interpreter.evm.TxContext.ID
		res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
		res.GenerateFinalSnapshot(ret, scope.Contract.Gas, ErrExecutionReverted)
	}
	return ret, ErrExecutionReverted
}

func popUndefined(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	err := &ErrInvalidOpCode{opcode: OpCode(scope.Contract.Code[*pc])}
	// cache the result of execution and the relevant read/write set
	if interpreter.evm.depth == 1 {
		txID := interpreter.evm.TxContext.ID
		res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
		res.GenerateFinalSnapshot(nil, scope.Contract.Gas, err)
	}
	return nil, err
}

func popStop(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	// cache the result of execution and the relevant read/write set
	if interpreter.evm.depth == 1 {
		txID := interpreter.evm.TxContext.ID
		res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
		res.GenerateFinalSnapshot(nil, scope.Contract.Gas, errStopToken)
	}
	return nil, errStopToken
}

func popSelfdestruct(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	if interpreter.readOnly {
		return nil, ErrWriteProtection
	}
	beneficiary, _ := scope.Stack.pop()
	balance := interpreter.evm.StateDB.GetBalance(scope.Contract.Address())
	interpreter.evm.StateDB.AddBalance(beneficiary.Bytes20(), balance)
	interpreter.evm.StateDB.Suicide(scope.Contract.Address())
	if interpreter.evm.Config.Debug {
		interpreter.evm.Config.Tracer.CaptureEnter(SELFDESTRUCT, scope.Contract.Address(), beneficiary.Bytes20(), []byte{}, 0, balance)
		interpreter.evm.Config.Tracer.CaptureExit([]byte{}, 0, nil)
	}

	// cache the result of execution and the relevant read/write set
	txID := interpreter.evm.TxContext.ID
	res, _ := interpreter.evm.PreExecutionTable.GetResult(txID)
	res.CacheReadSet(scope.Contract.Address(), nil)
	res.CacheWriteSet(scope.Contract.Address(), nil)
	if interpreter.evm.depth == 1 {
		res.GenerateFinalSnapshot(nil, scope.Contract.Gas, errStopToken)
	}
	return nil, errStopToken
}

// make log instruction function
func makeLog2(size int) executionFunc {
	return func(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
		if interpreter.readOnly {
			return nil, ErrWriteProtection
		}
		topics := make([]common.Hash, size)
		stack := scope.Stack
		mStart, _ := stack.pop()
		mSize, _ := stack.pop()
		for i := 0; i < size; i++ {
			addr, _ := stack.pop()
			topics[i] = addr.Bytes32()
		}

		d := scope.Memory.GetCopy(int64(mStart.Uint64()), int64(mSize.Uint64()))
		interpreter.evm.StateDB.AddLog(&types.Log{
			Address: scope.Contract.Address(),
			Topics:  topics,
			Data:    d,
			// This is a non-consensus field, but assigned here because
			// core/state doesn't know the current block number.
			BlockNumber: interpreter.evm.Context.BlockNumber.Uint64(),
		})
		return nil, nil
	}
}

// opPush1 is a specialized version of pushN
func popPush1(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
	var (
		codeLen = uint64(len(scope.Contract.Code))
		integer = new(uint256.Int)
	)
	*pc += 1
	if *pc < codeLen {
		scope.Stack.push(integer.SetUint64(uint64(scope.Contract.Code[*pc])))
	} else {
		scope.Stack.push(integer.Clear())
	}
	return nil, nil
}

// make push instruction function
func makePush2(size uint64, pushByteSize int) executionFunc {
	return func(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
		codeLen := len(scope.Contract.Code)

		startMin := codeLen
		if int(*pc+1) < startMin {
			startMin = int(*pc + 1)
		}

		endMin := codeLen
		if startMin+pushByteSize < endMin {
			endMin = startMin + pushByteSize
		}

		integer := new(uint256.Int)
		scope.Stack.push(integer.SetBytes(common.RightPadBytes(
			scope.Contract.Code[startMin:endMin], pushByteSize)))

		*pc += size
		return nil, nil
	}
}

// make dup instruction function
func makeDup2(size int64) executionFunc {
	return func(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
		scope.Stack.dup(int(size))
		return nil, nil
	}
}

// make swap instruction function
func makeSwap2(size int64) executionFunc {
	// switch n + 1 otherwise n would be swapped with n
	size++
	return func(pc *uint64, interpreter *EVMInterpreter, scope *ScopeContext) ([]byte, error) {
		scope.Stack.swap(int(size))
		return nil, nil
	}
}
