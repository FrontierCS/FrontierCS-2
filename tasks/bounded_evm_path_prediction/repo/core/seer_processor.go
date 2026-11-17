package core

import (
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/holiman/uint256"
	"math/big"
	"math/rand"
	"seerEVM/core/state"
	"seerEVM/core/types"
	"seerEVM/core/vm"
	"seerEVM/dependencyGraph"
	"seerEVM/minHeap"
	"sort"
	"strconv"
	"strings"
	"time"
)

func applyNormalTransaction(msg *Message, gp *GasPool, statedb *state.SeerTransaction, blockNumber *big.Int, blockHash common.Hash, tx *types.Transaction, usedGas *uint64, evm *vm.EVM) (*types.Receipt, error) {
	// Create a new context to be used in the EVM environment.
	txContext := NewEVMTxContext(msg, tx.Hash(), &big.Int{})
	evm.Reset(txContext, statedb)

	// Apply the transaction to the current state (included in the env).
	// 真正执行并更新stmTxStateDB的stateObjects，在官方的库中，后续调用Finalise才会将这些改动应用到statedb中
	result, err := ApplyMessage(evm, msg, gp)
	if err != nil {
		return nil, err
	}

	// Update the state with pending changes.
	var root []byte
	//if config.IsByzantium(blockNumber) {
	//	stmStateDB.Finalise(true, statedb.Index)
	//} else {
	//	root = stmStateDB.IntermediateRoot(config.IsEIP158(blockNumber), statedb.Index).Bytes()
	//}
	*usedGas += result.UsedGas

	// Create a new receipt for the transaction, storing the intermediate root and gas used
	// by the tx.
	receipt := &types.Receipt{Type: tx.Type(), PostState: root, CumulativeGasUsed: *usedGas}
	if result.Failed() {
		receipt.Status = types.ReceiptStatusFailed
	} else {
		receipt.Status = types.ReceiptStatusSuccessful
	}
	receipt.TxHash = tx.Hash()
	receipt.GasUsed = result.UsedGas

	// If the transaction created a contract, store the creation address in the receipt.
	if msg.To == nil {
		receipt.ContractAddress = crypto.CreateAddress(evm.TxContext.Origin, tx.Nonce())
	}

	// Set the receipt logs and create the bloom filter.
	receipt.Logs = statedb.GetLogs(tx.Hash(), blockNumber.Uint64(), blockHash)
	receipt.Bloom = types.CreateBloom(types.Receipts{receipt})
	receipt.BlockHash = blockHash
	receipt.BlockNumber = blockNumber
	receipt.TransactionIndex = uint(statedb.TxIndex())
	return receipt, err
}

// applySeerTransaction pre-executes a transaction to cache fast path info and conduct pre-execution repair if necessary
func applySeerTransaction(msg *Message, gp *GasPool, evm *vm.EVM, txSet map[common.Hash]*types.Transaction, tipMap map[common.Hash]*big.Int, enableRepair, enablePredictor, storeCheckpoint, fullStorage bool) error {
	if _, err := ApplyMessage(evm, msg, gp); err != nil {
		return err
	}

	if enableRepair {
		if evm.Interpreter().GetRepair() {
			var repairedTXs []common.Hash
			repairedLoc := evm.Interpreter().GetRepairedLoc()
			for addr, slotMap := range repairedLoc {
				for slot, offsetMap := range slotMap {
					for str := range offsetMap {
						var slotInt, offset uint256.Int
						slotInt.SetBytes(slot.Bytes())
						offsetNum, _ := strconv.Atoi(str[2:])
						offset.SetUint64(uint64(offsetNum))
						txs, err2 := evm.MVCache.GetRepairTXs(addr, slotInt, offset)
						if err2 != nil {
							return err2
						}
						repairedTXs = append(repairedTXs, txs...)
					}
				}
			}
			txs := sortTXs(repairedTXs, txSet, tipMap)
			fmt.Println("repair", len(repairedTXs), len(txs))
			for _, t := range txs {
				ret, _ := evm.PreExecutionTable.GetResult(t.Hash())
				txContext := ret.GetTxContext()
				vmenv := vm.NewEVM2(evm.Context, txContext, evm.StateDB, evm.VarTable, evm.PreExecutionTable,
					evm.MVCache, evm.ChainConfig(), vm.Config{EnablePreimageRecording: false}, true, enablePredictor, storeCheckpoint, fullStorage, nil)
				vmenv.Interpreter().SetFastEnabled()
				if err3 := repair(txContext, vmenv, t, ret); err3 != nil {
					return err3
				}
			}
		}
	}
	return nil
}

// applyFastPath executes a transaction in a fast-path mode
func applyFastPath(msg *Message, tx *types.Transaction, evm *vm.EVM, result *vm.Result, gp *GasPool, blockNumber *big.Int, blockHash common.Hash, usedGas *uint64, thread *SeerThread, recorder *Recorder, enableRepair, enablePredictor, enableFast bool) (*types.Receipt, bool, error) {
	var (
		res            *ExecutionResult
		err            error
		isBreak        bool
		isContractCall bool
	)

	// directly executes contract creation txs and common transfer txs
	if tx.To() == nil || !IsContract(evm.StateDB, tx.To()) {
		res, err = ApplyMessage(evm, msg, gp)
		if err != nil {
			return nil, isContractCall, err
		}
	} else {
		s := time.Now()
		isContractCall = true
		brs := result.GetBranches()

		if recorder != nil && recorder.GetObserveIndicator() {
			if len(brs) > 4 {
				recorder.AppendComplicatedTx(tx.Hash())
			}
		}

		for _, br := range brs {
			if recorder != nil && !recorder.GetObserveIndicator() {
				complicatedTxs := recorder.GetComplicatedMap()
				if _, ok := complicatedTxs[tx.Hash()]; ok {
					thread.IncrementCTotal()
				} else {
					thread.IncrementNTotal()
				}
			}
			// update the sstore info
			sstores := br.GetSstoreInfo()
			for _, sstore := range sstores {
				if err = updateSstore(sstore, evm, false); err != nil {
					return nil, isContractCall, err
				}
			}

			isTaken, err := checkBranchInExecution(evm, tx, br)
			if err != nil {
				return nil, isContractCall, err
			}

			isAccurate := isTaken == br.GetBranchDirection()

			if !enableRepair && !enablePredictor {
				if isAccurate {
					source := rand.NewSource(time.Now().UnixNano())
					r := rand.New(source)
					if r.Intn(100) >= 99 {
						isAccurate = false
					}
				}
			}

			if !isAccurate {
				if recorder != nil && !recorder.GetObserveIndicator() {
					complicatedTxs := recorder.GetComplicatedMap()
					if _, ok := complicatedTxs[tx.Hash()]; ok {
						thread.IncrementCUnsatisfied()
					} else {
						thread.IncrementNUnsatisfied()
					}
				}
				if !enableFast {
					res, err = ApplyMessage(evm, msg, gp)
					if err != nil {
						return nil, isContractCall, err
					}
					isBreak = true
					break
				}
				// encounter inconsistent path, conduct fast-path repair
				callMap := make(map[int]*vm.Snapshot)
				stackElement := uint256.Int{}
				curSnapshots := br.GetSnapshots()
				callStack := result.GetCallStack()
				latestSnapshot := curSnapshots[len(curSnapshots)-1]
				if len(callStack) > 0 {
					// internal call exists
					for depth, sps := range callStack {
						// put the latest snapshot under each depth into the call map
						if depth == latestSnapshot.GetDepth() {
							callMap[depth] = latestSnapshot
							continue
						}
						callMap[depth] = sps[len(sps)-1]
					}
				} else {
					callMap[1] = latestSnapshot
				}

				// modify the branch info
				if isTaken {
					pc := latestSnapshot.GetPC()
					latestSnapshot.UpdatePC(pc + 1)
					br.DecideDirection(1)
					if br.GetJudgement() != "EQ" {
						stackElement.SetUint64(1)
						latestSnapshot.GetStack().UpdatePeek(stackElement)
					}
				} else {
					br.DecideDirection(0)
					if br.GetJudgement() == "EQ" {
						jumpPc := latestSnapshot.GetJumpPC()
						latestSnapshot.UpdatePC(jumpPc)
					} else {
						pc := latestSnapshot.GetPC()
						latestSnapshot.UpdatePC(pc + 1)
						stackElement.SetUint64(0)
						latestSnapshot.GetStack().UpdatePeek(stackElement)
					}
				}

				// recover execution from the initial snapshot
				evm.Interpreter().SetFastEnabled()
				evm.Interpreter().SetCallMap(callMap)
				gas := uint64(0)
				if _, ok := callMap[1]; ok {
					gas = callMap[1].GetContract().Gas
				} else {
					gas = callMap[2].GetContract().Gas
				}

				res, err = FastApplyMessage(evm, msg, gp, gas, nil)
				if err != nil {
					return nil, isContractCall, err
				}

				isBreak = true
				break
			}
		}
		// all the branches are satisfied, execute the snapshot
		if !isBreak {
			if recorder != nil && !recorder.GetObserveIndicator() {
				complicatedTxs := recorder.GetComplicatedMap()
				if _, ok := complicatedTxs[tx.Hash()]; ok {
					thread.IncrementCSatisfiedTxs()
				} else {
					thread.IncrementNSatisfiedTxs()
				}
			}
			finalSnapshot := result.GetFinalSnapshot()
			// in case that some contract exists without using the exist-relevant opcodes
			if finalSnapshot != nil {
				for _, sstore := range finalSnapshot.GetSstoreInfo() {
					if err = updateSstore(sstore, evm, false); err != nil {
						return nil, isContractCall, err
					}
				}
				res, err = FastApplyMessage(evm, msg, gp, 0, finalSnapshot)
				if err != nil {
					return nil, isContractCall, err
				}
			} else {
				res, err = ApplyMessage(evm, msg, gp)
				if err != nil {
					return nil, isContractCall, err
				}
			}
		}
		e := time.Since(s)
		if recorder != nil && recorder.GetObserveIndicator() {
			recorder.SeerRecord(tx, e.Microseconds())
		}
	}

	// Update the state with pending changes.
	var root []byte
	*usedGas += res.UsedGas

	// Create a new receipt for the transaction, storing the intermediate root and gas used
	// by the tx.
	receipt := &types.Receipt{Type: tx.Type(), PostState: root, CumulativeGasUsed: *usedGas}
	if res.Failed() {
		receipt.Status = types.ReceiptStatusFailed
	} else {
		receipt.Status = types.ReceiptStatusSuccessful
	}
	receipt.TxHash = tx.Hash()
	receipt.GasUsed = res.UsedGas

	// If the transaction created a contract, store the creation address in the receipt.
	if msg.To == nil {
		receipt.ContractAddress = crypto.CreateAddress(evm.TxContext.Origin, tx.Nonce())
	}

	// Set the receipt logs and create the bloom filter.
	//receipt.Logs = statedb.GetLogs(tx.Hash(), blockNumber.Uint64(), blockHash)
	receipt.Bloom = types.CreateBloom(types.Receipts{receipt})
	receipt.BlockHash = blockHash
	receipt.BlockNumber = blockNumber
	//receipt.TransactionIndex = uint(statedb.TxIndex())
	return receipt, isContractCall, nil
}

// repair conducts pre-execution repair
func repair(context vm.TxContext, evm *vm.EVM, tx *types.Transaction, result *vm.Result) error {
	brs := result.GetBranches()
	for i, br := range brs {
		// update the sstore info
		sstores := br.GetSstoreInfo()
		for _, sstore := range sstores {
			if err := updateSstore(sstore, evm, true); err != nil {
				return err
			}
		}

		isTaken, err := checkBranchInPreExecution(evm, tx, br)
		if err != nil {
			return err
		}
		if isTaken != br.GetBranchDirection() {
			// encounter inconsistent path, conduct fast-path repair
			var callMap = make(map[int]*vm.Snapshot)
			var stackElement uint256.Int
			curSnapshots := br.GetSnapshots()
			callStack := result.GetCallStack()
			latestSnapshot := curSnapshots[len(curSnapshots)-1]
			if len(callStack) > 0 {
				// internal call exists
				for depth, sps := range callStack {
					// put the latest snapshot under each depth into the call map
					if depth == latestSnapshot.GetDepth() {
						callMap[depth] = latestSnapshot
						continue
					}
					callMap[depth] = sps[len(sps)-1]
				}
			} else {
				callMap[1] = latestSnapshot
			}

			// modify the branch info
			if isTaken {
				pc := latestSnapshot.GetPC()
				latestSnapshot.UpdatePC(pc + 1)
				br.DecideDirection(1)
				if br.GetJudgement() != "EQ" {
					stackElement.SetUint64(1)
					latestSnapshot.GetStack().UpdatePeek(stackElement)
				}
			} else {
				br.DecideDirection(0)
				if br.GetJudgement() == "EQ" {
					jumpPc := latestSnapshot.GetJumpPC()
					latestSnapshot.UpdatePC(jumpPc)
				} else {
					pc := latestSnapshot.GetPC()
					latestSnapshot.UpdatePC(pc + 1)
					stackElement.SetUint64(0)
					latestSnapshot.GetStack().UpdatePeek(stackElement)
				}
			}
			result.Reset(i, latestSnapshot.GetDepth(), latestSnapshot.GetPC())

			// recover execution from the initial snapshot
			evm.Interpreter().SetFastEnabled()
			evm.Interpreter().SetCallMap(callMap)
			_, _, err2 := evm.Call(vm.AccountRef(context.Origin), *context.To, context.Data, callMap[1].GetContract().Gas, context.Value)
			if err2 != nil {
				return err2
			}
			break
		}
	}
	return nil
}

// sort transactions in a set according to their gas fees
func sortTXs(repairedTXs []common.Hash, txSet map[common.Hash]*types.Transaction, tipMap map[common.Hash]*big.Int) []*types.Transaction {
	var (
		identicalTXMap = make(map[common.Hash]struct{})
		output         []*types.Transaction
	)

	for i, txID := range repairedTXs {
		tx, exist := txSet[txID]
		if !exist {
			continue
		}
		if i == 0 {
			output = append(output, tx)
			identicalTXMap[txID] = struct{}{}
			continue
		}

		if _, exist2 := identicalTXMap[txID]; !exist2 {
			tip := tipMap[txID]
			insertLoc := sort.Search(len(output), func(j int) bool {
				comparedTip := tipMap[output[j].Hash()]
				if tip.Cmp(comparedTip) > 0 {
					return true
				} else {
					return false
				}
			})
			output = append(output[:insertLoc], append([]*types.Transaction{tx}, output[insertLoc:]...)...)
			identicalTXMap[txID] = struct{}{}
		}
	}

	return output
}

// checkBranchInExecution checks whether the current state satisfies the stored branch info to perform quick path (during execution)
func checkBranchInExecution(evm *vm.EVM, tx *types.Transaction, branch *vm.BranchContext) (bool, error) {
	// do not encounter any branches under the last snapshot
	if !branch.GetFilled() {
		return branch.GetBranchDirection(), nil
	}

	var (
		firstVal, secondVal uint256.Int
		compact, compact2   bool
		err                 error
	)
	su, _ := branch.GetStateUnit().(*vm.StateUnit)
	slot := su.GetSlot()
	offset := su.GetOffset()
	bits := su.GetBits()
	if offset.Uint64() > 0 && bits < 256 {
		compact = true
	}

	firstVal, err = vm.GetComparedVal(evm, tx.Hash(), branch.GetAddr(), slot, offset, bits, su, compact, false)
	if err != nil {
		return false, err
	}

	tracingUnit := branch.GetTracingUnit()
	if branch.IsVar() {
		cUnit, _ := tracingUnit.(*vm.StateUnit)
		slot2 := cUnit.GetSlot()
		offset2 := cUnit.GetOffset()
		bits2 := cUnit.GetBits()
		if offset2.Uint64() > 0 && bits2 < 256 {
			compact2 = true
		}
		// utilizes the stateDB to check
		secondVal, err = vm.GetComparedVal(evm, tx.Hash(), branch.GetAddr(), slot2, offset2, bits2, cUnit, compact2, false)
		if err != nil {
			return false, err
		}
	} else {
		secondVal = tracingUnit.GetValue()
	}

	// compute the current branch direction
	direction := branch.GetJudgementDirection()
	judgement := vm.StringToOp(branch.GetJudgement())
	if direction {
		vm.Compute(&firstVal, &secondVal, judgement)
		if secondVal.Uint64() == 1 {
			return true, nil
		}
	} else {
		vm.Compute(&secondVal, &firstVal, judgement)
		if firstVal.Uint64() == 1 {
			return true, nil
		}
	}
	return false, nil
}

// checkBranchInPreExecution checks whether the current state satisfies the stored branch info to perform quick path (during pre-execution repair)
func checkBranchInPreExecution(evm *vm.EVM, tx *types.Transaction, branch *vm.BranchContext) (bool, error) {
	// do not encounter any branches under the last snapshot
	if !branch.GetFilled() {
		return branch.GetBranchDirection(), nil
	}

	var (
		firstVal, secondVal uint256.Int
		compact, compact2   bool
		err                 error
	)
	su, _ := branch.GetStateUnit().(*vm.StateUnit)
	slot := su.GetSlot()
	offset := su.GetOffset()
	bits := su.GetBits()
	if offset.Uint64() > 0 && bits < 256 {
		compact = true
	}

	// We do not need to perform pre-execution repair for those branches including the variables relevant to the block environment
	if strings.Compare(su.GetBlockEnv(), "nil") == 0 {
		// utilizes the multi-version cache to check
		firstVal, err = vm.GetComparedVal(evm, tx.Hash(), branch.GetAddr(), slot, offset, bits, su, compact, true)
		if err != nil {
			return false, err
		}
	} else {
		firstVal = su.GetValue()
	}

	tracingUnit := branch.GetTracingUnit()
	if branch.IsVar() {
		cUnit, _ := tracingUnit.(*vm.StateUnit)
		slot2 := cUnit.GetSlot()
		offset2 := cUnit.GetOffset()
		bits2 := cUnit.GetBits()
		if offset2.Uint64() > 0 && bits2 < 256 {
			compact2 = true
		}

		if strings.Compare(cUnit.GetBlockEnv(), "nil") == 0 {
			// utilizes the multi-version cache to check
			secondVal, err = vm.GetComparedVal(evm, tx.Hash(), branch.GetAddr(), slot2, offset2, bits2, cUnit, compact2, true)
			if err != nil {
				return false, err
			}
		} else {
			secondVal = tracingUnit.GetValue()
		}
	} else {
		secondVal = tracingUnit.GetValue()
	}

	// compute the current branch direction
	direction := branch.GetJudgementDirection()
	judgement := vm.StringToOp(branch.GetJudgement())
	if direction {
		vm.Compute(&firstVal, &secondVal, judgement)
		if secondVal.Uint64() == 1 {
			return true, nil
		} else {
			return false, nil
		}
	} else {
		vm.Compute(&secondVal, &firstVal, judgement)
		if firstVal.Uint64() == 1 {
			return true, nil
		} else {
			return false, nil
		}
	}
}

// updateSstore re-computes the stored value according to the cached sstore info
func updateSstore(sstore *vm.SstoreInfo, evm *vm.EVM, isMultiVersion bool) error {
	contractAddr := sstore.GetCallerAddr()
	locUnit := sstore.GetLocUnit()
	valUnit := sstore.GetValUnit()
	loc := locUnit.GetValue()
	originalVal := sstore.GetUpdatedValue()
	if sunit, ok := valUnit.(*vm.StateUnit); ok {
		slot := sunit.GetSlot()
		if sstore.GetCompact() {
			newVal, err := computeCompactedVar(evm, evm.TxContext.ID, contractAddr, originalVal, sunit, isMultiVersion)
			if err != nil {
				return err
			}
			if isMultiVersion {
				if newVal.Cmp(&originalVal) != 0 {
					// update the value stored in MVCache and the sstore info
					if _, err2 := evm.MVCache.SetStorageForWrite(contractAddr, loc, newVal, evm.TxContext.ID, evm.TxContext.GasTip); err2 != nil {
						return err2
					}
					sstore.UpdateVal(newVal)
				}
			} else {
				compactedSstore(evm, contractAddr, loc, newVal, sunit)
			}
		} else {
			zeroOffset := uint256.Int{}
			zeroOffset.SetUint64(0)
			newVal, err := vm.GetComparedVal(evm, evm.TxContext.ID, contractAddr, slot, zeroOffset, 256, sunit, false, isMultiVersion)
			if err != nil {
				return err
			}
			if isMultiVersion {
				if newVal.Cmp(&originalVal) != 0 {
					// update the value stored in MVCache and the sstore info
					if _, err2 := evm.MVCache.SetStorageForWrite(contractAddr, loc, newVal, evm.TxContext.ID, evm.TxContext.GasTip); err2 != nil {
						return err2
					}
					sstore.UpdateVal(newVal)
				}
			} else {
				evm.StateDB.SetState(contractAddr, loc.Bytes32(), newVal.Bytes32())
			}
		}
	} else {
		// 直接赋值，在真正执行时，直接存储到stateDB
		if !isMultiVersion {
			evm.StateDB.SetState(contractAddr, loc.Bytes32(), originalVal.Bytes32())
		}
	}
	return nil
}

// computeCompactedVar computes the latest stored value of a state variable when storage is compacted
func computeCompactedVar(evm *vm.EVM, txID common.Hash, contractAddr common.Address, originalVal uint256.Int, su *vm.StateUnit, isMultiVersion bool) (uint256.Int, error) {
	tracers := su.GetTracer()
	if len(tracers) > 0 {
		lastTr := tracers[len(tracers)-1]
		op := vm.StringToOp(lastTr.GetOps())
		if op == vm.OR {
			unit := lastTr.GetAttaching()
			if sunit, ok := unit.(*vm.StateUnit); ok {
				var compact bool
				sunit.DeleteLastOp() // delete the mul operation
				slot := sunit.GetSlot()
				offset := sunit.GetOffset()
				bits := sunit.GetBits()
				if offset.Uint64() > 0 && bits < 256 {
					compact = true
				}
				newVal, err := vm.GetComparedVal(evm, txID, contractAddr, slot, offset, bits, sunit, compact, isMultiVersion)
				if err != nil {
					return uint256.Int{}, err
				}
				return newVal, nil
			}
		}
	}
	return originalVal, nil
}

// compactedSstore performs compacted sstore operation based on the latest value of a state variable
func compactedSstore(evm *vm.EVM, contractAddr common.Address, loc, newVal uint256.Int, sunit *vm.StateUnit) {
	var (
		res1 = new(uint256.Int)
		res2 = new(uint256.Int)
	)
	val := evm.StateDB.GetState(contractAddr, loc.Bytes32())
	valu := new(uint256.Int)
	valu.SetBytes(val.Bytes())
	offset := sunit.GetOffset()
	bits := sunit.GetBits()
	mask := vm.MakeMask(bits)
	// 计算第一部分
	res1.Mul(mask, &offset)
	res1.Not(res1)
	res1.And(res1, valu)
	// 计算第二部分
	if sunit.GetSignExtend() {
		opVal := new(uint256.Int)
		opVal.SetUint64(uint64(bits/8 - 1))
		newVal.ExtendSign(&newVal, opVal)
	}
	res2.And(mask, &newVal)
	res2.Mul(res2, &offset)
	// 两部分Or操作
	res2.Or(res2, res1)
	evm.StateDB.SetState(contractAddr, loc.Bytes32(), res2.Bytes32())
}

// IsContract judges if the account is a contract address
func IsContract(stateDB vm.StateDB, addr *common.Address) bool {
	size := stateDB.GetCodeSize(*addr)
	return size > 0
}

func DCCDA(txNum int, Htxs *minHeap.TxsHeap, Hready *minHeap.ReadyHeap, Hcommit *minHeap.CommitHeap, stateDb *state.SeerStateDB, dg dependencyGraph.DependencyGraph, complicatedTXs map[int]struct{}) (time.Duration, float64, float64) {
	// 按照论文的说法，如果没有给定交易依赖图，那么所有交易的sv(storageVersion)就默认设置为-1，否则设置为交易的依赖中序号最大的那个
	abortedMap := make(map[int]int)

	if dg == nil {
		for i := 0; i < txNum; i++ {
			Htxs.Push(-1, i, 0)
		}
	} else {
		for i := 0; i < txNum; i++ {
			Htxs.Push(dg[i], i, 0)
		}
	}

	next := 0

	t := time.Now()
	for next < txNum {
		// Stage 1: Schedule
		for true {
			txToCheckReady := Htxs.Pop()
			if txToCheckReady == nil {
				break
			}
			if txToCheckReady.StorageVersion > next-1 { //保证之前依赖的最大的交易已经被执行完
				Htxs.Push(txToCheckReady.StorageVersion, txToCheckReady.Index, txToCheckReady.Incarnation)
				break
			} else {
				Hready.Push(txToCheckReady.Index, txToCheckReady.Incarnation, txToCheckReady.StorageVersion, false)
			}
		}

		// Stage 2: Execution
		// thread will fetch tasks from Hready itself
		// after execution, tasks will be push into Hcommit

		// Stage 3: Commit/Abort
		for true {
			txToCommit := Hcommit.Pop()
			if txToCommit == nil {
				break
			}
			if txToCommit.Index != next {
				Hcommit.Push(txToCommit)
				break
			}

			// 验证依赖版本sv+1到id-1所有交易有没有存在写的行为
			aborted := stateDb.ValidateReadSet(txToCommit.LastReads, txToCommit.StorageVersion)
			if aborted {
				abortedMap[txToCommit.Index]++
				Htxs.Push(txToCommit.Index-1, txToCommit.Index, txToCommit.Incarnation+1)
			} else {
				stateDb.Record(&state.TxInfoMini{Index: txToCommit.Index, Incarnation: txToCommit.Incarnation}, txToCommit.LastReads, txToCommit.LastWrites)
				next += 1
			}
		}
	}
	e := time.Since(t)
	fmt.Printf("Concurrent execution latency：%s\n", e)

	abortedCNum := 0
	abortedNNum := 0
	if complicatedTXs != nil {
		for index := range abortedMap {
			if _, exist := complicatedTXs[index]; exist {
				abortedCNum++
			} else {
				abortedNNum++
			}
		}
	}
	abortRateC := float64(abortedCNum) / float64(txNum)
	abortRateN := float64(abortedNNum) / float64(txNum)

	return e, abortRateC, abortRateN
}
