package core

import (
	"context"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/math"
	"github.com/ethereum/go-ethereum/params"
	"github.com/shirou/gopsutil/v4/mem"
	"log"
	"math/big"
	"seerEVM/core/state"
	"seerEVM/core/types"
	"seerEVM/core/vm"
	"seerEVM/minHeap"
	"sort"
	"strings"
	"sync"
	"time"
)

// SeerThread implements the Seer thread for executing and validating txs
type SeerThread struct {
	// thread id
	ThreadID int
	// 以太坊原生db
	commonStateDB *state.StateDB
	// 多线程共用的公共存储db，保存多版本数据
	publicStateDB *state.SeerStateDB
	// publicStateDB的克隆，专门用于模拟执行交易生成读写集，进而生成交易依赖图，在正式执行时不使用该db
	simulationStateDB *state.SeerStateDB
	// publicStateDB的克隆，专门用于生成交易依赖图，在正式执行时不使用该db
	constructionStateDB *state.SeerStateDB
	// 交易专属db，保存交易读写数据
	txStateDB *state.SeerTransaction
	// 待执行交易池，线程从这里面获取新任务并执行
	taskPool *minHeap.ReadyHeap
	// 待提交交易池，线程执行完任务后将交易放入
	commitPool *minHeap.CommitHeap

	// 存储变量相关的分支信息表
	varTable *vm.VarTable
	// 多版本缓存
	mvCache *state.MVCache
	// 预执行表
	preExecutionTable *vm.PreExecutionTable

	// 整个区块和区块中的交易
	block *types.Block
	// 区块上下文，一次性生成且不能改动
	blockContext vm.BlockContext
	// chainConfig
	chainConfig *params.ChainConfig

	// 记录分支预测命中率
	cTotalPredictions       int
	nTotalPredictions       int
	cUnsatisfiedPredictions int
	nUnsatisfiedPredictions int
	cSatisfiedTxs           int
	nSatisfiedTxs           int

	// 接收交易的通道
	txSource   <-chan []*types.Transaction
	disorder   <-chan *types.Transaction
	txSet      <-chan map[common.Hash]*types.Transaction
	tip        <-chan map[common.Hash]*big.Int
	blkChannel <-chan *types.Block
}

// NewThread creates a new instance of ICSE thread
func NewThread(threadId int, stateDB *state.StateDB, publicStateDB *state.SeerStateDB, stateDbForConstruction *state.SeerStateDB, readyHeap *minHeap.ReadyHeap,
	commitHeap *minHeap.CommitHeap, varTable *vm.VarTable, mvCache *state.MVCache, preExecutionTable *vm.PreExecutionTable,
	block *types.Block, blockContext vm.BlockContext, chainConfig *params.ChainConfig) *SeerThread {
	st := &SeerThread{
		ThreadID:                threadId,
		commonStateDB:           stateDB,
		publicStateDB:           publicStateDB,
		simulationStateDB:       nil,
		constructionStateDB:     stateDbForConstruction,
		txStateDB:               nil,
		taskPool:                readyHeap,
		commitPool:              commitHeap,
		varTable:                varTable,
		mvCache:                 mvCache,
		preExecutionTable:       preExecutionTable,
		block:                   block,
		blockContext:            blockContext,
		chainConfig:             chainConfig,
		cTotalPredictions:       0,
		nTotalPredictions:       0,
		cUnsatisfiedPredictions: 0,
		nUnsatisfiedPredictions: 0,
		cSatisfiedTxs:           0,
		nSatisfiedTxs:           0,
	}
	return st
}

func (st *SeerThread) Run(ctx context.Context, isFastPath, PrintDetails bool) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
			task := st.taskPool.Pop()
			if task == nil {
				continue
			}
			if PrintDetails {
				fmt.Printf("线程%d开始任务%+v\n", st.ThreadID, task)
			}
			var (
				readSet  []*state.ReadLoc
				writeSet state.WriteSets
			)
			if isFastPath {
				readSet, writeSet, _, _ = st.executeTaskFastPath(task, PrintDetails)
			} else {
				readSet, writeSet, _, _ = st.executeTask(task, PrintDetails)
			}
			commitItem := minHeap.NewCommitItem(task.Index, task.Incarnation, task.StorageVersion, readSet, writeSet)
			st.commitPool.Push(commitItem)
			if PrintDetails {
				fmt.Printf("线程%d完成任务%+v\n", st.ThreadID, task)
			}
		}
	}
}

// SingleRun runs a single thread for testing perfect dependency
func (st *SeerThread) SingleRun(txNum int, PrintDetails bool) {
	counter := 0
	for true {
		task := st.taskPool.Pop()
		if task == nil && counter == txNum {
			break
		} else if task == nil && counter == 0 {
			continue
		}
		if PrintDetails {
			fmt.Printf("线程%d开始任务%+v\n", st.ThreadID, task)
		}
		readSet, writeSet, _, _ := st.executeTask(task, PrintDetails)
		commitItem := minHeap.NewCommitItem(task.Index, task.Incarnation, task.StorageVersion, readSet, writeSet)
		st.commitPool.Push(commitItem)
		counter++
		if PrintDetails {
			fmt.Printf("线程%d完成任务%+v\n", st.ThreadID, task)
		}
	}
}

// PreExecution executes transactions speculatively with a branch prediction approach
func (st *SeerThread) PreExecution(txSet map[common.Hash]*types.Transaction, tipMap map[common.Hash]*big.Int, isNative, enablePredictor, storeCheckpoint bool, txGasDistribution map[common.Hash]int) (map[common.Hash]vm.ReadSet, map[common.Hash]vm.WriteSet) {
	var (
		header = st.block.Header()
		gp     = new(GasPool).AddGas(st.block.GasLimit())
	)

	for _, tx := range st.block.Transactions() {
		st.singlePreEexcution(tx, header, txSet, tipMap, gp, isNative, false, enablePredictor, storeCheckpoint, false, txGasDistribution)
	}
	// 清理多版本缓存
	st.mvCache.ClearCache()
	return st.OutputRWSets()
}

// PreExecutionWithDisorder executes transactions speculatively with a branch prediction approach (under the case of disordered tx input)
func (st *SeerThread) PreExecutionWithDisorder(isNative, enableRepair, enablePredictor, storeCheckpoint, fullStorage bool, recorder *Recorder) time.Duration {
	block := <-st.blkChannel
	txMap := <-st.txSet
	tipMap := <-st.tip
	executionQueue := <-st.txSource

	var (
		header       = block.Header()
		gp           = new(GasPool).AddGas(block.GasLimit())
		processedTxs = make([]*types.Transaction, 0, len(txMap))
	)

	t0 := time.Now()
	for {
		if len(processedTxs) == len(txMap) {
			break
		}
		select {
		case insertTx := <-st.disorder:
			if !enableRepair {
				// 立刻执行
				t := time.Now()
				st.singlePreEexcution(insertTx, header, txMap, tipMap, gp, isNative, enableRepair, enablePredictor, storeCheckpoint, fullStorage, nil)
				processedTxs = append(processedTxs, insertTx)
				e := time.Since(t)
				if recorder != nil && recorder.GetObserveIndicator() {
					complicatedTxs := recorder.GetComplicatedMap()
					if _, ok := complicatedTxs[insertTx.Hash()]; ok {
						recorder.RecordCtxPreLatency(e.Microseconds())
					} else {
						if int(insertTx.Gas()) > 21000 {
							recorder.RecordNtxPreLatency(e.Microseconds())
						}
					}
				}
				continue
			}
			t := time.Now()
			insertTip := math.BigMin(insertTx.GasTipCap(), new(big.Int).Sub(insertTx.GasFeeCap(), block.BaseFee()))
			if len(processedTxs) > 0 {
				latest := processedTxs[len(processedTxs)-1]
				latestTip := math.BigMin(latest.GasTipCap(), new(big.Int).Sub(latest.GasFeeCap(), block.BaseFee()))
				if insertTip.Cmp(latestTip) <= 0 {
					// insert into the executionQueue
					insertLoc := sort.Search(len(executionQueue), func(i int) bool {
						tip := math.BigMin(executionQueue[i].GasTipCap(), new(big.Int).Sub(executionQueue[i].GasFeeCap(), block.BaseFee()))
						if insertTip.Cmp(tip) > 0 {
							return true
						} else {
							return false
						}
					})
					executionQueue = append(executionQueue[:insertLoc], append([]*types.Transaction{insertTx}, executionQueue[insertLoc:]...)...)
					continue
				}
			}
			// 立刻执行
			st.singlePreEexcution(insertTx, header, txMap, tipMap, gp, isNative, enableRepair, enablePredictor, storeCheckpoint, fullStorage, nil)
			processedTxs = append(processedTxs, insertTx)
			e := time.Since(t)
			if recorder != nil && recorder.GetObserveIndicator() {
				complicatedTxs := recorder.GetComplicatedMap()
				if _, ok := complicatedTxs[insertTx.Hash()]; ok {
					recorder.RecordCtxPreLatency(e.Microseconds())
				} else {
					if int(insertTx.Gas()) > 21000 {
						recorder.RecordNtxPreLatency(e.Microseconds())
					}
				}
			}
		default:
			// 立刻执行
			t := time.Now()
			if len(executionQueue) == 0 {
				continue
			}
			nextTx := executionQueue[0]
			st.singlePreEexcution(nextTx, header, txMap, tipMap, gp, isNative, enableRepair, enablePredictor, storeCheckpoint, fullStorage, nil)
			executionQueue = executionQueue[1:]
			processedTxs = append(processedTxs, nextTx)
			e := time.Since(t)
			if recorder != nil && recorder.GetObserveIndicator() {
				complicatedTxs := recorder.GetComplicatedMap()
				if _, ok := complicatedTxs[nextTx.Hash()]; ok {
					recorder.RecordCtxPreLatency(e.Microseconds())
				} else {
					if int(nextTx.Gas()) > 21000 {
						recorder.RecordNtxPreLatency(e.Microseconds())
					}
				}
			}
		}
	}
	e0 := time.Since(t0)
	//fmt.Printf("Pre-execution latency is: %s\n", e0)

	// 清理多版本缓存
	st.mvCache.ClearCache()
	return e0
}

func (st *SeerThread) singlePreEexcution(tx *types.Transaction, header *types.Header, txMap map[common.Hash]*types.Transaction, tipMap map[common.Hash]*big.Int, gp *GasPool, isNative, enableRepair, enablePredictor, storeCheckpoint, fullStorage bool, txGasDistribution map[common.Hash]int) {
	var stateDB vm.StateDB
	var vmenv *vm.EVM
	if isNative {
		stateDB = st.commonStateDB.Copy()
	} else {
		// new tx statedb
		stateDB = state.NewIcseTxStateDB(tx, 0, 0, -1, st.publicStateDB)
	}

	msg, err := TransactionToMessage(tx, types.MakeSigner(st.chainConfig, header.Number), header.BaseFee)
	if err != nil {
		fmt.Printf("could not apply tx [%v]: %v\n", tx.Hash().Hex(), err)
	}
	tip := tipMap[tx.Hash()]
	txContext := NewEVMTxContext(msg, tx.Hash(), tip)

	vmenv = vm.NewEVM2(st.blockContext, txContext, stateDB, st.varTable, st.preExecutionTable,
		st.mvCache, st.chainConfig, vm.Config{EnablePreimageRecording: false}, true, enablePredictor, storeCheckpoint, fullStorage, txGasDistribution)

	res := st.preExecutionTable.InitialResult(tx.Hash())
	res.UpdateTxContext(txContext)

	// 避免Nonce错误
	stateDB.SetNonce(msg.From, msg.Nonce)
	//st.commonStateDB.SetNonce(msg.From, msg.Nonce)
	// 避免Balance错误
	mgval := new(big.Int).SetUint64(msg.GasLimit)
	mgval = mgval.Mul(mgval, msg.GasPrice)
	balanceCheck := mgval
	if msg.GasFeeCap != nil {
		balanceCheck = new(big.Int).SetUint64(msg.GasLimit)
		balanceCheck = balanceCheck.Mul(balanceCheck, msg.GasFeeCap)
		balanceCheck.Add(balanceCheck, msg.Value)
	}
	//st.commonStateDB.AddBalance(msg.From, balanceCheck)
	stateDB.AddBalance(msg.From, balanceCheck)

	_ = applySeerTransaction(msg, gp, vmenv, txMap, tipMap, enableRepair, enablePredictor, storeCheckpoint, fullStorage)
	//if err3 != nil {
	//	fmt.Printf("could not apply tx [%v]: %v\n", tx.Hash().Hex(), err3)
	//}
}

// FastExecution executes transactions in a fast-path mode using a single thread
func (st *SeerThread) FastExecution(stateDB *state.StateDB, recorder *Recorder, enableRepair, enablePredictor, enableFast, isStatistics, testMemory bool, fileName string) (*common.Hash, types.Receipts, []*types.Log, uint64, int, float64, error) {
	var (
		receipts    types.Receipts
		usedGas     = new(uint64)
		header      = st.block.Header()
		blockHash   = st.block.Hash()
		blockNumber = st.block.Number()
		allLogs     []*types.Log
		gp          = new(GasPool).AddGas(1000 * st.block.GasLimit())
		ctxNum      int
		memUsed     float64
		wg          sync.WaitGroup
	)

	if testMemory {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var sumUsed float64
			for j := 0; j < len(st.block.Transactions()); j++ {
				memInfo, _ := mem.VirtualMemory()
				sumUsed += memInfo.UsedPercent
			}
			memUsed = sumUsed / float64(len(st.block.Transactions()))
		}()
	}

	t := time.Now()
	// Iterate over and process the individual transactions
	for i, tx := range st.block.Transactions() {
		ret, _ := st.preExecutionTable.GetResult(tx.Hash())
		txContext := ret.GetTxContext()
		msg, err := TransactionToMessage(tx, types.MakeSigner(st.chainConfig, header.Number), header.BaseFee)
		if err != nil {
			return nil, nil, nil, 0, 0, 0, fmt.Errorf("could not apply tx %d [%v]: %w", i, tx.Hash().Hex(), err)
		}

		// 避免Nonce错误
		stateDB.SetNonce(msg.From, msg.Nonce)
		// 避免Balance错误
		mgval := new(big.Int).SetUint64(msg.GasLimit)
		mgval = mgval.Mul(mgval, msg.GasPrice)
		balanceCheck := mgval
		if msg.GasFeeCap != nil {
			balanceCheck = new(big.Int).SetUint64(msg.GasLimit)
			balanceCheck = balanceCheck.Mul(balanceCheck, msg.GasFeeCap)
			balanceCheck.Add(balanceCheck, msg.Value)
		}
		stateDB.AddBalance(msg.From, balanceCheck)

		vmenv := vm.NewEVM2(st.blockContext, txContext, stateDB, st.varTable, st.preExecutionTable, st.mvCache,
			st.chainConfig, vm.Config{EnablePreimageRecording: false}, false, false, false, false, nil)
		receipt, isContractCall, err := applyFastPath(msg, tx, vmenv, ret, gp, blockNumber, blockHash, usedGas, st, recorder, enableRepair, enablePredictor, enableFast)
		if err != nil {
			return nil, nil, nil, 0, 0, 0, fmt.Errorf("could not apply tx %d [%v]: %w", i, tx.Hash().Hex(), err)
		}
		if isContractCall {
			ctxNum++
		}
		receipts = append(receipts, receipt)
		allLogs = append(allLogs, receipt.Logs...)
		// 设置上链标识
		st.preExecutionTable.SetOnChain(tx.Hash())
	}
	e := time.Since(t)
	fmt.Printf("Execution latency is: %s\n", e)

	if testMemory {
		wg.Wait()
	}

	// 更新分支历史与感知器
	if enablePredictor {
		err := st.UpdateBranchHistory()
		if err != nil {
			return nil, nil, nil, 0, 0, 0, err
		}
	}
	// 清理分支变量表与预执行表的冗余项
	st.varTable.Sweep()
	st.preExecutionTable.ClearTable()
	// 计算分支统计信息
	if isStatistics {
		st.varTable.Statistics(ctxNum, fileName)
	}

	// Fail if Shanghai not enabled and len(withdrawals) is non-zero.
	withdrawals := st.block.Withdrawals()
	if len(withdrawals) > 0 && !st.chainConfig.IsShanghai(st.block.Time()) {
		return nil, nil, nil, 0, 0, memUsed, fmt.Errorf("withdrawals before shanghai")
	}
	// Finalize the block, applying any consensus engine specific extras (e.g. block rewards)
	accumulateRewards(st.chainConfig, stateDB, header, st.block.Uncles())
	root := stateDB.IntermediateRoot(st.chainConfig.IsEIP158(header.Number))
	return &root, receipts, allLogs, *usedGas, ctxNum, memUsed, nil
}

// FastExecutionRemovingIO executes transactions in a fast-path mode withou modifying states for removing I/O
func (st *SeerThread) FastExecutionRemovingIO(stateDB *state.StateDB) error {
	var (
		usedGas     = new(uint64)
		header      = st.block.Header()
		blockHash   = st.block.Hash()
		blockNumber = st.block.Number()
		gp          = new(GasPool).AddGas(st.block.GasLimit())
	)

	// Iterate over and process the individual transactions
	for i, tx := range st.block.Transactions() {
		ret, _ := st.preExecutionTable.GetResult(tx.Hash())
		txContext := ret.GetTxContext()
		msg, err := TransactionToMessage(tx, types.MakeSigner(st.chainConfig, header.Number), header.BaseFee)
		if err != nil {
			return fmt.Errorf("could not apply tx %d [%v]: %w", i, tx.Hash().Hex(), err)
		}

		// 避免Nonce错误
		stateDB.SetNonce(msg.From, msg.Nonce)
		// 避免Balance错误
		mgval := new(big.Int).SetUint64(msg.GasLimit)
		mgval = mgval.Mul(mgval, msg.GasPrice)
		balanceCheck := mgval
		if msg.GasFeeCap != nil {
			balanceCheck = new(big.Int).SetUint64(msg.GasLimit)
			balanceCheck = balanceCheck.Mul(balanceCheck, msg.GasFeeCap)
			balanceCheck.Add(balanceCheck, msg.Value)
		}
		stateDB.AddBalance(msg.From, balanceCheck)

		vmenv := vm.NewEVM2(st.blockContext, txContext, stateDB, st.varTable, st.preExecutionTable, st.mvCache,
			st.chainConfig, vm.Config{EnablePreimageRecording: false}, false, false, false, false, nil)
		_, _, err = applyFastPath(msg, tx, vmenv, ret, gp, blockNumber, blockHash, usedGas, st, nil, true, true, true)
		if err != nil {
			return fmt.Errorf("could not apply tx %d [%v]: %w", i, tx.Hash().Hex(), err)
		}
	}

	return nil
}

// executeTask one thread executes a transaction with its fast path (under concurrent execution)
func (st *SeerThread) executeTaskFastPath(task *minHeap.ReadyItem, PrintDetails bool) ([]*state.ReadLoc, state.WriteSets, *types.Receipt, []*types.Log) {
	var (
		usedGas     = new(uint64)
		header      = st.block.Header()
		blockHash   = st.block.Hash()
		blockNumber = st.block.Number()
		gp          = new(GasPool).AddGas(st.block.GasLimit())
		tx          = st.block.Transactions()[task.Index]
	)

	// new tx statedb
	st.txStateDB = state.NewIcseTxStateDB(tx, task.Index, task.Incarnation, task.StorageVersion, st.publicStateDB)
	// create evm
	ret, _ := st.preExecutionTable.GetResult(tx.Hash())
	txContext := ret.GetTxContext()
	msg, err := TransactionToMessage(tx, types.MakeSigner(st.chainConfig, header.Number), header.BaseFee)
	if err != nil {
		log.Panic(fmt.Errorf("could not format transaction %+v to message", task))
	}
	vmenv := vm.NewEVM2(st.blockContext, txContext, st.txStateDB, st.varTable, st.preExecutionTable, st.mvCache,
		st.chainConfig, vm.Config{EnablePreimageRecording: false}, false, false, false, false, nil)
	st.txStateDB.SetTxContext(tx.Hash(), task.Index)

	// 避免Nonce错误
	st.txStateDB.SetNonce(msg.From, msg.Nonce)
	// 避免Balance错误
	mgval := new(big.Int).SetUint64(msg.GasLimit)
	mgval = mgval.Mul(mgval, msg.GasPrice)
	balanceCheck := mgval
	if msg.GasFeeCap != nil {
		balanceCheck = new(big.Int).SetUint64(msg.GasLimit)
		balanceCheck = balanceCheck.Mul(balanceCheck, msg.GasFeeCap)
		balanceCheck.Add(balanceCheck, msg.Value)
	}
	st.txStateDB.AddBalance(msg.From, balanceCheck)

	receipt, _, err := applyFastPath(msg, tx, vmenv, ret, gp, blockNumber, blockHash, usedGas, st, nil, false, true, true)
	if err != nil {
		log.Panic(fmt.Errorf("could not apply transaction %+v: %s", task, err))
	}

	// get read/write set and store into public statedb
	readSet, writeSet := st.txStateDB.OutputRWSet() // 得到tx_statedb中记录的读写集
	// 过滤读写集中的矿工地址
	readSet, writeSet = FilterRWSet(readSet, writeSet, st.block.Header().Coinbase.String())

	// 打印读写集
	if PrintDetails {
		var rs []string
		for _, r := range readSet {
			rs = append(rs, r.Location.String())
		}
		fmt.Printf("交易%+v的读集是：\n%s\n交易%+v的写集是：\n%s\n", task, strings.Join(rs, "\n"), task, writeSet)
	}

	return readSet, writeSet, receipt, receipt.Logs
}

// executeTask executes a transaction without a fast path
func (st *SeerThread) executeTask(task *minHeap.ReadyItem, PrintDetails bool) ([]*state.ReadLoc, state.WriteSets, *types.Receipt, []*types.Log) {
	var (
		usedGas     = new(uint64)
		header      = st.block.Header()
		blockHash   = st.block.Hash()
		blockNumber = st.block.Number()
		gp          = new(GasPool).AddGas(st.block.GasLimit())
		tx          = st.block.Transactions()[task.Index]
	)

	// 如果目前正在生成依赖图阶段，则用publicStateDB2；如果是正式执行，则用publicStateDB
	var publicStateDB *state.SeerStateDB
	if task.IsSimulation {
		publicStateDB = st.simulationStateDB
		//publicStateDB = st.constructionStateDB
	} else {
		publicStateDB = st.publicStateDB
	}

	// new tx statedb
	st.txStateDB = state.NewIcseTxStateDB(tx, task.Index, task.Incarnation, task.StorageVersion, publicStateDB)

	// create evm
	vmenv := vm.NewEVM(st.blockContext, vm.TxContext{}, st.txStateDB, st.chainConfig, vm.Config{EnablePreimageRecording: false})
	msg, err := TransactionToMessage(tx, types.MakeSigner(st.chainConfig, header.Number), header.BaseFee)
	if err != nil {
		log.Panic(fmt.Errorf("could not format transaction %+v to message", task))
	}
	st.txStateDB.SetTxContext(tx.Hash(), task.Index)

	// 避免Nonce错误
	st.txStateDB.SetNonce(msg.From, msg.Nonce)
	// 避免Balance错误
	mgval := new(big.Int).SetUint64(msg.GasLimit)
	mgval = mgval.Mul(mgval, msg.GasPrice)
	balanceCheck := mgval
	if msg.GasFeeCap != nil {
		balanceCheck = new(big.Int).SetUint64(msg.GasLimit)
		balanceCheck = balanceCheck.Mul(balanceCheck, msg.GasFeeCap)
		balanceCheck.Add(balanceCheck, msg.Value)
	}
	st.txStateDB.AddBalance(msg.From, balanceCheck)

	receipt, err := applyNormalTransaction(msg, gp, st.txStateDB, blockNumber, blockHash, tx, usedGas, vmenv)
	if err != nil {
		log.Panic(fmt.Errorf("could not apply transaction %+v: %s", task, err))
	}

	// get read/write set and store into public statedb
	readSet, writeSet := st.txStateDB.OutputRWSet() // 得到tx_statedb中记录的读写集

	// 过滤读写集中的矿工地址
	readSet, writeSet = FilterRWSet(readSet, writeSet, st.block.Header().Coinbase.String())

	// 打印读写集
	if PrintDetails {
		var rs []string
		for _, r := range readSet {
			rs = append(rs, r.Location.String())
		}
		fmt.Printf("交易%+v的读集是：\n%s\n交易%+v的写集是：\n%s\n", task, strings.Join(rs, "\n"), task, writeSet)
	}

	// 读写集记录到public_statedb中 (真正执行时需要等到commit后才能刷新进statedb)
	if task.IsSimulation {
		st.constructionStateDB.Record(&state.TxInfoMini{Index: task.Index, Incarnation: task.Incarnation}, readSet, writeSet)
	}
	//publicStateDB.Record(&state.TxInfoMini{Index: task.Index, Incarnation: task.Incarnation}, readSet, writeSet)

	return readSet, writeSet, receipt, receipt.Logs
}

// FinalizeBlock conducts cache sweep, state finalise, and rewards distribution after execution
func (st *SeerThread) FinalizeBlock(stateDB *state.SeerStateDB) (common.Hash, error) {
	// 更新分支历史与感知器
	err := st.UpdateBranchHistory()
	if err != nil {
		return common.Hash{}, err
	}
	// 在预执行表中设置上链标识
	for _, tx := range st.block.Transactions() {
		st.preExecutionTable.SetOnChain(tx.Hash())
	}
	// 清理分支变量表与预执行表的冗余项
	st.varTable.Sweep()
	st.preExecutionTable.ClearTable()

	// Fail if Shanghai not enabled and len(withdrawals) is non-zero.
	withdrawals := st.block.Withdrawals()
	if len(withdrawals) > 0 && !st.chainConfig.IsShanghai(st.block.Time()) {
		return common.Hash{}, fmt.Errorf("withdrawals before shanghai")
	}
	// Finalize the block, applying any consensus engine specific extras (e.g. block rewards)
	accumulateRewards2(st.chainConfig, stateDB, st.block.Header(), st.block.Uncles())
	stateDB.FinaliseMVMemory()
	//root := stateDB.IntermediateRoot(st.chainConfig.IsEIP158(st.block.Header().Number))
	root, err2 := stateDB.Commit(st.chainConfig.IsEIP158(st.block.Header().Number))
	if err2 != nil {
		return common.Hash{}, err2
	}
	return root, nil
}

// OutputRWSets outputs read/write sets of pre-executed transactions
func (st *SeerThread) OutputRWSets() (map[common.Hash]vm.ReadSet, map[common.Hash]vm.WriteSet) {
	var readSets = make(map[common.Hash]vm.ReadSet)
	var writeSets = make(map[common.Hash]vm.WriteSet)
	for _, tx := range st.block.Transactions() {
		ret, _ := st.preExecutionTable.GetResult(tx.Hash())
		rs, ws := ret.OutputRWSet()
		readSets[tx.Hash()] = rs
		writeSets[tx.Hash()] = ws
	}
	return readSets, writeSets
}

// UpdateBranchHistory updates branch history info
func (st *SeerThread) UpdateBranchHistory() error {
	for _, tx := range st.block.Transactions() {
		ret, _ := st.preExecutionTable.GetResult(tx.Hash())
		brs := ret.GetBranches()
		if len(brs) > 0 {
			for _, br := range brs {
				sUnit, _ := br.GetStateUnit().(*vm.StateUnit)
				if br.GetFilled() && strings.Compare(sUnit.GetBlockEnv(), "nil") == 0 {
					if err := st.varTable.AddHistory(br); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func (st *SeerThread) IncrementCTotal()             { st.cTotalPredictions++ }
func (st *SeerThread) IncrementNTotal()             { st.nTotalPredictions++ }
func (st *SeerThread) IncrementCUnsatisfied()       { st.cUnsatisfiedPredictions++ }
func (st *SeerThread) IncrementNUnsatisfied()       { st.nUnsatisfiedPredictions++ }
func (st *SeerThread) IncrementCSatisfiedTxs()      { st.cSatisfiedTxs++ }
func (st *SeerThread) IncrementNSatisfiedTxs()      { st.nSatisfiedTxs++ }
func (st *SeerThread) UpdateBlock(blk *types.Block) { st.block = blk }
func (st *SeerThread) SetChannels(source <-chan []*types.Transaction, disorder <-chan *types.Transaction, txMap <-chan map[common.Hash]*types.Transaction, tip <-chan map[common.Hash]*big.Int, blkChannel <-chan *types.Block) {
	st.txSource = source
	st.disorder = disorder
	st.txSet = txMap
	st.tip = tip
	st.blkChannel = blkChannel
}
func (st *SeerThread) GetPredictionResults() (int, int, int, int, int, int) {
	return st.cTotalPredictions, st.nTotalPredictions, st.cUnsatisfiedPredictions, st.nUnsatisfiedPredictions, st.cSatisfiedTxs, st.nSatisfiedTxs
}
func (st *SeerThread) GetVarTable() *vm.VarTable                   { return st.varTable }
func (st *SeerThread) GetMVCache() *state.MVCache                  { return st.mvCache }
func (st *SeerThread) GetPreExecutionTable() *vm.PreExecutionTable { return st.preExecutionTable }

// FilterRWSet 将读写集中包含的矿工地址删除（因为每个交易都会读写矿工地址）
func FilterRWSet(readSet []*state.ReadLoc, writeSet state.WriteSets, minerAddr string) ([]*state.ReadLoc, state.WriteSets) {
	filteredReadSet := make([]*state.ReadLoc, 0, len(readSet))
	for _, r := range readSet {
		addr, _ := r.Location.Address()
		if addr.String() != minerAddr {
			filteredReadSet = append(filteredReadSet, r)
		}
	}

	for k, _ := range writeSet {
		if k.String() == minerAddr {
			delete(writeSet, k)
		}
	}

	return filteredReadSet, writeSet
}
