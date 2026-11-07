package experiments

import (
	"bufio"
	"context"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	math2 "github.com/ethereum/go-ethereum/common/math"
	"math/big"
	"os"
	"seerEVM/client"
	"seerEVM/config"
	"seerEVM/core"
	"seerEVM/core/state"
	"seerEVM/core/state/snapshot"
	"seerEVM/core/types"
	"seerEVM/core/vm"
	"seerEVM/database"
	"seerEVM/dependencyGraph"
	"seerEVM/minHeap"
	"sync"
	"time"
)

const exp_branch_statistics = "./branch_statistics.txt"
const exp_preExecution_large_ratio = "./preExecution_large_ratio.txt"
const exp_prediction_height = "./prediction_height.txt"
const exp_speedup_perTx_basic = "./speedup_perTx_basic.txt"
const exp_speedup_perTx_repair = "./speedup_perTx_repair.txt"
const exp_speedup_perTx_predictor = "./speedup_perTx_predictor.txt"
const exp_speedup_perTx_full = "./speedup_perTx_full.txt"
const exp_concurrent_speedup = "./concurrent_speedup.txt"
const exp_concurrent_abort = "./concurrent_abort.txt"
const exp_prediction_breakdown_basic = "./prediction_breakdown_basic.txt"
const exp_prediction_breakdown_repair = "./prediction_breakdown_repair.txt"
const exp_prediction_breakdown_predictor = "./prediction_breakdown_predictor.txt"
const exp_prediction_breakdown_full = "./prediction_breakdown_full.txt"
const exp_preExecution_breakdown_basic = "./preExecution_breakdown_basic.txt"
const exp_preExecution_breakdown_repair = "./preExecution_breakdown_repair.txt"
const exp_preExecution_breakdown_predictor = "./preExecution_breakdown_predictor.txt"
const exp_preExecution_breakdown_full = "./preExecution_breakdown_full.txt"
const exp_memorycpu_usage_basic = "./memorycpu_usage_basic.txt"
const exp_memorycpu_usage_repair = "./memorycpu_usage_repair.txt"
const exp_memorycpu_usage_predictor = "./memorycpu_usage_predictor.txt"
const exp_memorycpu_usage_full = "./memorycpu_usage_full.txt"
const exp_memorycpu_usage_sweep = "./memorycpu_usage_sweep.txt"
const exp_memorycpu_usage_fmv = "./memorycpu_usage_fmv.txt"
const exp_memorycpu_usage_eth = "./memorycpu_usage_eth.txt"
const exp_io_eth = "./io_eth.txt"
const exp_io_seer = "./io_seer.txt"

// TestPreExecutionLarge evaluates the pre-execution latency under varying number of transactions
func TestPreExecutionLarge(txNum int, startingHeight, offset int64, ratio float64, enableRepair bool) error {
	//var latInSec float64
	file, err := os.OpenFile(exp_preExecution_large_ratio, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintf(writer, "===== Run Seer with %.1f disorder ratio under %d number of txs =====\n", ratio, txNum)

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}

	blockPre, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		combinedTxs []*types.Transaction
		parent      *types.Header = blockPre.Header()
		parentRoot  *common.Hash  = &parent.Root
		// 该state.Database接口的具体类型为state.cachingDB，其中的disk字段为db
		stateCache state.Database = database.NewStateCache(db)
		snaps      *snapshot.Tree = database.NewSnap(db, stateCache, blockPre.Header())
		tipMap                    = make(map[common.Hash]*big.Int)
		txSource                  = make(chan []*types.Transaction)
		disorder                  = make(chan *types.Transaction)
		txMap                     = make(chan map[common.Hash]*types.Transaction)
		tipChan                   = make(chan map[common.Hash]*big.Int)
		blockChan                 = make(chan *types.Block)
		wg         sync.WaitGroup
	)

	txLen := 0
	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}
		newTxs := blk.Transactions()
		if txLen+len(newTxs) >= txNum {
			for j := 0; j < txNum-txLen; j++ {
				combinedTxs = append(combinedTxs, newTxs[j])
				tip := math2.BigMin(newTxs[j].GasTipCap(), new(big.Int).Sub(newTxs[j].GasFeeCap(), blk.BaseFee()))
				tipMap[newTxs[j].Hash()] = tip
			}
			break
		}
		for _, tx := range newTxs {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			tipMap[tx.Hash()] = tip
		}
		combinedTxs = append(combinedTxs, newTxs...)
		txLen += len(newTxs)
	}

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight+1))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}
	startBlock.AddTransactions(combinedTxs)

	serialDB, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)
	_, _, receipts, _, _, _ := serialProcessor.Process(startBlock, serialDB, vm.Config{EnablePreimageRecording: false}, nil, true)
	// divide normal contract txs and complicated contract txs
	recorder := core.NewRecorder(true)
	for _, receipt := range receipts {
		txHash := receipt.TxHash
		gasUsed := receipt.GasUsed
		if gasUsed >= 100000 {
			recorder.AppendComplicatedTx(txHash)
		}
	}
	db.Close()

	// 新建原生数据库
	db2, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db2.Close()
	stateCache = database.NewStateCache(db2)
	snaps = database.NewSnap(db2, stateCache, blockPre.Header())
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()
	blockContext := core.NewEVMBlockContext(startBlock.Header(), db2, nil)
	newThread := core.NewThread(0, stateDb, nil, nil, nil, nil, branchTable, mvCache, preTable, startBlock, blockContext, config.MainnetChainConfig)
	newThread.SetChannels(txSource, disorder, txMap, tipChan, blockChan)

	// 创建交易分发客户端
	cli := client.NewFakeClient(txSource, disorder, txMap, tipChan, blockChan)

	wg.Add(2)
	go func() {
		defer wg.Done()
		cli.Run(db, txNum, startingHeight, offset, true, false, startBlock, tipMap, ratio)
	}()
	go func() {
		defer wg.Done()
		newThread.PreExecutionWithDisorder(true, enableRepair, true, true, false, recorder)
		//latInSec = float64(lat.Microseconds()) / float64(1000000)
	}()
	wg.Wait()

	ctxPreLatencyInSec := float64(recorder.GetCtxPreLatency()) / float64(1000000)
	ntxPreLatencyInSec := float64(recorder.GetNtxPreLatency()) / float64(1000000)
	totalPreLatencyInSec := ctxPreLatencyInSec + ntxPreLatencyInSec
	fmt.Fprintf(writer, "Ctx pre-execution latency: %.2f s\n", ctxPreLatencyInSec)
	fmt.Fprintf(writer, "Ntx pre-execution latency: %.2f s\n", ntxPreLatencyInSec)
	fmt.Fprintf(writer, "Total pre-execution latency: %.2f s\n", totalPreLatencyInSec)
	//fmt.Fprintf(writer, "Total latency for pre-execution phase: %.2f s\n", latInSec)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestPredictionSuccess evaluates the average prediction accuracy in a specific interval of block heights
func TestPredictionSuccess(startingHeight, offset int64, ratio float64) error {
	file, err := os.OpenFile(exp_prediction_height, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		parent              *types.Header  = startBlock.Header()
		parentRoot          *common.Hash   = &parent.Root
		stateCache          state.Database = database.NewStateCache(db)
		snaps               *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		tipMap                             = make(map[common.Hash]*big.Int)
		txSource                           = make(chan []*types.Transaction)
		disorder                           = make(chan *types.Transaction)
		txMap                              = make(chan map[common.Hash]*types.Transaction)
		tipChan                            = make(chan map[common.Hash]*big.Int)
		blockChan                          = make(chan *types.Block)
		wg                  sync.WaitGroup
		expData1            [][]float64 = make([][]float64, 0)
		expData2            [][]float64 = make([][]float64, 0)
		totalCTxRatio       float64
		totalNTxRatio       float64
		totalCBranchRatio   float64
		totalNBranchRatio   float64
		validBlockNumForCTx int
		validBlockNumForNTx int
	)

	// 新建原生数据库
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)
	serialDB, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)

	// 构建预测用的内存结构
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()

	// 新建性能观察器
	recorder := core.NewRecorder(false)

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}

		for _, tx := range blk.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			tipMap[tx.Hash()] = tip
		}

		// 首先去除I/O的影响，先执行加载状态数据到内存
		_, _, receipts, _, _, _ := serialProcessor.Process(blk, serialDB, vm.Config{EnablePreimageRecording: false}, nil, false)
		_, _ = serialDB.Commit(config.MainnetChainConfig.IsEIP158(blk.Number()))

		// divide normal contract txs and complicated contract txs
		recorder.ResetComplicatedMap()
		for _, receipt := range receipts {
			txHash := receipt.TxHash
			gasUsed := receipt.GasUsed
			if gasUsed >= 100000 {
				recorder.AppendComplicatedTx(txHash)
			}
		}

		preStateDB := stateDb.Copy()
		blockContext := core.NewEVMBlockContext(blk.Header(), db, nil)
		newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, blk, blockContext, config.MainnetChainConfig)
		newThread.SetChannels(txSource, disorder, txMap, tipChan, blockChan)

		// 创建交易分发客户端
		cli := client.NewFakeClient(txSource, disorder, txMap, tipChan, blockChan)

		wg.Add(2)
		go func() {
			defer wg.Done()
			cli.Run(db, 0, i.Int64(), 0, false, false, nil, tipMap, ratio)
		}()
		go func() {
			defer wg.Done()
			newThread.PreExecutionWithDisorder(true, true, true, true, false, nil)
		}()
		wg.Wait()

		_, _, _, _, ctxNum, _, err := newThread.FastExecution(stateDb, recorder, true, true, true, false, false, "")
		if err != nil {
			fmt.Println("execution error", err)
		}
		// Commit all cached state changes into underlying memory database.
		root, _ := stateDb.Commit(config.MainnetChainConfig.IsEIP158(blk.Number()))

		cTotal, nTotal, cUnsatisfied, nUnsatisfied, cSatisfiedTxs, nSatisfiedTxs := newThread.GetPredictionResults()
		complicatedTxsNum := len(recorder.GetComplicatedMap())
		ctxRatio := float64(cSatisfiedTxs) / float64(complicatedTxsNum)
		ntxRatio := float64(nSatisfiedTxs) / float64(ctxNum-complicatedTxsNum)
		cBranchRatio := float64(cTotal-cUnsatisfied) / float64(cTotal)
		nBranchRatio := float64(nTotal-nUnsatisfied) / float64(nTotal)

		if ctxNum != 0 && cTotal != 0 {
			totalCTxRatio += ctxRatio
			totalCBranchRatio += cBranchRatio
			expData1 = append(expData1, []float64{ctxRatio, cBranchRatio})
			validBlockNumForCTx++
		}

		if ctxNum != 0 && nTotal != 0 {
			totalNTxRatio += ntxRatio
			totalNBranchRatio += nBranchRatio
			expData2 = append(expData2, []float64{ntxRatio, nBranchRatio})
			validBlockNumForNTx++
		}
		//fmt.Printf("Ratio of satisfied branch dircetions: %.2f, ratio of satisfied txs: %.2f\n", branchRatio, txRatio)
		fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay block number "+i.String(), root)

		// metadata reference to keep trie alive
		//serialDB.Database().TrieDB().Reference(root0, common.Hash{})
		//snaps = database.NewSnap(db, stateCache, blk.Header())
		//serialDB, _ = state.New(root0, stateCache, snaps)
		//serialDB.Database().TrieDB().Dereference(parentRoot)
		//parentRoot = root0

		// write to the experimental script
		tmpBlockNum := i.Int64() - startingHeight
		if tmpBlockNum%200 == 0 {
			fmt.Fprintf(writer, "===== Block height %d =====\n", tmpBlockNum)
			avgCTxRatio := totalCTxRatio / float64(validBlockNumForCTx)
			avgCBranchRatio := totalCBranchRatio / float64(validBlockNumForCTx)
			avgNTxRatio := totalNTxRatio / float64(validBlockNumForNTx)
			avgNBranchRatio := totalNBranchRatio / float64(validBlockNumForNTx)

			for _, row := range expData1 {
				_, err = fmt.Fprintf(writer, "%.2f && %.2f\n", row[0], row[1])
				if err != nil {
					fmt.Printf("write error: %v\n", err)
					return nil
				}
			}

			fmt.Fprintln(writer)

			for _, row := range expData2 {
				_, err = fmt.Fprintf(writer, "%.2f && %.2f\n", row[0], row[1])
				if err != nil {
					fmt.Printf("write error: %v\n", err)
					return nil
				}
			}

			fmt.Fprintf(writer, "Average ctx ratio: %.2f, average cbranch ratio: %.2f\n", avgCTxRatio, avgCBranchRatio)
			fmt.Fprintf(writer, "Average ntx ratio: %.2f, average nbranch ratio: %.2f\n", avgNTxRatio, avgNBranchRatio)

			err = writer.Flush()
			if err != nil {
				fmt.Printf("flush error: %v\n", err)
				return nil
			}

			totalCTxRatio = 0
			totalNTxRatio = 0
			totalCBranchRatio = 0
			totalNBranchRatio = 0
			validBlockNumForCTx = 0
			validBlockNumForNTx = 0
			expData1 = make([][]float64, 0)
			expData2 = make([][]float64, 0)
		}
	}

	return nil
}

// TestSpeedupPerTx evaluates the speedup distribution on single transaction execution (including design breakdown)
func TestSpeedupPerTx(startingHeight, offset int64, ratio float64, enableRepair, enablePredictor, enableFast bool) error {
	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		parent         *types.Header  = startBlock.Header()
		parentRoot     *common.Hash   = &parent.Root
		stateCache     state.Database = database.NewStateCache(db)
		snaps          *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		tipMap                        = make(map[common.Hash]*big.Int)
		txSource                      = make(chan []*types.Transaction)
		disorder                      = make(chan *types.Transaction)
		txMap                         = make(chan map[common.Hash]*types.Transaction)
		tipChan                       = make(chan map[common.Hash]*big.Int)
		blockChan                     = make(chan *types.Block)
		wg             sync.WaitGroup
		totalSpeedups  int
		totalCSpeedups int
		totalNSpeedups int
		largeRatio     float64
	)

	// 新建原生数据库
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)
	serialDB, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)

	// 构建预测用的内存结构
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()

	// 新建性能观察器
	recorder := core.NewRecorder(true)

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}

		for _, tx := range blk.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			tipMap[tx.Hash()] = tip
		}

		// 首先去除I/O的影响，先执行加载状态数据到内存
		_, _, _, _, _, _ = serialProcessor.Process(blk, serialDB.Copy(), vm.Config{EnablePreimageRecording: false}, nil, false)
		_, _, receipts, _, _, _ := serialProcessor.Process(blk, serialDB, vm.Config{EnablePreimageRecording: false}, recorder, false)
		root0, _ := serialDB.Commit(config.MainnetChainConfig.IsEIP158(blk.Number()))

		preStateDB := stateDb.Copy()
		blockContext := core.NewEVMBlockContext(blk.Header(), db, nil)
		newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, blk, blockContext, config.MainnetChainConfig)
		newThread.SetChannels(txSource, disorder, txMap, tipChan, blockChan)

		// 创建交易分发客户端
		cli := client.NewFakeClient(txSource, disorder, txMap, tipChan, blockChan)

		wg.Add(2)
		go func() {
			defer wg.Done()
			cli.Run(db, 0, i.Int64(), 0, false, false, nil, tipMap, ratio)
		}()
		go func() {
			defer wg.Done()
			newThread.PreExecutionWithDisorder(true, enableRepair, enablePredictor, true, false, nil)
		}()
		wg.Wait()

		_, _, _, _, _, _, err := newThread.FastExecution(stateDb, recorder, enableRepair, enablePredictor, enableFast, false, false, "")
		if err != nil {
			fmt.Println("execution error", err)
		}
		// Commit all cached state changes into underlying memory database.
		root, _ := stateDb.Commit(config.MainnetChainConfig.IsEIP158(blk.Number()))
		fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay block number "+i.String(), root)

		// divide normal contract txs and complicated contract txs
		for _, receipt := range receipts {
			txHash := receipt.TxHash
			gasUsed := receipt.GasUsed
			if gasUsed >= 100000 {
				recorder.AppendComplicatedTx(txHash)
			}
		}

		// reset stateDB every epoch to remove accumulated I/O overhead
		snaps = database.NewSnap(db, stateCache, blk.Header())
		serialDB, _ = state.New(root0, stateCache, snaps)
		stateDb, _ = state.New(root, stateCache, snaps)
	}

	var fileName string
	if enableRepair && !enablePredictor && !enableFast {
		fileName = exp_speedup_perTx_repair
	} else if enablePredictor && !enableFast {
		fileName = exp_speedup_perTx_predictor
	} else if enableFast {
		fileName = exp_speedup_perTx_full
	} else {
		fileName = exp_speedup_perTx_basic
	}

	file, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	fmt.Fprintf(writer, "===== Run Seer with %.1f disorder ratio =====\n", ratio)

	cSpeedupMap, nSpeedupMap := recorder.SpeedupCalculation()
	totalTxNum := recorder.GetValidTxNum()
	totalCTxNum := recorder.GetComplicatedTxNum()
	totalNTxNum := recorder.GetNormalTxNum()

	for speedup, num := range cSpeedupMap {
		if speedup >= 200 {
			//totalSpeedups += 50 * num
			totalTxNum -= num
			totalCTxNum -= num
		} else {
			totalSpeedups += speedup * num
			totalCSpeedups += speedup * num
		}
		rat := float64(num) / float64(recorder.GetComplicatedTxNum())
		if speedup >= 50 {
			largeRatio += rat
		} else {
			_, err = fmt.Fprintf(writer, "%d %.4f\n", speedup, rat)
			if err != nil {
				fmt.Printf("write error: %v\n", err)
				return nil
			}
		}
	}

	_, err = fmt.Fprintf(writer, ">=50 %.4f\n", largeRatio)
	if err != nil {
		fmt.Printf("write error: %v\n", err)
		return nil
	}

	_, err = fmt.Fprintf(writer, "Average speedup of complicated Txs: %.4f\n", float64(totalCSpeedups)/float64(totalCTxNum))
	if err != nil {
		fmt.Printf("write error: %v\n", err)
		return nil
	}

	fmt.Fprintln(writer)
	largeRatio = 0
	for speedup, num := range nSpeedupMap {
		if speedup >= 200 {
			//totalSpeedups += 50 * num
			totalTxNum -= num
			totalNTxNum -= num
		} else {
			totalSpeedups += speedup * num
			totalNSpeedups += speedup * num
		}
		rat := float64(num) / float64(recorder.GetNormalTxNum())
		if speedup >= 50 {
			largeRatio += rat
		} else {
			_, err = fmt.Fprintf(writer, "%d %.4f\n", speedup, rat)
			if err != nil {
				fmt.Printf("write error: %v\n", err)
				return nil
			}
		}
	}

	_, err = fmt.Fprintf(writer, ">=50 %.4f\n", largeRatio)
	if err != nil {
		fmt.Printf("write error: %v\n", err)
		return nil
	}

	_, err = fmt.Fprintf(writer, "Average speedup of normal Txs: %.4f\n", float64(totalNSpeedups)/float64(totalNTxNum))
	if err != nil {
		fmt.Printf("write error: %v\n", err)
		return nil
	}

	fmt.Println()
	_, err = fmt.Fprintf(writer, "Average speedup for all Txs: %.4f\n", float64(totalSpeedups)/float64(totalTxNum))
	if err != nil {
		fmt.Printf("write error: %v\n", err)
		return nil
	}

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestSeerBreakDown conducts factor analysis under design breakdown (prediction accuracy and pre-execution latency)
func TestSeerBreakDown(startingHeight, offset int64, ratio float64, enableRepair, enablePredictor, enableFast bool) error {
	var predictionFile, preExecutionFile string

	if enableRepair && !enablePredictor && !enableFast {
		predictionFile = exp_prediction_breakdown_repair
		preExecutionFile = exp_preExecution_breakdown_repair
	} else if enablePredictor && !enableFast {
		predictionFile = exp_prediction_breakdown_predictor
		preExecutionFile = exp_preExecution_breakdown_predictor
	} else if enableFast {
		predictionFile = exp_prediction_breakdown_full
		preExecutionFile = exp_preExecution_breakdown_full
	} else {
		predictionFile = exp_prediction_breakdown_basic
		preExecutionFile = exp_preExecution_breakdown_basic
	}

	filePrediction, err := os.OpenFile(predictionFile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer filePrediction.Close()

	filePreExecution, err := os.OpenFile(preExecutionFile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer filePreExecution.Close()

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}

	var (
		largeBlock         *types.Block
		count              int
		combinedTxs        types.Transactions
		tipMap             = make(map[common.Hash]*big.Int)
		txSource           = make(chan []*types.Transaction)
		disorder           = make(chan *types.Transaction)
		txMap              = make(chan map[common.Hash]*types.Transaction)
		tipChan            = make(chan map[common.Hash]*big.Int)
		blockChan          = make(chan *types.Block)
		wg                 sync.WaitGroup
		predictionData     [][]float64 = make([][]float64, 0)
		preExecutionData   [][]float64 = make([][]float64, 0)
		totalTxRatio       float64
		totalBranchRatio   float64
		preLatencyPerBlock float64
		totalPreLatency    float64
		validBlockNum      int
	)

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		parent     *types.Header  = startBlock.Header()
		parentRoot *common.Hash   = &parent.Root
		stateCache state.Database = database.NewStateCache(db)
		snaps      *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
	)

	// 新建原生数据库
	serialDB, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)
	// 新建性能观察器
	recorder := core.NewRecorder(false)

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}
		_, _, receipts, _, _, _ := serialProcessor.Process(blk, serialDB, vm.Config{EnablePreimageRecording: false}, nil, false)
		serialDB.Commit(config.MainnetChainConfig.IsEIP158(blk.Number()))
		// divide normal contract txs and complicated contract txs
		for _, receipt := range receipts {
			txHash := receipt.TxHash
			gasUsed := receipt.GasUsed
			if gasUsed >= 100000 {
				recorder.AppendComplicatedTx(txHash)
			}
		}
	}
	db.Close()

	db2, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db2.Close()

	// 构建预测用的内存结构
	stateCache = database.NewStateCache(db2)
	snaps = database.NewSnap(db2, stateCache, startBlock.Header())
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()

	min, max, addSpan = big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		count++
		blk, err2 := database.GetBlockByNumber(db2, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}
		if count == 1 {
			largeBlock = blk
		}
		combinedTxs = append(combinedTxs, blk.Transactions()...)

		for _, tx := range blk.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			tipMap[tx.Hash()] = tip
		}

		if count == 100 {
			largeBlock.AddTransactions(combinedTxs)
			preStateDB := stateDb.Copy()
			blockContext := core.NewEVMBlockContext(largeBlock.Header(), db2, nil)
			newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, largeBlock, blockContext, config.MainnetChainConfig)
			newThread.SetChannels(txSource, disorder, txMap, tipChan, blockChan)

			// 创建交易分发客户端
			cli := client.NewFakeClient(txSource, disorder, txMap, tipChan, blockChan)
			recorder.ChangeObserveIndicator()

			wg.Add(2)
			go func() {
				defer wg.Done()
				cli.Run(db, 0, i.Int64(), 0, true, false, largeBlock, tipMap, ratio)
			}()
			go func() {
				defer wg.Done()
				recorder.ResetPreLatency()
				newThread.PreExecutionWithDisorder(true, enableRepair, enablePredictor, true, false, recorder)
				preLatencyPerBlock = float64(recorder.GetCtxPreLatency()+recorder.GetNtxPreLatency()) / float64(1000000)
				totalPreLatency += preLatencyPerBlock
				preExecutionData = append(preExecutionData, []float64{float64(i.Int64()), preLatencyPerBlock})
			}()
			wg.Wait()

			recorder.ChangeObserveIndicator()
			_, _, _, _, ctxNum, _, err := newThread.FastExecution(stateDb, recorder, enableRepair, enablePredictor, enableFast, false, false, "")
			if err != nil {
				fmt.Println("execution error", err)
			}
			// Commit all cached state changes into underlying memory database.
			root, _ := stateDb.Commit(config.MainnetChainConfig.IsEIP158(largeBlock.Number()))

			cTotal, nTotal, cUnsatisfied, nUnsatisfied, cSatisfiedTxs, nSatisfiedTxs := newThread.GetPredictionResults()
			satisfiedTxs := cSatisfiedTxs + nSatisfiedTxs
			unsatisfied := cUnsatisfied + nUnsatisfied
			total := cTotal + nTotal
			txRatio := float64(satisfiedTxs) / float64(ctxNum)
			branchRatio := float64(total-unsatisfied) / float64(total)
			if ctxNum != 0 && total != 0 {
				totalTxRatio += txRatio
				totalBranchRatio += branchRatio
				predictionData = append(predictionData, []float64{txRatio, branchRatio})
				validBlockNum++
			}
			//fmt.Printf("Ratio of satisfied branch dircetions: %.2f, ratio of satisfied txs: %.2f\n", branchRatio, txRatio)
			fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay large block number "+i.String(), root)

			// reset stateDB every epoch to remove accumulated I/O overhead
			//snaps = database.NewSnap(db, stateCache, blk.Header())
			//serialDB, _ = state.New(root0, stateCache, snaps)
			//stateDb, _ = state.New(root, stateCache, snaps)

			count = 0
			combinedTxs = []*types.Transaction{}
			tipMap = make(map[common.Hash]*big.Int)
		}
	}

	// write the prediction results to the script
	avgTxRatio := totalTxRatio / float64(validBlockNum)
	avgBranchRatio := totalBranchRatio / float64(validBlockNum)
	writer := bufio.NewWriter(filePrediction)
	fmt.Fprintf(writer, "===== Run Seer with %.1f disorder ratio =====\n", ratio)
	for _, row := range predictionData {
		_, err = fmt.Fprintf(writer, "%.3f %.3f\n", row[0], row[1])
		if err != nil {
			fmt.Printf("write error: %v\n", err)
			return nil
		}
	}
	fmt.Fprintf(writer, "Avg. tx ratio: %.3f, avg. branch ratio: %.3f\n", avgTxRatio, avgBranchRatio)

	// write the prediction results to the experimental script
	avgLatency := totalPreLatency / float64(offset/100)
	writer2 := bufio.NewWriter(filePreExecution)
	fmt.Fprintf(writer2, "===== Run Seer with %.1f disorder ratio =====\n", ratio)
	for _, row := range preExecutionData {
		_, err = fmt.Fprintf(writer2, "%.3f %.3f\n", row[0], row[1])
		if err != nil {
			fmt.Printf("write error: %v\n", err)
			return nil
		}
	}
	fmt.Fprintf(writer2, "Avg. pre-execution latency: %.3f\n", avgLatency)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}
	err = writer2.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestSeerConcurrentLarge evaluates pre-execution and fast-path concurrent execution using the large block
func TestSeerConcurrentLarge(threads, txNum int, startingHeight, offset int64) error {
	file, err := os.OpenFile(exp_concurrent_speedup, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintf(writer, "===== Run Seer with %d threads under %d number of txs=====\n", threads, txNum)

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	blockPre, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight+1))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		parent     *types.Header = blockPre.Header()
		parentRoot *common.Hash  = &parent.Root
		// 该state.Database接口的具体类型为state.cachingDB，其中的disk字段为db
		stateCache    state.Database = database.NewStateCache(db)
		snaps         *snapshot.Tree = database.NewSnap(db, stateCache, blockPre.Header())
		readSets                     = make(map[common.Hash]vm.ReadSet)
		writeSets                    = make(map[common.Hash]vm.WriteSet)
		txLen         int
		serialLatency int64
	)

	// 新建原生stateDB，用于串行执行测试
	nativeDb, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)

	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	var concurrentTxs types.Transactions
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		block, err := database.GetBlockByNumber(db, i)
		if err != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err)
		}

		preStateDB := nativeDb.Copy()
		blockContext := core.NewEVMBlockContext(block.Header(), db, nil)
		newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, block, blockContext, config.MainnetChainConfig)

		newTxs := block.Transactions()
		if txLen+len(newTxs) >= txNum {
			var remainingTxs types.Transactions
			for j := 0; j < txNum-txLen; j++ {
				concurrentTxs = append(concurrentTxs, newTxs[j])
				remainingTxs = append(remainingTxs, newTxs[j])
			}
			txSet := make(map[common.Hash]*types.Transaction)
			tipMap := make(map[common.Hash]*big.Int)
			for _, tx := range remainingTxs {
				tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), block.BaseFee()))
				txSet[tx.Hash()] = tx
				tipMap[tx.Hash()] = tip
			}
			block.AddTransactions(remainingTxs)
			newThread.UpdateBlock(block)
			rSets, wSets := newThread.PreExecution(txSet, tipMap, true, true, true, nil)
			for id, read := range rSets {
				readSets[id] = read
			}
			for id, write := range wSets {
				writeSets[id] = write
			}
			st := time.Now()
			_, _, _, _, _, _ = serialProcessor.Process(block, nativeDb, vm.Config{EnablePreimageRecording: false}, nil, false)
			ed := time.Since(st)
			serialLatency += ed.Microseconds()
			_, _ = nativeDb.Commit(config.MainnetChainConfig.IsEIP158(startBlock.Number()))
			break
		} else {
			txSet := make(map[common.Hash]*types.Transaction)
			tipMap := make(map[common.Hash]*big.Int)
			for _, tx := range newTxs {
				tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), block.BaseFee()))
				txSet[tx.Hash()] = tx
				tipMap[tx.Hash()] = tip
			}
			rSets, wSets := newThread.PreExecution(txSet, tipMap, true, true, true, nil)
			for id, read := range rSets {
				readSets[id] = read
			}
			for id, write := range wSets {
				writeSets[id] = write
			}
			st := time.Now()
			_, _, _, _, _, _ = serialProcessor.Process(block, nativeDb, vm.Config{EnablePreimageRecording: false}, nil, false)
			ed := time.Since(st)
			serialLatency += ed.Microseconds()
			_, _ = nativeDb.Commit(config.MainnetChainConfig.IsEIP158(startBlock.Number()))
		}
		concurrentTxs = append(concurrentTxs, newTxs...)
		txLen += len(newTxs)
	}
	startBlock.AddTransactions(concurrentTxs)

	// 构建依赖图
	blockContext := core.NewEVMBlockContext(startBlock.Header(), db, nil)
	newThread := core.NewThread(0, nativeDb, nil, nil, nil, nil, branchTable, mvCache, preTable, startBlock, blockContext, config.MainnetChainConfig)
	dg := dependencyGraph.ConstructDependencyGraph(readSets, writeSets, startBlock.Transactions())
	// 尚不可执行的交易队列（因为storage<next，所以尚不可执行）
	Htxs := minHeap.NewTxsHeap()
	// 已经可以执行，但是处于等待状态的交易队列
	Hready := minHeap.NewReadyHeap()
	// 执行完毕，等待验证的交易队列
	Hcommit := minHeap.NewCommitHeap()
	// 执行DCC-DA算法

	// 新建并发执行所需的数据库IcseStateDB
	stateDb, _ := state.NewSeerStateDB(*parentRoot, stateCache, snaps)
	ctx, cancel := context.WithCancel(context.Background())
	for j := 1; j <= threads/2; j++ {
		go func(threadID int) {
			thread := core.NewThread(threadID, nil, stateDb, nil, Hready, Hcommit, branchTable, mvCache, preTable, startBlock, blockContext, config.MainnetChainConfig)
			thread.Run(ctx, true, false)
		}(j)
	}
	duration, _, _ := core.DCCDA(startBlock.Transactions().Len(), Htxs, Hready, Hcommit, stateDb, dg, nil)
	cancel()

	// Commit all cached state changes into underlying memory database.
	_, err2 := newThread.FinalizeBlock(stateDb)
	if err2 != nil {
		return fmt.Errorf("finalize block error: %s", err2)
	}

	speedup := float64(serialLatency) / float64(duration.Microseconds())
	serialTPS := float64(txNum) / (float64(serialLatency) / float64(1000000))
	concurrentTPS := float64(txNum) / (float64(duration.Microseconds()) / float64(1000000))
	fmt.Fprintf(writer, "Serial latency is: %.2f, tps is: %.2f\n", float64(serialLatency)/float64(1000000), serialTPS)
	fmt.Fprintf(writer, "Concurrent latency is: %s, tps is %.2f\n", duration, concurrentTPS)
	fmt.Fprintf(writer, "Avg. speedup is: %.2f\n", speedup)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestSeerConcurrentAbort evaluates transaction abort rate under concurrent execution with the large block
func TestSeerConcurrentAbort(threads, txNum int, startingHeight, offset int64) error {
	file, err := os.OpenFile(exp_concurrent_abort, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintf(writer, "===== Run Seer with %d threads under %d number of txs=====\n", threads, txNum)

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	blockPre, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight+1))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		parent     *types.Header = blockPre.Header()
		parentRoot *common.Hash  = &parent.Root
		// 该state.Database接口的具体类型为state.cachingDB，其中的disk字段为db
		stateCache     state.Database = database.NewStateCache(db)
		snaps          *snapshot.Tree = database.NewSnap(db, stateCache, blockPre.Header())
		readSets                      = make(map[common.Hash]vm.ReadSet)
		writeSets                     = make(map[common.Hash]vm.WriteSet)
		complicatedTXs                = make(map[int]struct{})
		txLen          int
	)

	// 新建原生stateDB，用于串行执行测试
	nativeDb, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)

	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	var concurrentTxs types.Transactions
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		block, err := database.GetBlockByNumber(db, i)
		if err != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err)
		}

		preStateDB := nativeDb.Copy()
		blockContext := core.NewEVMBlockContext(block.Header(), db, nil)
		newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, block, blockContext, config.MainnetChainConfig)

		newTxs := block.Transactions()
		if txLen+len(newTxs) >= txNum {
			var remainingTxs types.Transactions
			for j := 0; j < txNum-txLen; j++ {
				concurrentTxs = append(concurrentTxs, newTxs[j])
				remainingTxs = append(remainingTxs, newTxs[j])
			}
			txSet := make(map[common.Hash]*types.Transaction)
			tipMap := make(map[common.Hash]*big.Int)
			for _, tx := range remainingTxs {
				tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), block.BaseFee()))
				txSet[tx.Hash()] = tx
				tipMap[tx.Hash()] = tip
			}
			block.AddTransactions(remainingTxs)
			newThread.UpdateBlock(block)
			rSets, wSets := newThread.PreExecution(txSet, tipMap, true, true, true, nil)
			for id, read := range rSets {
				readSets[id] = read
			}
			for id, write := range wSets {
				writeSets[id] = write
			}
			_, _, receipts, _, _, _ := serialProcessor.Process(block, nativeDb, vm.Config{EnablePreimageRecording: false}, nil, false)
			_, _ = nativeDb.Commit(config.MainnetChainConfig.IsEIP158(startBlock.Number()))

			for j, receipt := range receipts {
				gasUsed := receipt.GasUsed
				if gasUsed >= 100000 {
					index := txLen + j
					complicatedTXs[index] = struct{}{}
				}
			}
			break
		} else {
			txSet := make(map[common.Hash]*types.Transaction)
			tipMap := make(map[common.Hash]*big.Int)
			for _, tx := range newTxs {
				tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), block.BaseFee()))
				txSet[tx.Hash()] = tx
				tipMap[tx.Hash()] = tip
			}
			rSets, wSets := newThread.PreExecution(txSet, tipMap, true, true, true, nil)
			for id, read := range rSets {
				readSets[id] = read
			}
			for id, write := range wSets {
				writeSets[id] = write
			}
			_, _, receipts, _, _, _ := serialProcessor.Process(block, nativeDb, vm.Config{EnablePreimageRecording: false}, nil, false)
			_, _ = nativeDb.Commit(config.MainnetChainConfig.IsEIP158(startBlock.Number()))

			for j, receipt := range receipts {
				gasUsed := receipt.GasUsed
				if gasUsed >= 100000 {
					index := txLen + j
					complicatedTXs[index] = struct{}{}
				}
			}
		}
		concurrentTxs = append(concurrentTxs, newTxs...)
		txLen += len(newTxs)
	}
	startBlock.AddTransactions(concurrentTxs)

	// 构建依赖图
	blockContext := core.NewEVMBlockContext(startBlock.Header(), db, nil)
	newThread := core.NewThread(0, nativeDb, nil, nil, nil, nil, branchTable, mvCache, preTable, startBlock, blockContext, config.MainnetChainConfig)

	dg := dependencyGraph.ConstructDependencyGraph(readSets, writeSets, startBlock.Transactions())
	// 尚不可执行的交易队列（因为storage<next，所以尚不可执行）
	Htxs := minHeap.NewTxsHeap()
	// 已经可以执行，但是处于等待状态的交易队列
	Hready := minHeap.NewReadyHeap()
	// 执行完毕，等待验证的交易队列
	Hcommit := minHeap.NewCommitHeap()
	// 执行DCC-DA算法

	// 新建并发执行所需的数据库IcseStateDB
	stateDb, _ := state.NewSeerStateDB(*parentRoot, stateCache, snaps)
	ctx, cancel := context.WithCancel(context.Background())
	for j := 1; j <= threads/2; j++ {
		go func(threadID int) {
			thread := core.NewThread(threadID, nil, stateDb, nil, Hready, Hcommit, branchTable, mvCache, preTable, startBlock, blockContext, config.MainnetChainConfig)
			thread.Run(ctx, true, false)
		}(j)
	}
	_, abortRateC, abortRateN := core.DCCDA(startBlock.Transactions().Len(), Htxs, Hready, Hcommit, stateDb, dg, complicatedTXs)
	fmt.Fprintf(writer, "Avg. abort rate for complicated TXs is: %.3f\n", abortRateC)
	fmt.Fprintf(writer, "Avg. abort rate for normal TXs is: %.3f\n", abortRateN)
	cancel()

	// Commit all cached state changes into underlying memory database.
	_, err2 := newThread.FinalizeBlock(stateDb)
	if err2 != nil {
		return fmt.Errorf("finalize block error: %s", err2)
	}

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestMemoryCPUBreakDown evaluates the memory cost during pre-execution and fast-path execution
func TestMemoryCPUBreakDown(startingHeight, offset int64, enableRepair, enablePredictor, enableFast, storeCheckpoint, fullStorage bool) error {
	var testFile string

	if enableRepair && !enablePredictor && !enableFast {
		testFile = exp_memorycpu_usage_repair
	} else if enablePredictor && !enableFast {
		testFile = exp_memorycpu_usage_predictor
	} else if enableFast {
		testFile = exp_memorycpu_usage_full
	} else {
		testFile = exp_memorycpu_usage_basic
	}

	if fullStorage {
		testFile = exp_memorycpu_usage_fmv
	}

	file, err := os.OpenFile(testFile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintln(writer, "===== Run Seer for memory and cpu test =====")

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		largeBlock   *types.Block
		count        int
		combinedTxs  types.Transactions
		tipMap       = make(map[common.Hash]*big.Int)
		txSource     = make(chan []*types.Transaction)
		disorder     = make(chan *types.Transaction)
		txMap        = make(chan map[common.Hash]*types.Transaction)
		tipChan      = make(chan map[common.Hash]*big.Int)
		blockChan    = make(chan *types.Block)
		wg           sync.WaitGroup
		parent       *types.Header  = startBlock.Header()
		parentRoot   *common.Hash   = &parent.Root
		stateCache   state.Database = database.NewStateCache(db)
		snaps        *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		avgCPUPre    float64
		maxCPUPre    float64
		avgCPUExe    float64
		maxCPUExe    float64
		totalMaxPre  float64
		totalAvgPre  float64
		totalMaxExe  float64
		totalAvgExe  float64
		avgMemPre    float64
		avgMemExe    float64
		totalMemUsed float64
	)

	// 新建原生数据库
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)

	// 构建预测用的内存结构
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(20, 0.1)
	preTable := vm.NewPreExecutionTable()

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		count++
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}
		if count == 1 {
			largeBlock = blk
		}
		combinedTxs = append(combinedTxs, blk.Transactions()...)

		for _, tx := range blk.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			tipMap[tx.Hash()] = tip
		}

		if count == 100 {
			largeBlock.AddTransactions(combinedTxs)
			preStateDB := stateDb.Copy()
			blockContext := core.NewEVMBlockContext(largeBlock.Header(), db, nil)
			newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, largeBlock, blockContext, config.MainnetChainConfig)
			newThread.SetChannels(txSource, disorder, txMap, tipChan, blockChan)

			// 创建交易分发客户端
			cli := client.NewFakeClient(txSource, disorder, txMap, tipChan, blockChan)

			wg.Add(4)
			go func() {
				defer wg.Done()
				cli.Run(db, 0, i.Int64(), 0, true, false, largeBlock, tipMap, 0.8)
			}()
			go func() {
				defer wg.Done()
				newThread.PreExecutionWithDisorder(true, enableRepair, enablePredictor, storeCheckpoint, fullStorage, nil)
			}()
			go func() {
				defer wg.Done()
				var maxSum, avgSum float64
				for j := 0; j < 20; j++ {
					avgCPU := GetCPUPercent()
					maxCPU := GetCPUPercent2()
					avgSum += avgCPU
					maxSum += maxCPU
				}
				avgCPUPre = avgSum / float64(20)
				maxCPUPre = maxSum / float64(20)
				totalAvgPre += avgCPUPre
				totalMaxPre += maxCPUPre
			}()
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < len(largeBlock.Transactions()); j++ {
					memUsed := GetMemPercent()
					sumUsed += memUsed
					time.Sleep(time.Millisecond)
				}
				avgMemPre = sumUsed / float64(len(largeBlock.Transactions()))
			}()
			wg.Wait()

			wg.Add(1)
			go func() {
				defer wg.Done()
				var maxSum, avgSum float64
				for j := 0; j < 10; j++ {
					avgCPU := GetCPUPercent()
					maxCPU := GetCPUPercent2()
					avgSum += avgCPU
					maxSum += maxCPU
				}
				avgCPUExe = avgSum / float64(10)
				maxCPUExe = maxSum / float64(10)
				totalAvgExe += avgCPUExe
				totalMaxExe += maxCPUExe
			}()

			_, _, _, _, _, avgMemExe, err = newThread.FastExecution(stateDb, nil, enableRepair, enablePredictor, enableFast, false, true, "")
			if err != nil {
				fmt.Println("execution error", err)
			}
			wg.Wait()

			fmt.Fprintf(writer, "Avg. CPU utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), avgCPUPre)
			fmt.Fprintf(writer, "Max. CPU utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), maxCPUPre)
			fmt.Fprintf(writer, "Avg. CPU utilization in execution in block %d: %.2f%%\n", i.Int64(), avgCPUExe)
			fmt.Fprintf(writer, "Max. CPU utilization in execution in block %d: %.2f%%\n", i.Int64(), maxCPUExe)
			fmt.Fprintf(writer, "Memory utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), avgMemPre)
			fmt.Fprintf(writer, "Memory utilization in execution in block %d: %.2f%%\n", i.Int64(), avgMemExe)
			totalMemUsed += (avgMemPre + avgMemExe) / float64(2)

			// Commit all cached state changes into underlying memory database.
			root, _ := stateDb.Commit(config.MainnetChainConfig.IsEIP158(largeBlock.Number()))

			//fmt.Printf("Ratio of satisfied branch dircetions: %.2f, ratio of satisfied txs: %.2f\n", branchRatio, txRatio)
			fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay large block number "+i.String(), root)

			count = 0
			combinedTxs = []*types.Transaction{}
			tipMap = make(map[common.Hash]*big.Int)
		}
	}

	fmt.Fprintln(writer)
	avgMem := totalMemUsed / float64(offset/100)
	avgCPUPre = totalAvgPre / float64(offset/100)
	maxCPUPre = totalMaxPre / float64(offset/100)
	avgCPUExe = totalAvgExe / float64(offset/100)
	maxCPUExe = totalMaxExe / float64(offset/100)
	fmt.Fprintf(writer, "Average memory utilization: %.2f%%\n", avgMem)
	fmt.Fprintf(writer, "Avg. CPU utilization in pre-execution: %.2f%%\n", avgCPUPre)
	fmt.Fprintf(writer, "Max. CPU utilization in pre-execution: %.2f%%\n", maxCPUPre)
	fmt.Fprintf(writer, "Avg. CPU utilization in execution: %.2f%%\n", avgCPUExe)
	fmt.Fprintf(writer, "Max. CPU utilization in execution: %.2f%%\n", maxCPUExe)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestMemoryCPUConcurrent evaluates the memory cost during pre-execution and concurrent fast-path execution
func TestMemoryCPUConcurrent(threads int, startingHeight, offset int64) error {
	file, err := os.OpenFile(exp_memorycpu_usage_full, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintf(writer, "===== Run Seer with %d threads for memory and cpu test =====\n", threads)

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		largeBlock   *types.Block
		count        int
		combinedTxs  types.Transactions
		parent       *types.Header  = startBlock.Header()
		parentRoot   *common.Hash   = &parent.Root
		stateCache   state.Database = database.NewStateCache(db)
		snaps        *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		readSets                    = make(map[common.Hash]vm.ReadSet)
		writeSets                   = make(map[common.Hash]vm.WriteSet)
		txSet                       = make(map[common.Hash]*types.Transaction)
		tipMap                      = make(map[common.Hash]*big.Int)
		avgCPUPre    float64
		maxCPUPre    float64
		avgCPUExe    float64
		maxCPUExe    float64
		totalMaxPre  float64
		totalAvgPre  float64
		totalMaxExe  float64
		totalAvgExe  float64
		avgMemPre    float64
		avgMemExe    float64
		totalMemUsed float64
		wg           sync.WaitGroup
	)

	nativeDb, _ := state.New(*parentRoot, stateCache, snaps)
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		count++
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}
		if count == 1 {
			largeBlock = blk
		}
		combinedTxs = append(combinedTxs, blk.Transactions()...)

		for _, tx := range blk.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			txSet[tx.Hash()] = tx
			tipMap[tx.Hash()] = tip
		}

		if count == 100 {
			largeBlock.AddTransactions(combinedTxs)
			preStateDB := nativeDb.Copy()
			blockContext := core.NewEVMBlockContext(largeBlock.Header(), db, nil)
			newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, largeBlock, blockContext, config.MainnetChainConfig)

			wg.Add(2)
			go func() {
				defer wg.Done()
				var maxSum, avgSum float64
				for j := 0; j < 10; j++ {
					avgCPU := GetCPUPercent()
					maxCPU := GetCPUPercent2()
					avgSum += avgCPU
					maxSum += maxCPU
				}
				avgCPUPre = avgSum / float64(10)
				maxCPUPre = maxSum / float64(10)
				totalAvgPre += avgCPUPre
				totalMaxPre += maxCPUPre
			}()
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < len(largeBlock.Transactions()); j++ {
					memUsed := GetMemPercent()
					sumUsed += memUsed
				}
				avgMemPre = sumUsed / float64(len(largeBlock.Transactions()))
			}()

			rSets, wSets := newThread.PreExecution(txSet, tipMap, true, true, true, nil)
			for id, read := range rSets {
				readSets[id] = read
			}
			for id, write := range wSets {
				writeSets[id] = write
			}
			wg.Wait()

			// 构建依赖图
			newThread = core.NewThread(0, nativeDb, nil, nil, nil, nil, branchTable, mvCache, preTable, largeBlock, blockContext, config.MainnetChainConfig)
			dg := dependencyGraph.ConstructDependencyGraph(readSets, writeSets, largeBlock.Transactions())
			// 尚不可执行的交易队列（因为storage<next，所以尚不可执行）
			Htxs := minHeap.NewTxsHeap()
			// 已经可以执行，但是处于等待状态的交易队列
			Hready := minHeap.NewReadyHeap()
			// 执行完毕，等待验证的交易队列
			Hcommit := minHeap.NewCommitHeap()
			// 新建并发执行所需的数据库IcseStateDB
			stateDb, _ := state.NewSeerStateDB(*parentRoot, stateCache, snaps)

			wg.Add(2)
			go func() {
				defer wg.Done()
				var maxSum, avgSum float64
				for j := 0; j < 10; j++ {
					avgCPU := GetCPUPercent()
					maxCPU := GetCPUPercent2()
					avgSum += avgCPU
					maxSum += maxCPU
				}
				avgCPUExe = avgSum / float64(10)
				maxCPUExe = maxSum / float64(10)
				totalAvgExe += avgCPUExe
				totalMaxExe += maxCPUExe
			}()
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < len(largeBlock.Transactions()); j++ {
					memUsed := GetMemPercent()
					sumUsed += memUsed
				}
				avgMemExe = sumUsed / float64(len(largeBlock.Transactions()))
			}()

			ctx, cancel := context.WithCancel(context.Background())
			for j := 1; j <= threads/2; j++ {
				go func(threadID int) {
					thread := core.NewThread(threadID, nil, stateDb, nil, Hready, Hcommit, branchTable, mvCache, preTable, largeBlock, blockContext, config.MainnetChainConfig)
					thread.Run(ctx, true, false)
				}(j)
			}
			core.DCCDA(largeBlock.Transactions().Len(), Htxs, Hready, Hcommit, stateDb, dg, nil)
			cancel()
			wg.Wait()

			fmt.Fprintf(writer, "Avg. CPU utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), avgCPUPre)
			fmt.Fprintf(writer, "Max. CPU utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), maxCPUPre)
			fmt.Fprintf(writer, "Avg. CPU utilization in execution in block %d: %.2f%%\n", i.Int64(), avgCPUExe)
			fmt.Fprintf(writer, "Max. CPU utilization in execution in block %d: %.2f%%\n", i.Int64(), maxCPUExe)
			fmt.Fprintf(writer, "Memory utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), avgMemPre)
			fmt.Fprintf(writer, "Memory utilization in execution in block %d: %.2f%%\n", i.Int64(), avgMemExe)
			totalMemUsed += (avgMemPre + avgMemExe) / float64(2)

			// Commit all cached state changes into underlying memory database.
			root, err3 := newThread.FinalizeBlock(stateDb)
			if err3 != nil {
				return fmt.Errorf("finalize block error: %s", err3)
			}
			fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay large block number "+i.String(), root)

			count = 0
			combinedTxs = []*types.Transaction{}
			txSet = make(map[common.Hash]*types.Transaction)
			tipMap = make(map[common.Hash]*big.Int)
		}
	}

	fmt.Fprintln(writer)
	avgMem := totalMemUsed / float64(offset/100)
	avgCPUPre = totalAvgPre / float64(offset/100)
	maxCPUPre = totalMaxPre / float64(offset/100)
	avgCPUExe = totalAvgExe / float64(offset/100)
	maxCPUExe = totalMaxExe / float64(offset/100)
	fmt.Fprintf(writer, "Average memory utilization: %.2f%%\n", avgMem)
	fmt.Fprintf(writer, "Avg. CPU utilization in pre-execution: %.2f%%\n", avgCPUPre)
	fmt.Fprintf(writer, "Max. CPU utilization in pre-execution: %.2f%%\n", maxCPUPre)
	fmt.Fprintf(writer, "Avg. CPU utilization in execution: %.2f%%\n", avgCPUExe)
	fmt.Fprintf(writer, "Max. CPU utilization in execution: %.2f%%\n", maxCPUExe)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestMemoryCPUSweep evaluates the memory and cpu cost of the memory-saving version of Seer
func TestMemoryCPUSweep(startingHeight, offset int64) error {
	file, err := os.OpenFile(exp_memorycpu_usage_sweep, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintln(writer, "===== Run Seer for memory and cpu test =====")

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		largeBlock   *types.Block
		count        int
		combinedTxs  types.Transactions
		tipMap       = make(map[common.Hash]*big.Int)
		txSource     = make(chan []*types.Transaction)
		disorder     = make(chan *types.Transaction)
		txMap        = make(chan map[common.Hash]*types.Transaction)
		tipChan      = make(chan map[common.Hash]*big.Int)
		blockChan    = make(chan *types.Block)
		wg           sync.WaitGroup
		parent       *types.Header  = startBlock.Header()
		parentRoot   *common.Hash   = &parent.Root
		stateCache   state.Database = database.NewStateCache(db)
		snaps        *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		cpuPre       float64
		cpuExe       float64
		avgPre       float64
		avgExe       float64
		avgMemPre    float64
		avgMemExe    float64
		totalMemUsed float64
	)

	// 新建原生数据库
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)

	// 构建预测用的内存结构
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(1, 1)
	preTable := vm.NewPreExecutionTable()

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		count++
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}
		if count == 1 {
			largeBlock = blk
		}
		combinedTxs = append(combinedTxs, blk.Transactions()...)

		for _, tx := range blk.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			tipMap[tx.Hash()] = tip
		}

		if count == 100 {
			largeBlock.AddTransactions(combinedTxs)
			preStateDB := stateDb.Copy()
			blockContext := core.NewEVMBlockContext(largeBlock.Header(), db, nil)
			newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, largeBlock, blockContext, config.MainnetChainConfig)
			newThread.SetChannels(txSource, disorder, txMap, tipChan, blockChan)

			// 创建交易分发客户端
			cli := client.NewFakeClient(txSource, disorder, txMap, tipChan, blockChan)

			wg.Add(4)
			go func() {
				defer wg.Done()
				cli.Run(db, 0, i.Int64(), 0, true, false, largeBlock, tipMap, 0.8)
			}()
			go func() {
				defer wg.Done()
				newThread.PreExecutionWithDisorder(true, true, true, true, false, nil)
			}()
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < 20; j++ {
					cpuUsed1 := GetCPUPercent2()
					sumUsed += cpuUsed1
				}
				avgPre = sumUsed / float64(20)
				cpuPre += avgPre
			}()
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < len(largeBlock.Transactions()); j++ {
					memUsed := GetMemPercent()
					sumUsed += memUsed
				}
				avgMemPre = sumUsed / float64(len(largeBlock.Transactions()))
			}()
			wg.Wait()

			wg.Add(2)
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < 10; j++ {
					cpuUsed1 := GetCPUPercent2()
					sumUsed += cpuUsed1
				}
				avgExe = sumUsed / float64(10)
				cpuExe += avgExe
			}()
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < len(largeBlock.Transactions()); j++ {
					memUsed := GetMemPercent()
					sumUsed += memUsed
				}
				avgMemExe = sumUsed / float64(len(largeBlock.Transactions()))
			}()

			_, _, _, _, _, _, err = newThread.FastExecution(stateDb, nil, true, true, true, false, false, "")
			if err != nil {
				fmt.Println("execution error", err)
			}
			wg.Wait()

			fmt.Fprintf(writer, "CPU utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), avgPre)
			fmt.Fprintf(writer, "CPU utilization in execution in block %d: %.2f%%\n", i.Int64(), avgExe)
			fmt.Fprintf(writer, "Memory utilization in pre-execution in block %d: %.2f%%\n", i.Int64(), avgMemPre)
			fmt.Fprintf(writer, "Memory utilization in execution in block %d: %.2f%%\n", i.Int64(), avgMemExe)
			totalMemUsed += (avgMemPre + avgMemExe) / float64(2)

			// Commit all cached state changes into underlying memory database.
			root, _ := stateDb.Commit(config.MainnetChainConfig.IsEIP158(largeBlock.Number()))

			//fmt.Printf("Ratio of satisfied branch dircetions: %.2f, ratio of satisfied txs: %.2f\n", branchRatio, txRatio)
			fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay large block number "+i.String(), root)

			count = 0
			combinedTxs = []*types.Transaction{}
			tipMap = make(map[common.Hash]*big.Int)
		}
	}

	fmt.Fprintln(writer)
	avgMem := totalMemUsed / float64(offset/100)
	avgCPUPre := cpuPre / float64(offset/100)
	avgCPUExe := cpuExe / float64(offset/100)
	fmt.Fprintf(writer, "Average memory utilization: %.2f%%\n", avgMem)
	fmt.Fprintf(writer, "CPU utilization in pre-execution: %.2f%%\n", avgCPUPre)
	fmt.Fprintf(writer, "CPU utilization in execution: %.2f%%\n", avgCPUExe)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestIOSeer evaluates the I/O overhead during the pre-execution and execution phase of Seer
func TestIOSeer(startingHeight, offset int64) error {
	file, err := os.OpenFile(exp_io_seer, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintln(writer, "===== Run Seer for I/O test =====")

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		largeBlock       *types.Block
		count            int
		combinedTxs      types.Transactions
		tipMap           = make(map[common.Hash]*big.Int)
		txSource         = make(chan []*types.Transaction)
		disorder         = make(chan *types.Transaction)
		txMap            = make(chan map[common.Hash]*types.Transaction)
		tipChan          = make(chan map[common.Hash]*big.Int)
		blockChan        = make(chan *types.Block)
		wg               sync.WaitGroup
		parent           *types.Header  = startBlock.Header()
		parentRoot       *common.Hash   = &parent.Root
		stateCache       state.Database = database.NewStateCache(db)
		snaps            *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		totalIOReadsPre  float64
		totalIOReadsExe  float64
		totalIOWritesPre float64
		totalIOWritesExe float64
		avgIOReadsPre    float64
		avgIOReadsExe    float64
		avgIOWritesPre   float64
		avgIOWritesExe   float64
		readCountPre     int
		writeCountPre    int
		readCountExe     int
		writeCountExe    int
	)

	// 新建原生数据库
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)
	pid := os.Getpid()

	// 构建预测用的内存结构
	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(20, 0.1)
	preTable := vm.NewPreExecutionTable()

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		count++
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}

		if count == 1 {
			largeBlock = blk
		}
		combinedTxs = append(combinedTxs, blk.Transactions()...)

		for _, tx := range blk.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), blk.BaseFee()))
			tipMap[tx.Hash()] = tip
		}

		if count == 100 {
			largeBlock.AddTransactions(combinedTxs)
			blockContext := core.NewEVMBlockContext(largeBlock.Header(), db, nil)
			newThread := core.NewThread(0, stateDb.Copy(), nil, nil, nil, nil, branchTable, mvCache, preTable, largeBlock, blockContext, config.MainnetChainConfig)
			newThread.SetChannels(txSource, disorder, txMap, tipChan, blockChan)

			// 创建交易分发客户端
			cli := client.NewFakeClient(txSource, disorder, txMap, tipChan, blockChan)

			wg.Add(3)
			go func() {
				defer wg.Done()
				cli.Run(db, 0, i.Int64(), 0, true, false, largeBlock, tipMap, 0.8)
			}()
			go func() {
				defer wg.Done()
				newThread.PreExecutionWithDisorder(true, true, true, true, false, nil)
			}()
			go func() {
				defer wg.Done()
				readsList, writesList, err3 := RunBpftrace(pid, 120*time.Second)
				if err3 != nil {
					fmt.Printf("error is: %v\n", err3)
				}
				IOReads, IOWrites := CalculateAverage(readsList, writesList)
				if IOReads > 0 {
					totalIOReadsPre += IOReads
					if IOReads > 500 {
						readCountPre++
					}
				}
				if IOWrites > 0 {
					totalIOWritesPre += IOWrites
					writeCountPre++
				}
				//fmt.Fprintf(writer, "I/O reads in block in pre-execution %d: %.2f\n", i.Int64(), IOReads)
				//fmt.Fprintf(writer, "I/O writes in block in pre-execution %d: %.2f\n", i.Int64(), IOWrites)
			}()
			wg.Wait()
			fmt.Println("Pre-execution ends")

			serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)
			serialProcessor.Process(largeBlock, stateDb.Copy(), vm.Config{EnablePreimageRecording: false}, nil, true)

			wg.Add(1)
			go func() {
				defer wg.Done()
				readsList, writesList, err3 := RunBpftrace(pid, 30*time.Second)
				if err3 != nil {
					fmt.Printf("error is: %v\n", err3)
				}
				IOReads, IOWrites := CalculateAverage(readsList, writesList)
				if IOReads > 0 {
					totalIOReadsExe += IOReads
					readCountExe++
				}
				if IOWrites > 0 {
					totalIOWritesExe += IOWrites
					if IOWrites > 100 {
						writeCountExe++
					}
				}
				//fmt.Fprintf(writer, "I/O reads in block in execution %d: %.2f\n", i.Int64(), IOReads)
				//fmt.Fprintf(writer, "I/O writes in block in execution %d: %.2f\n", i.Int64(), IOWrites)
			}()

			time.Sleep(5 * time.Second)
			_, _, _, _, _, _, err = newThread.FastExecution(stateDb, nil, true, true, true, false, false, "")
			if err != nil {
				fmt.Println("execution error", err)
			}
			root, _ := stateDb.Commit(config.MainnetChainConfig.IsEIP158(largeBlock.Number()))
			fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay large block number "+i.String(), root)
			// flush into the underlying database
			if err3 := stateDb.Database().TrieDB().Commit(root, true); err3 != nil {
				fmt.Printf("flush error is: %v\n", err3)
			}
			wg.Wait()

			count = 0
			combinedTxs = []*types.Transaction{}
			tipMap = make(map[common.Hash]*big.Int)
		}
	}

	fmt.Fprintln(writer)
	avgIOReadsPre = totalIOReadsPre / float64(readCountPre)
	avgIOWritesPre = totalIOWritesPre / float64(writeCountPre)
	avgIOReadsExe = totalIOReadsExe / float64(readCountExe)
	avgIOWritesExe = totalIOWritesExe / float64(writeCountExe)

	fmt.Fprintf(writer, "Average pre-execution I/O reads: %.2f\n", avgIOReadsPre)
	fmt.Fprintf(writer, "Average pre-execution I/O writes: %.2f\n", avgIOWritesPre)
	fmt.Fprintf(writer, "Average execution I/O reads: %.2f\n", avgIOReadsExe)
	fmt.Fprintf(writer, "Average execution I/O writes: %.2f\n", avgIOWritesExe)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestMemoryCPUBaseline evaluates the memory cost during the native Ethereum serial execution
func TestMemoryCPUBaseline(startingHeight, offset int64) error {
	file, err := os.OpenFile(exp_memorycpu_usage_eth, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintln(writer, "===== Run Ethereum for memory and cpu test =====")

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		largeBlock   *types.Block
		count        int
		combinedTxs  types.Transactions
		parent       *types.Header  = startBlock.Header()
		parentRoot   *common.Hash   = &parent.Root
		stateCache   state.Database = database.NewStateCache(db)
		snaps        *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		totalMaxCPU  float64
		totalAvgCPU  float64
		avgMaxCPU    float64
		avgAvgCPU    float64
		totalMemUsed float64
		avgMemUsed   float64
		wg           sync.WaitGroup
	)

	serialDB, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		count++
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}

		if count == 1 {
			largeBlock = blk
		}
		combinedTxs = append(combinedTxs, blk.Transactions()...)

		if count == 100 {
			largeBlock.AddTransactions(combinedTxs)

			wg.Add(2)
			go func() {
				defer wg.Done()
				var maxSum, avgSum float64
				for j := 0; j < 10; j++ {
					avgUsed := GetCPUPercent()
					maxUsed := GetCPUPercent2()
					avgSum += avgUsed
					maxSum += maxUsed
				}
				avgMaxCPU = maxSum / float64(10)
				avgAvgCPU = avgSum / float64(10)
				totalMaxCPU += avgMaxCPU
				totalAvgCPU += avgAvgCPU
			}()
			go func() {
				defer wg.Done()
				var sumUsed float64
				for j := 0; j < len(largeBlock.Transactions()); j++ {
					memUsed := GetMemPercent()
					sumUsed += memUsed
				}
				avgMemUsed = sumUsed / float64(len(largeBlock.Transactions()))
				totalMemUsed += avgMemUsed
			}()

			serialProcessor.Process(largeBlock, serialDB, vm.Config{EnablePreimageRecording: false}, nil, true)
			wg.Wait()

			fmt.Fprintf(writer, "Max CPU utilization in execution in block %d: %.2f%%\n", i.Int64(), avgMaxCPU)
			fmt.Fprintf(writer, "Average CPU utilization in execution in block %d: %.2f%%\n", i.Int64(), avgAvgCPU)
			fmt.Fprintf(writer, "Memory utilization in block %d: %.2f%%\n", i.Int64(), avgMemUsed)

			root, _ := serialDB.Commit(config.MainnetChainConfig.IsEIP158(largeBlock.Number()))
			fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay large block number "+i.String(), root)

			count = 0
			combinedTxs = []*types.Transaction{}
		}
	}

	fmt.Fprintln(writer)
	avgMem := totalMemUsed / float64(offset/100)
	avgMaxCPU = totalMaxCPU / float64(offset/100)
	avgAvgCPU = totalAvgCPU / float64(offset/100)
	fmt.Fprintf(writer, "Average memory utilization: %.2f%%\n", avgMem)
	fmt.Fprintf(writer, "Max CPU utilization in execution: %.2f%%\n", avgMaxCPU)
	fmt.Fprintf(writer, "Average CPU utilization in execution: %.2f%%\n", avgAvgCPU)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestIOBaseline evaluates the I/O overhead during the native Ethereum serial execution
func TestIOBaseline(startingHeight, offset int64) error {
	file, err := os.OpenFile(exp_io_eth, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	fmt.Fprintln(writer, "===== Run Ethereum for I/O test =====")

	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	startBlock, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		largeBlock    *types.Block
		count         int
		combinedTxs   types.Transactions
		parent        *types.Header  = startBlock.Header()
		parentRoot    *common.Hash   = &parent.Root
		stateCache    state.Database = database.NewStateCache(db)
		snaps         *snapshot.Tree = database.NewSnap(db, stateCache, startBlock.Header())
		totalIOReads  float64
		totalIOWrites float64
		avgIOReads    float64
		avgIOWrites   float64
		writeCount    int
		readCount     int
		wg            sync.WaitGroup
	)

	pid := os.Getpid()
	serialDB, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		count++
		blk, err2 := database.GetBlockByNumber(db, i)
		if err2 != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err2)
		}

		if count == 1 {
			largeBlock = blk
		}
		combinedTxs = append(combinedTxs, blk.Transactions()...)

		if count == 100 {
			largeBlock.AddTransactions(combinedTxs)

			wg.Add(1)
			go func() {
				defer wg.Done()
				readsList, writesList, err3 := RunBpftrace(pid, 150*time.Second)
				if err3 != nil {
					fmt.Printf("error is: %v\n", err3)
				}
				IOReads, IOWrites := CalculateAverage(readsList, writesList)
				if IOReads > 0 {
					totalIOReads += IOReads
					readCount++
				}
				if IOWrites > 0 {
					totalIOWrites += IOWrites
					if IOWrites > 100 {
						writeCount++
					}
				}
				//fmt.Fprintf(writer, "I/O reads in block %d: %.2f\n", i.Int64(), IOReads)
				//fmt.Fprintf(writer, "I/O writes in block %d: %.2f\n", i.Int64(), IOWrites)
			}()

			serialProcessor.Process(largeBlock, serialDB, vm.Config{EnablePreimageRecording: false}, nil, true)
			root, _ := serialDB.Commit(config.MainnetChainConfig.IsEIP158(largeBlock.Number()))
			fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay large block number "+i.String(), root)
			// flush into the underlying database
			if err3 := serialDB.Database().TrieDB().Commit(root, true); err3 != nil {
				fmt.Printf("flush error is: %v\n", err3)
			}
			wg.Wait()

			count = 0
			combinedTxs = []*types.Transaction{}
		}
	}

	fmt.Fprintln(writer)
	avgIOReads = totalIOReads / float64(readCount)
	avgIOWrites = totalIOWrites / float64(writeCount)
	avgIOOperations := avgIOReads + avgIOWrites

	fmt.Fprintf(writer, "Average I/O reads: %.2f\n", avgIOReads)
	fmt.Fprintf(writer, "Average I/O writes: %.2f\n", avgIOWrites)
	fmt.Fprintf(writer, "Average I/O operations: %.2f\n", avgIOOperations)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}

// TestGasUsed evaluates the distribution of gas used by transactions
func TestGasUsed(startingHeight, offset int64) error {
	db, err := database.OpenDatabaseWithFreezer(&config.DefaultsEthConfig)
	if err != nil {
		return fmt.Errorf("open leveldb error: %s", err)
	}
	defer db.Close()

	blockPre, err := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
	if err != nil {
		return fmt.Errorf("function GetBlockByNumber error: %s", err)
	}

	var (
		parent     *types.Header = blockPre.Header()
		parentRoot *common.Hash  = &parent.Root
		// 该state.Database接口的具体类型为state.cachingDB，其中的disk字段为db
		stateCache       state.Database = database.NewStateCache(db)
		snaps            *snapshot.Tree = database.NewSnap(db, stateCache, blockPre.Header())
		totalCTxs        int
		totalCTXsGasUsed uint64
		totalNTXsGasUsed uint64
	)

	// 新建原生数据库
	serialDB, _ := state.New(*parentRoot, stateCache, snaps)
	serialProcessor := core.NewStateProcessor(config.MainnetChainConfig, db)
	// 保存used gas分布
	gasDistribution := make(map[string]int)
	// '0' --> <21000, '1' --> 21000-60000, '2' --> 60000-100000, '3' --> >=100000
	txGasDistribution := make(map[common.Hash]int)
	txGasMap := make(map[common.Hash]uint64)
	recorder := core.NewRecorder(true)

	min, max, addSpan := big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		block, err := database.GetBlockByNumber(db, i)
		if err != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err)
		}

		_, _, receipts, _, _, _ := serialProcessor.Process(block, serialDB, vm.Config{EnablePreimageRecording: false}, nil, false)
		root, _ := serialDB.Commit(config.MainnetChainConfig.IsEIP158(block.Number()))
		fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "serially replay block number "+i.String(), root)

		for j, receipt := range receipts {
			address := block.Transactions()[j].To()
			if address != nil && core.IsContract(serialDB, address) {
				txHash := receipt.TxHash
				gasUsed := receipt.GasUsed
				txGasMap[txHash] = gasUsed
				if int(gasUsed) < 21000 {
					txGasDistribution[txHash] = 0
					gasDistribution["<21000"]++
				} else if int(gasUsed) < 60000 {
					txGasDistribution[txHash] = 1
					gasDistribution["21000-60000"]++
				} else if int(gasUsed) < 100000 {
					txGasDistribution[txHash] = 2
					gasDistribution["60000-100000"]++
				} else {
					txGasDistribution[txHash] = 3
					gasDistribution[">=100000"]++
				}
			}
		}

		// reset stateDB every epoch to remove accumulated I/O overhead
		snaps = database.NewSnap(db, stateCache, block.Header())
		serialDB, _ = state.New(root, stateCache, snaps)
	}

	branchTable := vm.CreateNewTable()
	mvCache := state.NewMVCache(10, 0.1)
	preTable := vm.NewPreExecutionTable()
	stateDb, _ := state.New(*parentRoot, stateCache, snaps)

	min, max, addSpan = big.NewInt(startingHeight+1), big.NewInt(startingHeight+offset+1), big.NewInt(1)
	for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
		block, err := database.GetBlockByNumber(db, i)
		if err != nil {
			return fmt.Errorf("get block %s error: %s", i.String(), err)
		}

		// 创建用于预测执行的stateDB
		preStateDB := stateDb.Copy()
		// EVM执行所需要的区块上下文，一次性生成且后续不能改动
		blockContext := core.NewEVMBlockContext(block.Header(), db, nil)
		newThread := core.NewThread(0, preStateDB, nil, nil, nil, nil, branchTable, mvCache, preTable, block, blockContext, config.MainnetChainConfig)

		txSet := make(map[common.Hash]*types.Transaction)
		tipMap := make(map[common.Hash]*big.Int)
		for _, tx := range block.Transactions() {
			tip := math2.BigMin(tx.GasTipCap(), new(big.Int).Sub(tx.GasFeeCap(), block.BaseFee()))
			txSet[tx.Hash()] = tx
			tipMap[tx.Hash()] = tip
		}

		newThread.PreExecution(txSet, tipMap, true, true, true, txGasDistribution)
		_, _, _, _, ctxNum, _, err := newThread.FastExecution(stateDb, recorder, false, true, true, true, false, exp_branch_statistics)
		if err != nil {
			fmt.Println("execution error", err)
		}
		// Commit all cached state changes into underlying memory database.
		root, _ := stateDb.Commit(config.MainnetChainConfig.IsEIP158(block.Number()))

		fmt.Println("["+time.Now().Format("2006-01-02 15:04:05")+"]", "successfully replay block number "+i.String(), root)

		// reset stateDB every epoch to remove accumulated I/O overhead
		snaps = database.NewSnap(db, stateCache, block.Header())
		stateDb, _ = state.New(root, stateCache, snaps)
		totalCTxs += ctxNum
	}

	file, err := os.OpenFile(exp_branch_statistics, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)

	ratio1 := float64(gasDistribution["<21000"]) / float64(totalCTxs)
	fmt.Fprintf(writer, "<21000: %.2f\n", ratio1)
	ratio2 := float64(gasDistribution["21000-60000"]) / float64(totalCTxs)
	fmt.Fprintf(writer, "21000-60000: %.2f\n", ratio2)
	ratio3 := float64(gasDistribution["60000-100000"]) / float64(totalCTxs)
	fmt.Fprintf(writer, "60000-100000: %.2f\n", ratio3)
	ratio4 := float64(gasDistribution[">=100000"]) / float64(totalCTxs)
	fmt.Fprintf(writer, ">=100000: %.2f\n", ratio4)

	stateRelatedForCTxs0 := branchTable.StateRelatedForCTxs0
	stateRelatedForCTxs1 := branchTable.StateRelatedForCTxs1
	stateRelatedForCTxs2 := branchTable.StateRelatedForCTxs2
	stateRelatedForCTxs3 := branchTable.StateRelatedForCTxs3
	txNumInCTxs0 := gasDistribution["<21000"]
	txNumInCTxs1 := gasDistribution["21000-60000"]
	txNumInCTxs2 := gasDistribution["60000-100000"]
	txNumInCTxs3 := gasDistribution[">=100000"]
	fmt.Fprintf(writer, "The average number of state variable-related branches each tx (<21000) is: %.2f\n", float64(stateRelatedForCTxs0)/float64(txNumInCTxs0))
	fmt.Fprintf(writer, "The average number of state variable-related branches each tx (21000-60000) is: %.2f\n", float64(stateRelatedForCTxs1)/float64(txNumInCTxs1))
	fmt.Fprintf(writer, "The average number of state variable-related branches each tx (60000-100000) is: %.2f\n", float64(stateRelatedForCTxs2)/float64(txNumInCTxs2))
	fmt.Fprintf(writer, "The average number of state variable-related branches each tx (>=100000) is: %.2f\n", float64(stateRelatedForCTxs3)/float64(txNumInCTxs3))

	txNum := len(txGasMap)
	complicatedTXs := recorder.GetComplicatedMap()
	ctxNum := len(complicatedTXs)
	for txHash, gas := range txGasMap {
		if _, ok := complicatedTXs[txHash]; ok {
			totalCTXsGasUsed += gas
		} else {
			totalNTXsGasUsed += gas
		}
	}
	ntxRatio := float64(txNum-ctxNum) / float64(txNum)
	ctxRatio := float64(ctxNum) / float64(txNum)
	ntxAvgGas := float64(totalNTXsGasUsed) / float64(txNum-ctxNum)
	ctxAvgGas := float64(totalCTXsGasUsed) / float64(ctxNum)
	fmt.Fprintf(writer, "The average number of state variable-related branches for complicated TXs is: %.2f, the ratio is: %.2f\n", ctxAvgGas, ctxRatio)
	fmt.Fprintf(writer, "The average number of state variable-related branches for normal TXs is: %.2f, the ratio is: %.2f\n", ntxAvgGas, ntxRatio)

	err = writer.Flush()
	if err != nil {
		fmt.Printf("flush error: %v\n", err)
		return nil
	}

	return nil
}
