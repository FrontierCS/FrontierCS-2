package client

import (
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethdb"
	"math/big"
	"math/rand"
	"seerEVM/core/types"
	"seerEVM/database"
	"sort"
	"time"
)

// FakeClient 提供新交易
type FakeClient struct {
	TxSource    chan<- []*types.Transaction
	Disorder    chan<- *types.Transaction
	TxMap       chan<- map[common.Hash]*types.Transaction
	Tip         chan<- map[common.Hash]*big.Int
	Block       chan<- *types.Block
	Txs         []*types.Transaction
	alreadySend map[string]struct{}
}

func NewFakeClient(txSource chan<- []*types.Transaction, disorder chan<- *types.Transaction, txMap chan<- map[common.Hash]*types.Transaction, tip chan<- map[common.Hash]*big.Int, block chan<- *types.Block) *FakeClient {
	fc := new(FakeClient)
	fc.TxSource = txSource
	fc.Disorder = disorder
	fc.TxMap = txMap
	fc.Tip = tip
	fc.Block = block
	fc.alreadySend = make(map[string]struct{})
	return fc
}

func (fc *FakeClient) generateTxs(db ethdb.Database, txNum int, startingHeight, offset int64, largeBlock bool) (*types.Block, error) {
	if largeBlock {
		txLen := 0
		min, max, addSpan := big.NewInt(startingHeight), big.NewInt(startingHeight+offset), big.NewInt(1)
		for i := min; i.Cmp(max) == -1; i = i.Add(i, addSpan) {
			block, err2 := database.GetBlockByNumber(db, i)
			if err2 != nil {
				return nil, fmt.Errorf("get block %s error: %s", i.String(), err2)
			}
			newTxs := block.Transactions()
			if txLen+len(newTxs) >= txNum {
				for j := 0; j < txNum-txLen; j++ {
					fc.Txs = append(fc.Txs, newTxs[j])
				}
				break
			}
			fc.Txs = append(fc.Txs, newTxs...)
			txLen += len(newTxs)
		}
		startBlock, _ := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
		startBlock.AddTransactions(fc.Txs)
		fc.Block <- startBlock
		return startBlock, nil
	} else {
		block, err2 := database.GetBlockByNumber(db, new(big.Int).SetInt64(startingHeight))
		if err2 != nil {
			return nil, fmt.Errorf("get block %s error: %s", big.NewInt(startingHeight).String(), err2)
		}
		for _, tx := range block.Transactions() {
			fc.Txs = append(fc.Txs, tx)
		}
		fc.Block <- block
		return block, nil
	}
}

func (fc *FakeClient) Run(db ethdb.Database, txNum int, startingHeight, offset int64, directlyUsed, largeBlock bool, blk *types.Block, tipMap map[common.Hash]*big.Int, ratio float64) *types.Block {
	// 将交易分为两部分，一部分是先发送的，以ratio概率随机选择一部分交易滞后发送，模拟乱序过程
	var newTxs []*types.Transaction
	var sortedTxs []*types.Transaction
	var disorderTxs []*types.Transaction
	var txMap = make(map[common.Hash]*types.Transaction)
	var sentDisorderedTxs = make(map[int]struct{})
	var sentNum int

	if !directlyUsed {
		blk, _ = fc.generateTxs(db, txNum, startingHeight, offset, largeBlock)
	} else {
		fc.Block <- blk
		fc.Txs = blk.Transactions()
	}

	for _, tx := range fc.Txs {
		// avoid some txs that have been sent
		_, exist := fc.alreadySend[tx.Hash().String()]
		if exist {
			panic("already exist!")
		} else {
			fc.alreadySend[tx.Hash().String()] = struct{}{}
		}
		txMap[tx.Hash()] = tx

		source := rand.NewSource(time.Now().UnixNano())
		r := rand.New(source)
		if r.Float64() < ratio {
			disorderTxs = append(disorderTxs, tx)
		} else {
			newTxs = append(newTxs, tx)
		}
	}

	// firstly sort txs in newTxs
	for i, tx := range newTxs {
		tip := tipMap[tx.Hash()]
		if i == 0 {
			sortedTxs = append(sortedTxs, tx)
			continue
		}
		insertLoc := sort.Search(len(sortedTxs), func(j int) bool {
			comparedTip := tipMap[sortedTxs[j].Hash()]
			if tip.Cmp(comparedTip) > 0 {
				return true
			} else {
				return false
			}
		})
		sortedTxs = append(sortedTxs[:insertLoc], append([]*types.Transaction{tx}, sortedTxs[insertLoc:]...)...)
	}

	fc.TxMap <- txMap
	fc.Tip <- tipMap
	fc.TxSource <- sortedTxs

	time.Sleep(100 * time.Millisecond)

	for {
		// 将disorder乱序
		if sentNum == len(disorderTxs) {
			break
		}
		source := rand.NewSource(time.Now().UnixNano())
		r := rand.New(source)
		randomIndex := r.Intn(len(disorderTxs))
		if _, ok := sentDisorderedTxs[randomIndex]; !ok {
			sentDisorderedTxs[randomIndex] = struct{}{}
		} else {
			continue
		}
		disorderTx := disorderTxs[randomIndex]
		fc.Disorder <- disorderTx
		sentNum++
		time.Sleep(time.Millisecond)
	}

	return blk
}
