package dependencyGraph

import (
	"github.com/ethereum/go-ethereum/common"
	"seerEVM/core/types"
	"seerEVM/core/vm"
)

type DependencyGraph map[int]int

type RecorderWithTxID struct {
	r    *vm.Recorder
	TxID int
}

// ConstructDependencyGraph 输入所有交易的读写集，切片长度代表了交易数量，交易从0开始计数
// 本函数在运行时，外部不应该修改传入的读写集，否则引起竞态
// 输出：map类型，交易id->交易依赖的最大交易id
func ConstructDependencyGraph(readMap map[common.Hash]vm.ReadSet, writeMap map[common.Hash]vm.WriteSet, txs []*types.Transaction) DependencyGraph {
	var (
		readSets  = make(map[int]vm.ReadSet)
		writeSets = make(map[int]vm.WriteSet)
		dg        = make(map[int]int)
		nezha     = make(map[common.Address][]RecorderWithTxID) // 合约地址->在该地址上有写操作的所有交易
		assist    []int
	)

	for i, tx := range txs {
		rs, ok := readMap[tx.Hash()]
		if ok {
			readSets[i] = rs
		}
		ws, ok2 := writeMap[tx.Hash()]
		if ok2 {
			writeSets[i] = ws
			assist = append(assist, i)
		}
	}

	// 1)遍历每个交易的写集，放入nezha中
	for _, index := range assist {
		ws := writeSets[index]
		for addr, reco := range ws {
			nezha[addr] = append(nezha[addr], RecorderWithTxID{
				r:    reco,
				TxID: index,
			})
		}
	}

	// 2)遍历每个交易的读集，查询该交易依赖的最大交易
	n := len(txs)
	for i := 0; i < n; i++ {
		dependencyNum := -1
		rs, ok := readSets[i]
		if ok {
			// 该交易的读集里可能有多个读操作，找到每个读操作依赖的最大交易序号，然后取最大
			for addr, readSet := range rs {
				dn := -1 // 该读操作依赖的最大交易序号
				for _, r := range nezha[addr] {
					if r.TxID >= i {
						break
					}
					// 两者都是访问state，构成依赖
					if readSet.IsStateAccessed() && r.r.IsStateAccessed() {
						dn = r.TxID
					}
					// 两者都是访问slot，且访问同一个slot，构成依赖
					if readSet.IsStorageAccessed() && r.r.IsStorageAccessed() {
						slots1 := readSet.GetSlots()
						slots2 := r.r.GetSlots()
						for s, _ := range slots1 {
							_, exist := slots2[s]
							if exist {
								dn = r.TxID
								break
							}
						}
					}
				}
				if dn > dependencyNum {
					dependencyNum = dn
				}
			}
		}
		dg[i] = dependencyNum
	}

	return dg
}

// NewSimpleDependencyGraph 用于测试，输出一个依赖全为-1的依赖图，也即所有交易无冲突
func NewSimpleDependencyGraph(num int) DependencyGraph {
	dp := make(DependencyGraph)
	for i := 0; i < num; i++ {
		dp[i] = -1
	}
	return dp
}
