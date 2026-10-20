package database

import (
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethdb"
	"math/big"
	"seerEVM/config"
	"seerEVM/core/rawdb"
	"seerEVM/core/types"
)

// OpenDatabaseWithFreezer 新建geth提供的raw database并不加封装地返回，改动path和ancient
func OpenDatabaseWithFreezer(ethConfig *config.EthConfig) (ethdb.Database, error) {
	rawConfig := DefaultRawConfig()

	if ethConfig.NoPruning && ethConfig.TrieDirtyCache > 0 {
		if ethConfig.SnapshotCache > 0 {
			ethConfig.TrieCleanCache += ethConfig.TrieDirtyCache * 3 / 5
			ethConfig.SnapshotCache += ethConfig.TrieDirtyCache * 2 / 5
		} else {
			ethConfig.TrieCleanCache += ethConfig.TrieDirtyCache
		}
		ethConfig.TrieDirtyCache = 0
	}

	db, err := rawdb.Open(rawdb.OpenOptions{
		Type:              "",
		Directory:         rawConfig.Path,
		AncientsDirectory: rawConfig.Ancient,
		Namespace:         rawConfig.Namespace,
		Cache:             ethConfig.DatabaseCache,
		Handles:           rawConfig.Handles,
		ReadOnly:          rawConfig.ReadOnly,
	})
	return db, err
}

// OpenDatabaseWithFreezer2 uses a new chaindata and ancient path
func OpenDatabaseWithFreezer2(ethConfig *config.EthConfig) (ethdb.Database, error) {
	rawConfig := &RawConfig{
		Path:      "../ethereumdata_1465_small/copychain2",
		Cache:     2048,
		Handles:   5120,
		Ancient:   "../ethereumdata_1465_small/copychain2/ancient",
		Namespace: "eth/db/chaindata/",
		ReadOnly:  false,
	}

	if ethConfig.NoPruning && ethConfig.TrieDirtyCache > 0 {
		if ethConfig.SnapshotCache > 0 {
			ethConfig.TrieCleanCache += ethConfig.TrieDirtyCache * 3 / 5
			ethConfig.SnapshotCache += ethConfig.TrieDirtyCache * 2 / 5
		} else {
			ethConfig.TrieCleanCache += ethConfig.TrieDirtyCache
		}
		ethConfig.TrieDirtyCache = 0
	}

	db, err := rawdb.Open(rawdb.OpenOptions{
		Type:              "",
		Directory:         rawConfig.Path,
		AncientsDirectory: rawConfig.Ancient,
		Namespace:         rawConfig.Namespace,
		Cache:             ethConfig.DatabaseCache,
		Handles:           rawConfig.Handles,
		ReadOnly:          rawConfig.ReadOnly,
	})
	return db, err
}

// GetBlockByNumber 从raw database中获取第i个区块
func GetBlockByNumber(db ethdb.Database, number *big.Int) (*types.Block, error) {
	var (
		block *types.Block
		err   error
	)
	hash := rawdb.ReadCanonicalHash(db, number.Uint64()) // 获取区块hash
	if (hash != common.Hash{}) {
		block = rawdb.ReadBlock(db, hash, number.Uint64())
		if block == nil {
			err = fmt.Errorf("read block(" + number.String() + ") error! block is nil")
		}
	} else {
		err = fmt.Errorf("read block(" + number.String() + ") error! hash is nil")
	}
	return block, err
}
