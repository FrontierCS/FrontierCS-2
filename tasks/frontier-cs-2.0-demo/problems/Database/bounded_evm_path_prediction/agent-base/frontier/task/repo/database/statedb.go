package database

import (
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/log"
	"seerEVM/core/rawdb"
	"seerEVM/core/state"
	"seerEVM/core/state/snapshot"
	"seerEVM/core/types"
	"seerEVM/trie"
)

func NewStateCache(db ethdb.Database) state.Database {
	config := defaultStateDBConfig()
	return state.NewDatabaseWithConfig(db, &trie.Config{
		Cache:     config.Cache,
		Journal:   config.Journal,
		Preimages: config.Preimages,
	})
}

func NewSnap(db ethdb.Database, stateCache state.Database, header *types.Header) *snapshot.Tree {
	var recover bool

	if layer := rawdb.ReadSnapshotRecoveryNumber(db); layer != nil && *layer > header.Number.Uint64() {
		log.Warn("Enabling snapshot recovery", "chainhead", header.Number.Uint64(), "diskbase", *layer)
		recover = true
	}
	snapconfig := snapshot.Config{
		CacheSize:  256,
		Recovery:   recover,
		NoBuild:    true,
		AsyncBuild: false,
	}

	snaps, _ := snapshot.New(snapconfig, db, stateCache.TrieDB(), header.Root)
	return snaps
}
