package state

import (
	"bytes"
	"fmt"
	"io"
	"math/big"
	"sort"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/metrics"
	"github.com/ethereum/go-ethereum/rlp"
	"seerEVM/core/types"
	"seerEVM/trie"
)

// TxInfoMini defines a version of tx execution
type TxInfoMini struct {
	Index       int
	Incarnation int
}

func (t *TxInfoMini) Copy() *TxInfoMini {
	return &TxInfoMini{
		Index:       t.Index,
		Incarnation: t.Incarnation,
	}
}

func (t *TxInfoMini) Compare(comparedVer *TxInfoMini) bool {
	if t.Index == comparedVer.Index && t.Incarnation == comparedVer.Incarnation {
		return true
	}
	return false
}

type SSlot struct {
	Value  common.Hash
	TxInfo *TxInfoMini
}

func newSSlot(value common.Hash, txInfo *TxInfoMini) *SSlot {
	return &SSlot{
		Value:  value,
		TxInfo: txInfo.Copy(),
	}
}

func (slot *SSlot) Copy() *SSlot {
	return &SSlot{
		Value:  common.BytesToHash(common.CopyBytes(slot.Value.Bytes())),
		TxInfo: slot.TxInfo.Copy(),
	}
}

type Slot struct {
	Value []*SSlot
	len   int
}

// newSlot, 初始化为从leveldb中读取
func newEmptySlot() *Slot {
	return &Slot{
		Value: make([]*SSlot, 0),
		len:   0,
	}
}

// newSlot, 初始化为从leveldb中读取
func newSlot(value common.Hash) *Slot {
	s := newEmptySlot()
	// 初始化为 -1, 无效为-2
	s.Value = append(s.Value, newSSlot(value, &TxInfoMini{Index: -1, Incarnation: -1}))
	s.len += 1
	return s
}

func (s *Slot) Copy() *Slot {
	cpSlot := newEmptySlot()
	for _, value := range s.Value {
		cpSlot.Value = append(cpSlot.Value, &SSlot{Value: value.Value, TxInfo: value.TxInfo.Copy()})
	}
	cpSlot.len = s.len
	return cpSlot
}

type StmStorage map[common.Hash]common.Hash

func (s StmStorage) String() (str string) {
	for key, value := range s {
		str += fmt.Sprintf("%X : %X\n", key, value)
	}
	return
}

func (s StmStorage) Copy() StmStorage {
	cpy := make(StmStorage, len(s))
	for key, value := range s {
		cpy[key] = value
	}
	return cpy
}

type SStateAccount struct {
	StateAccount *types.StateAccount
	Code         []byte
	// Cache flags.
	// When an object is marked suicided it will be deleted from the trie
	// during the "update" phase of the state transition.
	dirtyCode bool // true if the code was updated
	suicided  bool
	deleted   bool
}

func newSStateAccount(data *types.StateAccount, code []byte, dirtyCode, suicided, deleted bool) *SStateAccount {
	return &SStateAccount{
		StateAccount: data.Copy(),
		Code:         common.CopyBytes(code),
		dirtyCode:    dirtyCode,
		suicided:     suicided,
		deleted:      deleted,
	}
}

func (ssa *SStateAccount) Copy() *SStateAccount {
	return &SStateAccount{
		StateAccount: ssa.StateAccount.Copy(),
		Code:         common.CopyBytes(ssa.Code),
		dirtyCode:    ssa.dirtyCode,
		suicided:     ssa.suicided,
		deleted:      ssa.deleted,
	}
}

type StmStateAccount struct {
	StateAccount []*SStateAccount
	len          int
}

func newEmptyStmStateAccount() *StmStateAccount {
	return &StmStateAccount{
		StateAccount: make([]*SStateAccount, 0),
		len:          0,
	}
}

// newStateAccount 新建一个空的，或者是从leveldb中读取
func newStateAccount(data *types.StateAccount, statedb *SeerStateDB, addrHash common.Hash) *SStateAccount {
	// sa := newEmptyStmStateAccount()
	stateAccount := newSStateAccount(data, nil, false, false, false)
	// 该地址为合约地址，则需提取Code
	if !bytes.Equal(data.CodeHash, types.EmptyCodeHash.Bytes()) {
		code, err := statedb.db.ContractCode(addrHash, common.BytesToHash(data.CodeHash))
		if err != nil {
			statedb.setError(fmt.Errorf("can't load code hash %x: %v", data.CodeHash, err))
		}
		stateAccount.Code = common.CopyBytes(code)
	}
	return stateAccount
}

//func (ssa *StmStateAccount) Copy() *StmStateAccount {
//	cpStmStateAccount := &StmStateAccount{
//		StateAccount: make([]SStateAccount, 0),
//		len:          ssa.len,
//	}
//	for _, value := range ssa.StateAccount {
//		var code []byte = nil
//		if value.Code != nil {
//			code = common.CopyBytes(value.Code)
//		}
//		stateAccount := SStateAccount{
//			StateAccount: value.StateAccount,
//			Code:         code,
//			dirtyCode:    false,
//			suicided:     false,
//			deleted:      false,
//			TxInfo:       value.TxInfo.Copy(),
//		}
//		cpStmStateAccount.StateAccount = append(cpStmStateAccount.StateAccount, stateAccount)
//	}
//	return cpStmStateAccount
//}

// stmStateObject represents an Ethereum account which is being modified.
//
// The usage pattern is as follows:
// First you need to obtain a state object.
// Account values can be accessed and modified through the object.
// Finally, call commitTrie to write the modified storage trie into a database.
// 多版本的obj
type stmStateObject struct { // 多版本的object
	address  common.Address
	addrHash common.Hash    // hash of ethereum address of the account
	data     *SStateAccount // committed account data (latest version)
	db       *SeerStateDB

	// Write caches.
	trie Trie // storage trie, which becomes non-nil on first access

	multiVersionState   *stateMap
	multiVersionStorage map[common.Hash]*slotMap
	mvStateMutex        sync.RWMutex
	mvStorageMutex      sync.RWMutex

	dirtyStorage StmStorage // StmStorage entries that have been modified in the current transaction execution
	dirtyMarker  bool       // dirtyMarker indicates that whether this object has been modified
}

type stateOperation struct {
	incarnation int
	account     *SStateAccount
}

type storageOperation struct {
	incarnation  int
	storageValue common.Hash
}

type stateMap struct {
	ops     map[int]*stateOperation
	ordered []int // record the ordered version list
}

type slotMap struct {
	ops     map[int]*storageOperation
	ordered []int // record the ordered version list
}

// empty returns whether the account is considered empty.
// 没有 或 最后一个为空
func (s *stmStateObject) empty() bool {
	stateAccount := s.data.StateAccount
	return stateAccount.Nonce == 0 && stateAccount.Balance.Sign() == 0 && bytes.Equal(stateAccount.CodeHash, types.EmptyCodeHash.Bytes())
}

// newStmStateObject creates a state object, 开始读读时候会新建
func newStmStateObject(db *SeerStateDB, address common.Address, data types.StateAccount) *stmStateObject {
	//fmt.Println(address, data.Balance)
	if data.Balance == nil {
		data.Balance = new(big.Int)
	}
	if data.CodeHash == nil {
		data.CodeHash = types.EmptyCodeHash.Bytes()
	}
	if data.Root == (common.Hash{}) {
		data.Root = types.EmptyRootHash
	}

	return &stmStateObject{
		db:       db,
		address:  address,
		addrHash: crypto.Keccak256Hash(address[:]),
		data:     newStateAccount(&data, db, crypto.Keccak256Hash(address[:])),
		multiVersionState: &stateMap{
			ops:     make(map[int]*stateOperation),
			ordered: make([]int, 0, 5000),
		},
		multiVersionStorage: make(map[common.Hash]*slotMap),
		dirtyStorage:        make(StmStorage),
		dirtyMarker:         false,
	}
}

func createStmStateObject(db *SeerStateDB, address common.Address) *stmStateObject {
	return &stmStateObject{
		db:       db,
		address:  address,
		addrHash: crypto.Keccak256Hash(address[:]),
		data:     new(SStateAccount),
		multiVersionState: &stateMap{
			ops:     make(map[int]*stateOperation),
			ordered: make([]int, 0, 5000),
		},
		multiVersionStorage: make(map[common.Hash]*slotMap),
		dirtyStorage:        make(StmStorage),
		dirtyMarker:         false,
	}
}

// EncodeRLP implements rlp.Encoder.
func (s *stmStateObject) EncodeRLP(w io.Writer) error {
	return rlp.Encode(w, s.data.StateAccount)
}

// getTrie returns the associated storage trie. The trie will be opened
// if it's not loaded previously. An error will be returned if trie can't
// be loaded.
func (s *stmStateObject) getTrie(db Database) (Trie, error) {
	stateAccount := s.data.StateAccount // 选择最后一个stateAccount
	if stateAccount != nil {
		if s.trie == nil {
			// Try fetching from prefetcher first
			// We don't prefetch empty tries
			if stateAccount.Root != types.EmptyRootHash && s.db.prefetcher != nil {
				// When the miner is creating the pending state, there is no
				// prefetcher
				s.trie = s.db.prefetcher.trie(s.addrHash, stateAccount.Root)
			}
			if s.trie == nil {
				tr, err := db.OpenStorageTrie(s.db.originalRoot, s.addrHash, stateAccount.Root)
				if err != nil {
					return nil, err
				}
				s.trie = tr
			}
		}
	} else {
		return nil, nil
	}
	return s.trie, nil
}

// GetState retrieves a value from the account storage trie.
//func (s *stmStateObject) GetState(db Database, key common.Hash, txIndex, txIncarnation int) *SSlot {
//	// If we have a dirty value for this state entry, return it
//	slot, dirty := s.dirtyStorage[key]
//	if dirty {
//		return slot.Value[slot.len-1].Copy()
//	}
//	// Otherwise return the entry's original value, 需要从db中获取
//	return s.GetCommittedState(db, key, txIndex, txIncarnation)
//}

// setState stores the written state account data into MVMemory
func (s *stmStateObject) setState(index, incarnation int, writeSet *WriteSet) {
	s.mvStateMutex.Lock()
	defer s.mvStateMutex.Unlock()
	op := &stateOperation{
		incarnation: incarnation,
		account:     writeSet.Account,
	}
	s.multiVersionState.ops[index] = op
	s.multiVersionState.ordered = append(s.multiVersionState.ordered, index)
	sort.Ints(s.multiVersionState.ordered)
}

// setStorage stores the written storage slots into MVMemory
func (s *stmStateObject) setStorage(index, incarnation int, writeSet *WriteSet) {
	s.mvStorageMutex.Lock()
	defer s.mvStorageMutex.Unlock()
	for key, value := range writeSet.AccessedSlots {
		op := &storageOperation{
			incarnation:  incarnation,
			storageValue: value,
		}
		var smap *slotMap
		if v, ok := s.multiVersionStorage[key]; !ok {
			smap = &slotMap{
				ops:     make(map[int]*storageOperation),
				ordered: make([]int, 0, 2000),
			}
		} else {
			smap = v
		}
		smap.ops[index] = op
		smap.ordered = append(smap.ordered, index)
		sort.Ints(smap.ordered)
		s.multiVersionStorage[key] = smap
	}
}

// GetCommittedState retrieves a value from the committed account storage trie.
func (s *stmStateObject) GetCommittedState(db Database, key common.Hash) common.Hash {
	// If the object was destructed in *this* block (and potentially resurrected),
	// the storage has been cleared out, and we should *not* consult the previous
	// database about any storage values. The only possible alternatives are:
	//   1) resurrect happened, and new slot values were set -- those should
	//      have been handles via pendingStorage above.
	//   2) we don't have new values, and can deliver empty response back
	//if _, destructed := s.db.stateObjectsDestruct[s.address]; destructed {
	//	return common.Hash{}
	//}
	// If no live objects are available, attempt to use snapshots
	var (
		enc []byte
		err error
	)
	if s.db.snap != nil {
		start := time.Now()
		enc, err = s.db.snap.Storage(s.addrHash, crypto.Keccak256Hash(key.Bytes()))
		if metrics.EnabledExpensive {
			s.db.SnapshotStorageReads += time.Since(start)
		}
	}
	// If the snapshot is unavailable or reading from it fails, load from the database.
	if s.db.snap == nil || err != nil {
		start := time.Now()
		tr, err := s.getTrie(db)
		if err != nil {
			s.db.setError(err)
			return common.Hash{}
		}
		enc, err = tr.TryGet(key.Bytes())
		if metrics.EnabledExpensive {
			s.db.StorageReads += time.Since(start)
		}
		if err != nil {
			s.db.setError(err)
			return common.Hash{}
		}
	}
	var value common.Hash
	if len(enc) > 0 {
		_, content, _, err := rlp.Split(enc)
		if err != nil {
			s.db.setError(err)
		}
		value.SetBytes(content)
	}
	return value
}

// updateAccount updates the latest version of account data
func (s *stmStateObject) updateAccount() {
	latestVer := -1
	ordered := s.multiVersionState.ordered
	for i := len(ordered) - 1; i >= 0; i-- {
		if _, ok := s.multiVersionState.ops[ordered[i]]; ok {
			latestVer = ordered[i]
			break
		}
	}
	if latestVer > -1 {
		s.data = s.multiVersionState.ops[latestVer].account
		s.dirtyMarker = true
	}
	// clear the multi-version state
	s.multiVersionState = &stateMap{
		ops:     make(map[int]*stateOperation),
		ordered: make([]int, 0, 5000),
	}
}

// updateAccount updates the latest version of value for each dirty storage slot
func (s *stmStateObject) updateStorage() {
	for address, smap := range s.multiVersionStorage {
		latestVer := -1
		for i := len(smap.ordered) - 1; i >= 0; i-- {
			if _, ok := smap.ops[smap.ordered[i]]; ok {
				latestVer = smap.ordered[i]
				break
			}
		}
		if latestVer > -1 {
			latestValue := smap.ops[latestVer].storageValue
			s.dirtyStorage[address] = latestValue
		}
	}

	if len(s.dirtyStorage) > 0 {
		s.dirtyMarker = true
	}
	// clear the multi-version storage
	s.multiVersionStorage = make(map[common.Hash]*slotMap)
}

// finalise moves all dirty storage slots into the pending area to be hashed or
// committed later. It is invoked at the end of executing a batch of transactions.
func (s *stmStateObject) finalise(prefetch bool) {
	slotsToPrefetch := make([][]byte, 0, len(s.dirtyStorage))
	for key := range s.dirtyStorage {
		slotsToPrefetch = append(slotsToPrefetch, common.CopyBytes(key[:]))
	}
	data := s.data.StateAccount
	if s.db.prefetcher != nil && prefetch && len(slotsToPrefetch) > 0 && data.Root != types.EmptyRootHash {
		s.db.prefetcher.prefetch(s.addrHash, data.Root, slotsToPrefetch)
	}
}

// updateTrie writes cached storage modifications into the object's storage trie.
// It will return nil if the trie has not been loaded and no changes have been
// made. An error will be returned if the trie can't be loaded/updated correctly.
func (s *stmStateObject) updateTrie(db Database) (Trie, error) {
	// Make sure all dirty slots are finalized into the dirty storage area
	s.finalise(false) // Don't prefetch anymore, pull directly if need be
	if len(s.dirtyStorage) == 0 {
		return s.trie, nil
	}
	// Track the amount of time wasted on updating the storage trie
	if metrics.EnabledExpensive {
		defer func(start time.Time) { s.db.StorageUpdates += time.Since(start) }(time.Now())
	}
	// The snapshot storage map for the object
	var (
		storage map[common.Hash][]byte
		hasher  = s.db.hasher
	)
	tr, err := s.getTrie(db)
	if err != nil {
		s.db.setError(err)
		return nil, err
	}
	// Insert all the pending updates into the trie
	usedStorage := make([][]byte, 0, len(s.dirtyStorage))
	for key, slotValue := range s.dirtyStorage {
		// Skip noop changes, persist actual changes
		var v []byte
		if (slotValue == common.Hash{}) {
			if err := tr.TryDelete(key[:]); err != nil {
				s.db.setError(err)
				return nil, err
			}
			s.db.StorageDeleted += 1
		} else {
			// Encoding []byte cannot fail, ok to ignore the error.
			v, _ = rlp.EncodeToBytes(common.TrimLeftZeroes(slotValue[:]))
			if err := tr.TryUpdate(key[:], v); err != nil {
				s.db.setError(err)
				return nil, err
			}
			s.db.StorageUpdated += 1
		}
		// If state snapshotting is active, cache the data til commit
		if s.db.snap != nil {
			if storage == nil {
				// Retrieve the old storage map, if available, create a new one otherwise
				if storage = s.db.snapStorage[s.addrHash]; storage == nil {
					storage = make(map[common.Hash][]byte)
					s.db.snapStorage[s.addrHash] = storage
				}
			}
			storage[crypto.HashData(hasher, key[:])] = v // v will be nil if it's deleted
		}
		usedStorage = append(usedStorage, common.CopyBytes(key[:])) // Copy needed for closure
	}
	if s.db.prefetcher != nil {
		data := s.data.StateAccount
		s.db.prefetcher.used(s.addrHash, data.Root, usedStorage)
	}
	if len(s.dirtyStorage) > 0 {
		s.dirtyStorage = make(StmStorage)
	}
	return tr, nil
}

// UpdateRoot sets the trie root to the current root hash of. An error
// will be returned if trie root hash is not computed correctly.
func (s *stmStateObject) updateRoot(db Database) {
	tr, err := s.updateTrie(db)
	if err != nil {
		return
	}
	// If nothing changed, don't bother with hashing anything
	if tr == nil {
		return
	}
	// Track the amount of time wasted on hashing the storage trie
	if metrics.EnabledExpensive {
		defer func(start time.Time) { s.db.StorageHashes += time.Since(start) }(time.Now())
	}
	//data := s.data.StateAccount[s.data.len-1].StateAccount
	//data.Root = tr.Hash()
	s.data.StateAccount.Root = tr.Hash()
}

// commitTrie submits the storage changes into the storage trie and re-computes
// the root. Besides, all trie changes will be collected in a nodeset and returned.
func (s *stmStateObject) commitTrie(db Database) (*trie.NodeSet, error) {
	tr, err := s.updateTrie(db)
	if err != nil {
		return nil, err
	}
	// If nothing changed, don't bother with committing anything
	if tr == nil {
		return nil, nil
	}
	// Track the amount of time wasted on committing the storage trie
	if metrics.EnabledExpensive {
		defer func(start time.Time) { s.db.StorageCommits += time.Since(start) }(time.Now())
	}
	root, nodes := tr.Commit(false)
	data := s.data.StateAccount
	data.Root = root
	return nodes, nil
}

func (s *stmStateObject) setStateAccount(txObj *stmTxStateObject, txIndex, txIncarnation int) {
	s.data = newSStateAccount(txObj.data.StateAccount, txObj.data.Code, txObj.data.dirtyCode, txObj.data.suicided, txObj.data.deleted)
}

// Address returns the address of the contract/account
func (s *stmStateObject) Address() common.Address {
	return s.address
}

func (s *stmStateObject) CodeHash() []byte {
	data := s.data.StateAccount
	return data.CodeHash
}

func (s *stmStateObject) Balance() *big.Int {
	data := s.data.StateAccount
	return data.Balance
}

func (s *stmStateObject) Nonce() uint64 {
	data := s.data.StateAccount
	return data.Nonce
}

func (s *stmStateObject) Root() common.Hash {
	data := s.data.StateAccount
	return data.Root
}
