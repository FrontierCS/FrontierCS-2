package state

import (
	"bytes"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"math/big"
	"seerEVM/core/types"
)

type SStorage map[common.Hash]*SSlot

type stmTxStateObject struct {
	txIndex          int
	txIncarnation    int
	txStorageVersion int

	address  common.Address
	addrHash common.Hash // hash of ethereum address of the account
	data     *SStateAccount
	stmTx    *SeerTransaction
	stateDB  *SeerStateDB

	dirtyStorage   Storage // Storage entries that have been modified in the current transaction execution 缓存当前合约账户中写⼊的变量地址与变量值
	originStorage  Storage // Storage cache of original entries to dedup rewrites, reset for every transaction  缓存当前合约账户中被读取过的变量地址与变量值
	modifiedMarker bool    // indicates whether the account state has been modified
}

// empty returns whether the account is considered empty.
func (s *stmTxStateObject) empty() bool {
	data := s.data.StateAccount
	return data.Nonce == 0 && data.Balance.Sign() == 0 && bytes.Equal(data.CodeHash, types.EmptyCodeHash.Bytes())
}

// newStmTxStateObject, SStateAccount 记录了其数据所读取的版本
func newStmTxStateObject(stmTx *SeerTransaction, stateDB *SeerStateDB, address common.Address, data SStateAccount, txIndex, txIncarnation, txStorageVersion int) *stmTxStateObject {
	return &stmTxStateObject{
		txIndex:          txIndex,
		txIncarnation:    txIncarnation,
		txStorageVersion: txStorageVersion,
		stmTx:            stmTx,
		stateDB:          stateDB,
		address:          address,
		addrHash:         crypto.Keccak256Hash(address[:]),
		data:             newSStateAccount(data.StateAccount, data.Code, data.dirtyCode, data.suicided, data.deleted),
		dirtyStorage:     make(Storage),
		originStorage:    make(Storage),
		modifiedMarker:   false,
	}
}

func (s *stmTxStateObject) markSuicided() {
	s.data.suicided = true
	s.modifiedMarker = true
}

func (s *stmTxStateObject) touch() {
	s.stmTx.TxDB.journal.append(stmTouchChange{
		account: &s.address,
	})
	if s.address == ripemd {
		// Explicitly put it in the dirty-cache, which is otherwise generated from
		// flattened journals.
		s.stmTx.TxDB.journal.dirty(s.address)
	}
}

// GetState retrieves a value from the account storage trie.
func (s *stmTxStateObject) GetState(key common.Hash) common.Hash {
	// If we have a dirty value for this state entry, return it
	value, dirty := s.dirtyStorage[key]
	if dirty {
		return value
	}
	// 如果本交易没有写入，查看本交易是否获取过
	origin, ok := s.originStorage[key]
	if ok {
		return origin
	}
	// 否则需并发的从statedb中读取
	readRes := s.stateDB.readStorageVersion(s.address, key, s.txStorageVersion)
	if err := s.stmTx.process(readRes, s.address, &key); err != nil {
		//log.Println(err)
		if err.Error() == "notFound" {
			return common.Hash{}
		}
	}
	stateHash := readRes.DirtyStorage.storageValue
	s.originStorage[key] = stateHash
	return stateHash
}

// GetCommittedState retrieves a value from the committed account storage trie.
func (s *stmTxStateObject) GetCommittedState(key common.Hash) common.Hash {
	// 如果本交易没有写入，查看本交易是否获取过
	origin, ok := s.originStorage[key]
	if ok {
		return origin
	}
	// 否则需并发的从statedb中读取
	readRes := s.stateDB.readStorageVersion(s.address, key, s.txStorageVersion)
	if err := s.stmTx.process(readRes, s.address, &key); err != nil {
		//log.Println(err)
		if err.Error() == "notFound" {
			return common.Hash{}
		}
	}
	stateHash := readRes.DirtyStorage.storageValue
	s.originStorage[key] = stateHash
	return stateHash
}

// SetState updates a value in account storage.
func (s *stmTxStateObject) SetState(key, value common.Hash) {
	// If the new value is the same as old, don't set
	prev := s.GetState(key)
	if prev == value {
		return
	}
	// New value is different, update and journal the change
	s.stmTx.TxDB.journal.append(stmStorageChange{
		account:  &s.address,
		key:      key,
		prevalue: prev,
	})
	s.setState(key, value)
}

func (s *stmTxStateObject) setState(key, value common.Hash) {
	s.dirtyStorage[key] = value
}

// AddBalance adds amount to s's balance.
// It is used to add funds to the destination account of a transfer.
func (s *stmTxStateObject) AddBalance(amount *big.Int) {
	// EIP161: We must check emptiness for the objects such that the account
	// clearing (0,0,0 objects) can take effect.
	if amount.Sign() == 0 {
		if s.empty() {
			s.touch()
		}
		return
	}
	//fmt.Println("SetBalance Error", s.Balance(), amount)
	s.SetBalance(new(big.Int).Add(s.Balance(), amount))
}

// SubBalance removes amount from s's balance.
// It is used to remove funds from the origin account of a transfer.
func (s *stmTxStateObject) SubBalance(amount *big.Int) {
	if amount.Sign() == 0 {
		return
	}
	s.SetBalance(new(big.Int).Sub(s.Balance(), amount))
}

func (s *stmTxStateObject) SetBalance(amount *big.Int) {
	s.stmTx.TxDB.journal.append(stmBalanceChange{
		account: &s.address,
		prev:    new(big.Int).Set(s.data.StateAccount.Balance),
	})
	s.setBalance(amount)
}

func (s *stmTxStateObject) setBalance(amount *big.Int) {
	s.data.StateAccount.Balance = amount
	s.modifiedMarker = true
}

//
// Attribute accessors
//

// Address returns the address of the contract/account
func (s *stmTxStateObject) Address() common.Address {
	return s.address
}

// Code returns the contract code associated with this object, if any.
func (s *stmTxStateObject) Code() []byte {
	if s.data.Code != nil {
		return s.data.Code
	}
	if bytes.Equal(s.CodeHash(), types.EmptyCodeHash.Bytes()) {
		return nil
	}
	//code := s.statedb.GetCode(s.address)
	//txInfo := TxInfoMini{Index: code.TxInfo.Index, Incarnation: code.TxInfo.Incarnation}
	//s.code = &SCode{Code: common.CopyBytes(code.Code), TxInfo: txInfo}
	//return s.code.Code
	// 开始时须先获取code
	return nil
}

// CodeSize returns the size of the contract code associated with this object,
// or zero if none. This method is an almost mirror of Code, but uses a cache
// inside the database to avoid loading codes seen recently.
func (s *stmTxStateObject) CodeSize() int {
	code := s.Code()
	return len(code)
}

func (s *stmTxStateObject) SetCode(codeHash common.Hash, code []byte) {
	prevcode := s.Code()
	s.stmTx.TxDB.journal.append(stmCodeChange{
		account:  &s.address,
		prevhash: s.CodeHash(),
		prevcode: prevcode,
	})
	s.setCode(codeHash, code)
}

func (s *stmTxStateObject) setCode(codeHash common.Hash, code []byte) {
	s.data.Code = code
	s.data.StateAccount.CodeHash = codeHash[:]
	s.data.dirtyCode = true
	s.modifiedMarker = true
}

func (s *stmTxStateObject) SetNonce(nonce uint64) {
	s.stmTx.TxDB.journal.append(stmNonceChange{
		account: &s.address,
		prev:    s.data.StateAccount.Nonce,
	})
	s.setNonce(nonce)
}

func (s *stmTxStateObject) setNonce(nonce uint64) {
	s.data.StateAccount.Nonce = nonce
	s.modifiedMarker = true
}

func (s *stmTxStateObject) CodeHash() []byte {
	return s.data.StateAccount.CodeHash
}

func (s *stmTxStateObject) Balance() *big.Int {
	return s.data.StateAccount.Balance
}

func (s *stmTxStateObject) Nonce() uint64 {
	return s.data.StateAccount.Nonce
}

func (s *stmTxStateObject) Root() common.Hash {
	return s.data.StateAccount.Root
}
