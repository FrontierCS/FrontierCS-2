package state

import (
	"github.com/ethereum/go-ethereum/common"
	"math/big"
)

// stmJournalEntry is a modification entry in the state change stmJournal that can be
// reverted on demand.
type stmJournalEntry interface {
	// revert undoes the changes introduced by this stmJournal entry.
	revert(db *seerTxStateDB)

	// dirtied returns the Ethereum address modified by this stmJournal entry.
	dirtied() *common.Address
}

// stmJournal contains the list of state modifications applied since the last state
// commit. These are tracked to be able to be reverted in the case of an execution
// exception or request for reversal.
type stmJournal struct {
	entries []stmJournalEntry      // Current changes tracked by the stmJournal
	dirties map[common.Address]int // Dirty accounts and the number of changes
}

// newJournal creates a new initialized stmJournal.
func newStmJournal() *stmJournal {
	return &stmJournal{
		dirties: make(map[common.Address]int),
	}
}

// append inserts a new modification entry to the end of the change stmJournal.
func (j *stmJournal) append(entry stmJournalEntry) {
	j.entries = append(j.entries, entry)
	if addr := entry.dirtied(); addr != nil {
		j.dirties[*addr]++
	}
}

// revert undoes a batch of journalled modifications along with any reverted
// dirty handling too.
func (j *stmJournal) revert(statedb *seerTxStateDB, snapshot int) {
	for i := len(j.entries) - 1; i >= snapshot; i-- {
		// Undo the changes made by the operation
		j.entries[i].revert(statedb)

		// Drop any dirty tracking induced by the change
		if addr := j.entries[i].dirtied(); addr != nil {
			if j.dirties[*addr]--; j.dirties[*addr] == 0 {
				delete(j.dirties, *addr)
			}
		}
	}
	j.entries = j.entries[:snapshot]
}

// dirty explicitly sets an address to dirty, even if the change entries would
// otherwise suggest it as clean. This method is an ugly hack to handle the RIPEMD
// precompile consensus exception.
func (j *stmJournal) dirty(addr common.Address) {
	j.dirties[addr]++
}

// length returns the current number of entries in the stmJournal.
func (j *stmJournal) length() int {
	return len(j.entries)
}

type (
	// Changes to the account trie.
	stmCreateObjectChange struct {
		account *common.Address
	}
	stmResetObjectChange struct {
		prev         *stmTxStateObject
		prevdestruct bool
	}
	stmSuicideChange struct {
		account     *common.Address
		prev        bool // whether account had already suicided
		prevbalance *big.Int
	}

	// Changes to individual accounts.
	stmBalanceChange struct {
		account *common.Address
		prev    *big.Int
	}
	stmNonceChange struct {
		account *common.Address
		prev    uint64
	}
	stmStorageChange struct {
		account       *common.Address
		key, prevalue common.Hash
	}
	stmCodeChange struct {
		account            *common.Address
		prevcode, prevhash []byte
	}

	// Changes to other state values.
	stmRefundChange struct {
		prev uint64
	}
	stmAddLogChange struct {
		txhash common.Hash
	}
	stmAddPreimageChange struct {
		hash common.Hash
	}
	stmTouchChange struct {
		account *common.Address
	}
	// Changes to the access list
	stmAccessListAddAccountChange struct {
		address *common.Address
	}
	stmAccessListAddSlotChange struct {
		address *common.Address
		slot    *common.Hash
	}

	stmTransientStorageChange struct {
		account       *common.Address
		key, prevalue common.Hash
	}
)

func (ch stmCreateObjectChange) revert(s *seerTxStateDB) {
	delete(s.stateObjects, *ch.account)
	delete(s.stateObjectsDirty, *ch.account)
}

func (ch stmCreateObjectChange) dirtied() *common.Address {
	return ch.account
}

func (ch stmResetObjectChange) revert(s *seerTxStateDB) {
	s.setStateObject(ch.prev)
	if !ch.prevdestruct {
		delete(s.stateObjectsDestruct, ch.prev.address)
	}
}

func (ch stmResetObjectChange) dirtied() *common.Address {
	return nil
}

func (ch stmSuicideChange) revert(s *seerTxStateDB) {
	obj := s.getStateObject(*ch.account)
	if obj != nil {
		obj.data.suicided = ch.prev
		obj.setBalance(ch.prevbalance)
	}
}

func (ch stmSuicideChange) dirtied() *common.Address {
	return ch.account
}

func (ch stmTouchChange) revert(s *seerTxStateDB) {
}

func (ch stmTouchChange) dirtied() *common.Address {
	return ch.account
}

func (ch stmBalanceChange) revert(s *seerTxStateDB) {
	s.getStateObject(*ch.account).setBalance(ch.prev)
}

func (ch stmBalanceChange) dirtied() *common.Address {
	return ch.account
}

func (ch stmNonceChange) revert(s *seerTxStateDB) {
	s.getStateObject(*ch.account).setNonce(ch.prev)
}

func (ch stmNonceChange) dirtied() *common.Address {
	return ch.account
}

func (ch stmCodeChange) revert(s *seerTxStateDB) {
	s.getStateObject(*ch.account).setCode(common.BytesToHash(ch.prevhash), ch.prevcode)
}

func (ch stmCodeChange) dirtied() *common.Address {
	return ch.account
}

func (ch stmStorageChange) revert(s *seerTxStateDB) {
	s.getStateObject(*ch.account).setState(ch.key, ch.prevalue)
}

func (ch stmStorageChange) dirtied() *common.Address {
	return ch.account
}

func (ch stmTransientStorageChange) revert(s *seerTxStateDB) {
	s.setTransientState(*ch.account, ch.key, ch.prevalue)
}

func (ch stmTransientStorageChange) dirtied() *common.Address {
	return nil
}

func (ch stmRefundChange) revert(s *seerTxStateDB) {
	s.refund = ch.prev
}

func (ch stmRefundChange) dirtied() *common.Address {
	return nil
}

func (ch stmAddLogChange) revert(s *seerTxStateDB) {
	logs := s.logs[ch.txhash]
	if len(logs) == 1 {
		delete(s.logs, ch.txhash)
	} else {
		s.logs[ch.txhash] = logs[:len(logs)-1]
	}
	s.logSize--
}

func (ch stmAddLogChange) dirtied() *common.Address {
	return nil
}

func (ch stmAddPreimageChange) revert(s *seerTxStateDB) {
	delete(s.preimages, ch.hash)
}

func (ch stmAddPreimageChange) dirtied() *common.Address {
	return nil
}

func (ch stmAccessListAddAccountChange) revert(s *seerTxStateDB) {
	/*
		One important invariant here, is that whenever a (addr, slot) is added, if the
		addr is not already present, the add causes two stmJournal entries:
		- one for the address,
		- one for the (address,slot)
		Therefore, when unrolling the change, we can always blindly delete the
		(addr) at this point, since no storage adds can remain when come upon
		a single (addr) change.
	*/
	s.accessList.DeleteAddress(*ch.address)
}

func (ch stmAccessListAddAccountChange) dirtied() *common.Address {
	return nil
}

func (ch stmAccessListAddSlotChange) revert(s *seerTxStateDB) {
	s.accessList.DeleteSlot(*ch.address, *ch.slot)
}

func (ch stmAccessListAddSlotChange) dirtied() *common.Address {
	return nil
}
