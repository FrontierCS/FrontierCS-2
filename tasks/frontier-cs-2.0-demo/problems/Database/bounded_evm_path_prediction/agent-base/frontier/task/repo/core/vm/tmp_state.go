package vm

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/holiman/uint256"
)

// TmpState defines the temporary state space that stores all accessed and updated state slots
// In sstore operation, we can easily obtain the variable information according to storage slot
type TmpState struct {
	Space map[common.Address]map[common.Hash]*Fragment
}

type Fragment struct {
	slot      uint256.Int
	stateVars map[string]*StateVar
	isCompact bool
}

func NewTmpState() *TmpState {
	tmp := &TmpState{Space: make(map[common.Address]map[common.Hash]*Fragment)}
	return tmp
}

func (ts *TmpState) GetFragment(addr common.Address, slotInt uint256.Int) (*Fragment, bool) {
	fragmentMap, ok := ts.Space[addr]
	if !ok {
		fragmentMap = make(map[common.Hash]*Fragment)
		ts.Space[addr] = fragmentMap
		return nil, false
	}
	slotHash := common.Hash(slotInt.Bytes32())
	if fragment, ok2 := fragmentMap[slotHash]; ok2 {
		return fragment, true
	}
	return nil, false
}

func (ts *TmpState) InsertUnit(addr common.Address, slotInt uint256.Int) *Fragment {
	fragmentMap, ok := ts.Space[addr]
	if !ok {
		fragmentMap = make(map[common.Hash]*Fragment)
		ts.Space[addr] = fragmentMap
	}
	slotHash := common.Hash(slotInt.Bytes32())
	fragment, ok2 := fragmentMap[slotHash]
	if !ok2 {
		fragment = NewFragment(slotInt)
		fragmentMap[slotHash] = fragment
	}
	return fragment
}

func NewFragment(slot uint256.Int) *Fragment {
	frag := &Fragment{
		slot:      slot,
		stateVars: make(map[string]*StateVar),
		isCompact: false,
	}
	return frag
}

func (f *Fragment) GetSlot() uint256.Int { return f.slot }
func (f *Fragment) GetCompact() bool     { return f.isCompact }

func (f *Fragment) Compact() {
	f.isCompact = true
}

func (f *Fragment) GetVar(offset uint256.Int) *StateVar {
	if stateVar, ok := f.stateVars[offset.String()]; ok {
		return stateVar
	} else {
		// sstore操作直接存储256位变量的情况
		stateVar = newStateVar(uint256.Int{0}, offset, 256)
		f.stateVars[offset.String()] = stateVar
		return stateVar
	}
}

func (f *Fragment) GenerateVar(stateUnit *StateUnit) {
	offset := stateUnit.GetOffset()
	bits := stateUnit.GetBits()
	val := stateUnit.GetStorageValue()
	if offset.Uint64() > 0 && bits < 256 {
		f.Compact()
	}
	newVar := newStateVar(val, offset, bits)
	f.stateVars[offset.String()] = newVar
}

type StateVar struct {
	slotVal    uint256.Int
	updatedVal uint256.Int
	offset     uint256.Int
	bits       int
	updated    bool
}

func newStateVar(slotVal, offset uint256.Int, bits int) *StateVar {
	return &StateVar{
		slotVal: slotVal,
		offset:  offset,
		bits:    bits,
	}
}

func (sv *StateVar) GetSlotVal() uint256.Int    { return sv.slotVal }
func (sv *StateVar) GetUpdatedVal() uint256.Int { return sv.updatedVal }
func (sv *StateVar) GetBits() int               { return sv.bits }
func (sv *StateVar) GetOffset() uint256.Int     { return sv.offset }
func (sv *StateVar) GetFlag() bool              { return sv.updated }

func (sv *StateVar) SetUpdatedVal(val uint256.Int) {
	sv.updatedVal = val
	sv.updated = true
}
