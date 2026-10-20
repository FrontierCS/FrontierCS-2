package vm

import (
	"errors"
	"github.com/ethereum/go-ethereum/common"
	"github.com/holiman/uint256"
	"sync"
)

// PreExecutionTable stores necessary information of a pre-executed tx for subsequent actual execution
type PreExecutionTable struct {
	Space map[common.Hash]*Result
	Mutex sync.RWMutex
}

func NewPreExecutionTable() *PreExecutionTable {
	return &PreExecutionTable{
		Space: make(map[common.Hash]*Result),
	}
}

func (pe *PreExecutionTable) InitialResult(txID common.Hash) *Result {
	newResult := NewResult()
	pe.Space[txID] = newResult
	return newResult
}

// GetResult obtains the pre-executed results for a specific transaction
func (pe *PreExecutionTable) GetResult(txID common.Hash) (*Result, error) {
	pe.Mutex.RLock()
	defer pe.Mutex.RUnlock()
	result, ok := pe.Space[txID]
	if !ok {
		return nil, errors.New("cannot find pre-execution info")
	}
	return result, nil
}

// SetOnChain marks the on-chain label for transactions that are persisted on the blockchain
func (pe *PreExecutionTable) SetOnChain(txID common.Hash) error {
	result, ok := pe.Space[txID]
	if !ok {
		return errors.New("cannot find pre-execution info")
	}
	result.onChain = true
	return nil
}

// ClearTable clears unuseful pre-exeuction information (executed transactions) from the table
func (pe *PreExecutionTable) ClearTable() {
	for id, ret := range pe.Space {
		if ret.onChain {
			delete(pe.Space, id)
		}
	}
}

// Result stores necessary pre-execution information for a specific tx
type Result struct {
	branchCtx     []*BranchContext
	finalSnapshot *FinalSnapshot      // record the final snapshot at the end of execution
	callStack     map[int][]*Snapshot // record the map structure from call depth to the set of corresponding snapshots
	txContext     TxContext
	onChain       bool
}

func NewResult() *Result {
	return &Result{
		branchCtx: make([]*BranchContext, 0, 100),
		callStack: make(map[int][]*Snapshot),
		onChain:   false,
	}
}

func (rs *Result) GetBranches() []*BranchContext     { return rs.branchCtx }
func (rs *Result) GetFinalSnapshot() *FinalSnapshot  { return rs.finalSnapshot }
func (rs *Result) GetCallStack() map[int][]*Snapshot { return rs.callStack }
func (rs *Result) GetTxContext() TxContext           { return rs.txContext }
func (rs *Result) GetOnChain() bool                  { return rs.onChain }
func (rs *Result) UpdateTxContext(context TxContext) { rs.txContext = context }

// Reset deletes the invalid branch info and snapshots (used for pre-execution repair)
func (rs *Result) Reset(loc1, depth int, pc uint64) {
	rs.finalSnapshot = &FinalSnapshot{}
	rs.branchCtx = rs.branchCtx[:loc1+1]
	for dep, sps := range rs.callStack {
		if dep == depth {
			for i, sp := range sps {
				if sp.GetPC() == pc {
					sps = sps[:i+1]
				}
			}
		} else if dep > depth {
			delete(rs.callStack, dep)
		}
	}
}

// InitialBranch initials a new branch context (used in non-checkpoint version)
func (rs *Result) InitialBranch() {
	ctx := initializeBranchContext()
	rs.branchCtx = append(rs.branchCtx, ctx)
}

// CacheSStoreInfo stores sstore information and adds to the latest branch context (has not been filled with branch information)
func (rs *Result) CacheSStoreInfo(contractAddr common.Address, updatedVal uint256.Int, loc, val TracingUnit, isCompact bool) {
	if len(rs.branchCtx) > 0 {
		latestCtx := rs.branchCtx[len(rs.branchCtx)-1]
		if !latestCtx.isFilled {
			latestCtx.updateSStoreInfo(contractAddr, updatedVal, loc, val, isCompact)
			return
		}
	}
	ctx := initializeBranchContext()
	ctx.updateSStoreInfo(contractAddr, updatedVal, loc, val, isCompact)
	rs.branchCtx = append(rs.branchCtx, ctx)
}

// CacheSnapshot stores EVM snapshot and adds to the latest branch context (has not been filled with branch information)
func (rs *Result) CacheSnapshot(stack StackInterface, memory *Memory, pc, jumpPc uint64, contract *Contract, depth int) {
	if len(rs.branchCtx) > 0 {
		latestCtx := rs.branchCtx[len(rs.branchCtx)-1]
		if !latestCtx.isFilled {
			snapshot := latestCtx.updateSnapshot(stack, memory, pc, jumpPc, contract, depth)
			rs.callStack[depth] = append(rs.callStack[depth], snapshot)
			return
		}
	}
	ctx := initializeBranchContext()
	snapshot := ctx.updateSnapshot(stack, memory, pc, jumpPc, contract, depth)
	rs.callStack[depth] = append(rs.callStack[depth], snapshot)
	rs.branchCtx = append(rs.branchCtx, ctx)
}

// CacheReadSet caches tx's read set during pre-execution
func (rs *Result) CacheReadSet(callerAddr common.Address, slot *common.Hash) {
	if len(rs.branchCtx) > 0 {
		latestCtx := rs.branchCtx[len(rs.branchCtx)-1]
		if !latestCtx.isFilled {
			latestCtx.addReadRecord(callerAddr, slot)
			return
		}
	}
	ctx := initializeBranchContext()
	ctx.addReadRecord(callerAddr, slot)
	rs.branchCtx = append(rs.branchCtx, ctx)
}

// CacheWriteSet caches tx's write set during pre-execution
func (rs *Result) CacheWriteSet(callerAddr common.Address, slot *common.Hash) {
	if len(rs.branchCtx) > 0 {
		latestCtx := rs.branchCtx[len(rs.branchCtx)-1]
		if !latestCtx.isFilled {
			latestCtx.addWriteRecord(callerAddr, slot)
			return
		}
	}
	ctx := initializeBranchContext()
	ctx.addWriteRecord(callerAddr, slot)
	rs.branchCtx = append(rs.branchCtx, ctx)
}

// GenerateFinalSnapshot generates a final snapshot for storing final execution results and sstore information
func (rs *Result) GenerateFinalSnapshot(result []byte, gas uint64, err error) {
	var (
		sstoreInfo []*SstoreInfo
		rSet       ReadSet
		wSet       WriteSet
	)
	if len(rs.branchCtx) > 0 {
		latestCtx := rs.branchCtx[len(rs.branchCtx)-1]
		if !latestCtx.isFilled {
			// 在执行结束前的相关sstore信息和读写集会暂时缓存在临时分支的上下文中
			// 获取sstore和读写集后将临时分支的上下文删除
			sstoreInfo = latestCtx.sstoreInfo
			rSet = latestCtx.readSet
			wSet = latestCtx.writeSet
			rs.branchCtx = rs.branchCtx[:len(rs.branchCtx)-1]
		}
	}
	final := newFinalSnapshot(result, gas, err, sstoreInfo, rSet, wSet)
	rs.finalSnapshot = final
}

// UpdateBranchInfo updates branch info when encountering branch relevant to state variables
func (rs *Result) UpdateBranchInfo(contractAddr common.Address, sUnit, cUnit TracingUnit, isVar, direction bool, id, judgement string) {
	latestCtx := rs.branchCtx[len(rs.branchCtx)-1]
	latestCtx.addBranchInfo(contractAddr, sUnit, cUnit, isVar, direction, id, judgement)
}

// UpdateDirection updates the predicted branch direction
func (rs *Result) UpdateDirection(res int) {
	latestCtx := rs.branchCtx[len(rs.branchCtx)-1]
	latestCtx.DecideDirection(res)
}

// OutputRWSet outputs all cached read/write sets during pre-execution
func (rs *Result) OutputRWSet() (ReadSet, WriteSet) {
	var finalRSet = make(map[common.Address]*Recorder)
	var finalWSet = make(map[common.Address]*Recorder)
	for _, ctx := range rs.branchCtx {
		CombineRWSet(finalRSet, ctx.readSet)
		CombineRWSet(finalWSet, ctx.writeSet)
	}
	if rs.finalSnapshot != nil {
		readSet, writeSet := rs.finalSnapshot.GetRWSet()
		CombineRWSet(finalRSet, readSet)
		CombineRWSet(finalWSet, writeSet)
	}
	return finalRSet, finalWSet
}

// FinalSnapshot records final execution results and sstore information
type FinalSnapshot struct {
	ret        []byte
	gas        uint64
	err        error
	sstoreInfo []*SstoreInfo
	readSet    ReadSet
	writeSet   WriteSet
}

func newFinalSnapshot(result []byte, gas uint64, err error, info []*SstoreInfo, rSet ReadSet, wSet WriteSet) *FinalSnapshot {
	return &FinalSnapshot{
		ret:        result,
		gas:        gas,
		err:        err,
		sstoreInfo: info,
		readSet:    rSet,
		writeSet:   wSet,
	}
}

func (fs *FinalSnapshot) GetResult() []byte             { return fs.ret }
func (fs *FinalSnapshot) GetGas() uint64                { return fs.gas }
func (fs *FinalSnapshot) GetError() error               { return fs.err }
func (fs *FinalSnapshot) GetSstoreInfo() []*SstoreInfo  { return fs.sstoreInfo }
func (fs *FinalSnapshot) GetRWSet() (ReadSet, WriteSet) { return fs.readSet, fs.writeSet }

// BranchContext records the necessary branch context for fast path execution
type BranchContext struct {
	branchID     string
	contractAddr common.Address // the initial caller contract address
	sUnit        TracingUnit
	cUnit        TracingUnit
	judgement    string
	isVar        bool
	isTaken      bool
	direction    bool // defines the judgement direction (x cmp. y or y cmp. x)
	snapshots    []*Snapshot
	sstoreInfo   []*SstoreInfo
	readSet      ReadSet
	writeSet     WriteSet
	isFilled     bool // whether filled with branch information
}

func initializeBranchContext() *BranchContext {
	return &BranchContext{
		snapshots:  make([]*Snapshot, 0, 20),
		sstoreInfo: make([]*SstoreInfo, 0, 20),
		readSet:    make(map[common.Address]*Recorder),
		writeSet:   make(map[common.Address]*Recorder),
		isFilled:   false,
	}
}

func (bc *BranchContext) GetBranchID() string           { return bc.branchID }
func (bc *BranchContext) GetAddr() common.Address       { return bc.contractAddr }
func (bc *BranchContext) GetStateUnit() TracingUnit     { return bc.sUnit }
func (bc *BranchContext) GetTracingUnit() TracingUnit   { return bc.cUnit }
func (bc *BranchContext) IsVar() bool                   { return bc.isVar }
func (bc *BranchContext) GetJudgement() string          { return bc.judgement }
func (bc *BranchContext) GetSnapshots() []*Snapshot     { return bc.snapshots }
func (bc *BranchContext) GetSstoreInfo() []*SstoreInfo  { return bc.sstoreInfo }
func (bc *BranchContext) GetRWSet() (ReadSet, WriteSet) { return bc.readSet, bc.writeSet }
func (bc *BranchContext) GetBranchDirection() bool      { return bc.isTaken }
func (bc *BranchContext) GetJudgementDirection() bool   { return bc.direction }
func (bc *BranchContext) GetFilled() bool               { return bc.isFilled }

func (bc *BranchContext) addBranchInfo(contractAddr common.Address, sUnit, cUnit TracingUnit, isVar, direction bool, id, judgement string) {
	bc.branchID = id
	bc.contractAddr = contractAddr
	bc.sUnit = sUnit
	bc.cUnit = cUnit
	bc.isVar = isVar
	bc.direction = direction
	bc.judgement = judgement
	bc.isFilled = true
}

func (bc *BranchContext) DecideDirection(res int) {
	if res == 0 {
		bc.isTaken = false
	} else {
		bc.isTaken = true
	}
}

func (bc *BranchContext) updateSStoreInfo(contractAddr common.Address, updatedVal uint256.Int, loc, val TracingUnit, isCompact bool) {
	newInfo := newSstoreInfo(contractAddr, updatedVal, loc, val, isCompact)
	bc.sstoreInfo = append(bc.sstoreInfo, newInfo)
}

func (bc *BranchContext) updateSnapshot(stack StackInterface, memory *Memory, pc, jumpPc uint64, contract *Contract, depth int) *Snapshot {
	newSnapshot := createSnapShot(stack.copy(true), stack.copy(false), memory.Copy(), pc, jumpPc, contract.Copy(), depth)
	bc.snapshots = append(bc.snapshots, newSnapshot)
	return newSnapshot
}

func (bc *BranchContext) addReadRecord(callerAddr common.Address, slot *common.Hash) {
	rd, ok := bc.readSet[callerAddr]
	if !ok {
		rd = newRecorder(callerAddr)
		bc.readSet[callerAddr] = rd
	}
	if slot != nil {
		rd.markAccessedStorage(*slot)
	} else {
		rd.markAccessedState()
	}
}

func (bc *BranchContext) addWriteRecord(callerAddr common.Address, slot *common.Hash) {
	rd, ok := bc.writeSet[callerAddr]
	if !ok {
		rd = newRecorder(callerAddr)
		bc.writeSet[callerAddr] = rd
	}
	if slot != nil {
		rd.markAccessedStorage(*slot)
	} else {
		rd.markAccessedState()
	}
}

// Snapshot creates a tmp snapshot of current execution stack and relevant information
type Snapshot struct {
	curTracingStack StackInterface
	curStack        StackInterface
	curMemory       *Memory
	pc              uint64
	jumpPc          uint64 // for tracking jumpi pos value in 'eq' judgement
	contract        *Contract
	depth           int
}

func createSnapShot(tracingStack, stack StackInterface, memory *Memory, pc, jumpPc uint64, contract *Contract, depth int) *Snapshot {
	return &Snapshot{
		curTracingStack: tracingStack,
		curStack:        stack,
		curMemory:       memory,
		pc:              pc,
		jumpPc:          jumpPc,
		contract:        contract,
		depth:           depth,
	}
}

func (sp *Snapshot) UpdatePC(pc uint64)              { sp.pc = pc }
func (sp *Snapshot) GetTracingStack() StackInterface { return sp.curTracingStack }
func (sp *Snapshot) GetStack() StackInterface        { return sp.curStack }
func (sp *Snapshot) GetMemory() *Memory              { return sp.curMemory }
func (sp *Snapshot) GetPC() uint64                   { return sp.pc }
func (sp *Snapshot) GetJumpPC() uint64               { return sp.jumpPc }
func (sp *Snapshot) GetContract() *Contract          { return sp.contract }
func (sp *Snapshot) GetDepth() int                   { return sp.depth }

// SstoreInfo records the updated variable under the current snapshot
type SstoreInfo struct {
	callerAddr common.Address
	updatedVal uint256.Int
	loc        TracingUnit
	val        TracingUnit
	compact    bool
}

func (ss *SstoreInfo) GetCallerAddr() common.Address { return ss.callerAddr }
func (ss *SstoreInfo) GetUpdatedValue() uint256.Int  { return ss.updatedVal }
func (ss *SstoreInfo) GetLocUnit() TracingUnit       { return ss.loc }
func (ss *SstoreInfo) GetValUnit() TracingUnit       { return ss.val }
func (ss *SstoreInfo) GetCompact() bool              { return ss.compact }
func (ss *SstoreInfo) UpdateVal(value uint256.Int)   { ss.updatedVal = value }

func newSstoreInfo(callerAddr common.Address, updatedVal uint256.Int, loc, val TracingUnit, isCompact bool) *SstoreInfo {
	return &SstoreInfo{
		callerAddr: callerAddr,
		updatedVal: updatedVal,
		loc:        loc,
		val:        val,
		compact:    isCompact,
	}
}

type ReadSet map[common.Address]*Recorder
type WriteSet map[common.Address]*Recorder

func CombineRWSet(set1, set2 map[common.Address]*Recorder) {
	for addr, rd2 := range set2 {
		if rd1, ok := set1[addr]; !ok {
			set1[addr] = rd2
		} else {
			if rd2.stateAccessed {
				rd1.stateAccessed = rd2.stateAccessed
			}
			if rd2.storageAccessed {
				rd1.storageAccessed = rd2.storageAccessed
			}
			for slot := range rd2.accessedSlots {
				if _, exist := rd1.accessedSlots[slot]; !exist {
					rd1.accessedSlots[slot] = struct{}{}
				}
			}
		}
	}
}

// Recorder used for subsequent concurrency control (more accurate read/write set)
type Recorder struct {
	address         common.Address
	accessedSlots   map[common.Hash]struct{}
	stateAccessed   bool
	storageAccessed bool
}

func (rd *Recorder) GetAddr() common.Address            { return rd.address }
func (rd *Recorder) GetSlots() map[common.Hash]struct{} { return rd.accessedSlots }
func (rd *Recorder) IsStateAccessed() bool              { return rd.stateAccessed }
func (rd *Recorder) IsStorageAccessed() bool            { return rd.storageAccessed }

func newRecorder(callerAddr common.Address) *Recorder {
	return &Recorder{
		address:         callerAddr,
		accessedSlots:   make(map[common.Hash]struct{}),
		stateAccessed:   false,
		storageAccessed: false,
	}
}

func (rd *Recorder) markAccessedState() {
	rd.stateAccessed = true
}

func (rd *Recorder) markAccessedStorage(slot common.Hash) {
	rd.accessedSlots[slot] = struct{}{}
	rd.storageAccessed = true
}
