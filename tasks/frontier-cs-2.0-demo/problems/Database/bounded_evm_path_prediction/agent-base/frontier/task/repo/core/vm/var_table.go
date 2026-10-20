package vm

import (
	"bufio"
	"errors"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/holiman/uint256"
	"os"
)

const sweepRound = 70
const statisticsRound = 200
const threshold = 0.1

// VarTable records the information of branch relevant to state variables
type VarTable struct {
	Space map[common.Address]*SubTable
	Epoch int
	// for test the ratio of state variable-related branches
	BranchNum            int
	StateRelatedForCTxs0 int
	StateRelatedForCTxs1 int
	StateRelatedForCTxs2 int
	StateRelatedForCTxs3 int
	StateRelated         int
	txNum                int
}

func CreateNewTable() *VarTable {
	return &VarTable{
		Space:                make(map[common.Address]*SubTable),
		Epoch:                0,
		BranchNum:            0,
		StateRelatedForCTxs0: 0,
		StateRelatedForCTxs1: 0,
		StateRelatedForCTxs2: 0,
		StateRelatedForCTxs3: 0,
		StateRelated:         0,
		txNum:                0,
	}
}

func (vt *VarTable) GetSubTable(address common.Address) (*SubTable, error) {
	subTable, ok := vt.Space[address]
	if !ok {
		return nil, errors.New("cannot find contract name")
	}
	return subTable, nil
}

func (vt *VarTable) InsertSubTable(address common.Address) *SubTable {
	subTable := createNewSubTable()
	vt.Space[address] = subTable
	return subTable
}

// VarExist checks whether a state variable is relevant to branches
func (vt *VarTable) VarExist(address common.Address, slot common.Hash, offset uint256.Int) bool {
	subTable, err := vt.GetSubTable(address)
	if err != nil {
		return false
	}
	entry, err2 := subTable.GetEntry(slot)
	if err2 != nil {
		return false
	}
	_, ok := entry.space[offset.String()]
	if !ok {
		return false
	}
	return true
}

// AddHistory supplies the latest observed branch direction to predictor state.
func (vt *VarTable) AddHistory(branch *BranchContext) error {
	contractAddr := branch.GetAddr()
	sub, err := vt.GetSubTable(contractAddr)
	if err != nil {
		return err
	}
	sUnit, _ := branch.GetStateUnit().(*StateUnit)
	slot := sUnit.GetSlot()
	entry, err2 := sub.GetEntry(slot.Bytes32())
	if err2 != nil {
		return err2
	}
	offset := branch.GetStateUnit().GetOffset()
	branchID := branch.GetBranchID()
	direction := branch.GetBranchDirection()
	res := Bool2branchRes(direction)
	entry.Update(offset, branchID, res)
	return nil
}

func (vt *VarTable) GetEpoch() int { return vt.Epoch }

// Sweep sweeps the storage space every $sweepRound$ epoch
func (vt *VarTable) Sweep() {
	vt.Epoch++
	if vt.Epoch%sweepRound == 0 {
		for cAddr, subTable := range vt.Space {
			if len(subTable.Space) == 0 {
				delete(vt.Space, cAddr)
			}
			for slot, entry := range subTable.Space {
				if len(entry.space) == 0 {
					delete(subTable.Space, slot)
				}
				entry.sweeper.sweep(vt.Epoch, entry.space)
			}
		}
	}
}

// Statistics calculates the ratio of state variable-related branches and the ratio of regular branches every $statisticsRound$ epoch
func (vt *VarTable) Statistics(txNum int, fileName string) {
	vt.txNum += txNum
	file, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("open error: %v\n", err)
	}
	defer file.Close()

	if vt.Epoch%statisticsRound == 0 {
		writer := bufio.NewWriter(file)
		fmt.Fprintf(writer, "The ratio of state variable-related branches is: %.2f\n", float64(vt.StateRelated)/float64(vt.BranchNum))
		fmt.Fprintf(writer, "The average number of state variable-related branches each tx is: %.2f\n", float64(vt.StateRelated)/float64(vt.txNum))
		vt.BranchNum = 0
		vt.StateRelated = 0
		vt.txNum = 0
		regular := 0
		totalNum := 0
		for _, subTable := range vt.Space {
			for _, entry := range subTable.Space {
				for _, branchMap := range entry.space {
					for _, branch := range branchMap {
						length := branch.predictor.observationCount
						if length == WARMUP_OBSERVATIONS {
							totalNum++
							if branch.isRegular() {
								regular++
							}
						}
					}
				}
			}
		}
		fmt.Fprintf(writer, "The ratio of regular branches is: %.2f\n", float64(regular)/float64(totalNum))

		err = writer.Flush()
		if err != nil {
			fmt.Printf("flush error: %v\n", err)
		}
	}
}

// SubTable defines a sub-table to manage the storage of all slots relevant to branches under a contract address
type SubTable struct {
	Space map[common.Hash]*Entry
}

func (st *SubTable) GetEntry(slot common.Hash) (*Entry, error) {
	entry, ok := st.Space[slot]
	if !ok {
		return nil, errors.New("cannot find slot variable")
	}
	return entry, nil
}

func (st *SubTable) InsertEntry(slot common.Hash, isCompact bool) *Entry {
	entry := createNewEntry(isCompact)
	st.Space[slot] = entry
	return entry
}

func createNewSubTable() *SubTable {
	return &SubTable{
		Space: make(map[common.Hash]*Entry),
	}
}

// Entry defines an entry for storing all branch information relevant to a storage slot
type Entry struct {
	// branch的命名方式: 函数签名 + 值 (常数) / 函数签名 + slot + offset (状态变量)
	space     map[string]map[string]*branchInfo
	isCompact bool
	sweeper   *Sweeper
}

func (en *Entry) BranchExist(offset uint256.Int, branchID string) bool {
	branchMap, ok := en.space[offset.String()]
	if !ok {
		return false
	}
	_, ok2 := branchMap[branchID]
	if !ok2 {
		return false
	}
	return true
}

func (en *Entry) GenerateBranchInfo(branchID string, offset, value uint256.Int, isVar bool, varInfo *VarInfo, epoch int) {
	branch := newBranchInfo(branchID, value, isVar, varInfo, epoch)
	branchMap, ok := en.space[offset.String()]
	if ok {
		if _, ok2 := branchMap[branchID]; !ok2 {
			branchMap[branchID] = branch
		}
	} else {
		branchMap = make(map[string]*branchInfo)
		branchMap[branchID] = branch
		en.space[offset.String()] = branchMap
	}
}

//func (en *Entry) addHistory(offset uint256.Int, branchID string, taken bool) {
//	branchMap := en.space[offset.String()]
//	branch := branchMap[branchID]
//	branch.history = append(branch.history, taken)
//	// 如果分支历史满了，删除第一个元素，实现实时更新
//	if len(branch.history) > historyLen {
//		branch.history = branch.history[1:]
//	}
//}
//
//func (en *Entry) GetHistory(offset uint256.Int, branchID string) ([]bool, error) {
//	if en.BranchExist(offset, branchID) {
//		branchMap := en.space[offset.String()]
//		branch := branchMap[branchID]
//		return branch.history, nil
//	}
//	return nil, errors.New("cannot find branch history")
//}

// Predict calls the predict function of the predictor model
func (en *Entry) Predict(offset uint256.Int, branchID string) int {
	branchMap := en.space[offset.String()]
	branch, ok := branchMap[branchID]
	if ok {
		model := branch.getPredictor()
		res := model.predict(false)
		model.lastPred = append(model.lastPred, res)
		if res == taken || res == notTaken {
			branchMap[branchID].regular = true
		} else {
			branchMap[branchID].regular = false
		}
		return res
	}
	return uncertain
}

// Update supplies an observed outcome to predictor state.
func (en *Entry) Update(offset uint256.Int, branchID string, dir int) {
	branchMap := en.space[offset.String()]
	branch, ok := branchMap[branchID]
	if ok {
		model := branch.getPredictor()
		if len(model.lastPred) == 0 {
			model.update(dir, model.predict(true))
		} else {
			model.update(dir, model.lastPred[0])
		}
	}
}

func (en *Entry) GetJudgementVal(offset uint256.Int, branchID string) (*VarInfo, uint256.Int) {
	branchMap := en.space[offset.String()]
	branch := branchMap[branchID]
	if branch.isVar {
		return branch.varInfo, branch.value
	}
	return nil, branch.value
}

func (en *Entry) UpdateJudgementVal(judgementVal, updatedVal, offset uint256.Int, branchID string) {
	if updatedVal != judgementVal {
		branchMap := en.space[offset.String()]
		branch := branchMap[branchID]
		branch.value = updatedVal
	}
}

func createNewEntry(isCompact bool) *Entry {
	return &Entry{
		space:     make(map[string]map[string]*branchInfo),
		isCompact: isCompact,
		sweeper:   newSweeper(sweepRound, threshold),
	}
}

// Sweeper sweeps inactive branch information in a round
type Sweeper struct {
	round     int
	threshold float32
}

func newSweeper(round int, threshold float32) *Sweeper {
	return &Sweeper{
		round:     round,
		threshold: threshold,
	}
}

// Sweep defines the sweep function of a sweeper
func (s *Sweeper) sweep(epoch int, entry map[string]map[string]*branchInfo) {
	for offset, branchMap := range entry {
		if len(branchMap) == 0 {
			delete(entry, offset)
			continue
		}
		for id, branch := range branchMap {
			frequency := branch.getFrequency()
			lifeCycle := epoch - branch.getCreationEpoch()
			ratio := float32(frequency) / float32(lifeCycle)
			if ratio < s.threshold {
				delete(branchMap, id)
			}
		}
	}
}

func (s *Sweeper) getRound() int         { return s.round }
func (s *Sweeper) getThreshold() float32 { return s.threshold }

// branchInfo defines a struct for storing relevant branch information
type branchInfo struct {
	id        string
	value     uint256.Int
	isVar     bool
	varInfo   *VarInfo
	predictor *predictor // online state for this branch identity
	creation  int        // record the epoch of creation
	regular   bool       // record if the branch has a regular direction history
}

type VarInfo struct {
	slot   uint256.Int
	offset uint256.Int
	bits   int
}

func NewVarInfo(slot, offset uint256.Int, bits int) *VarInfo {
	return &VarInfo{
		slot:   slot,
		offset: offset,
		bits:   bits,
	}
}

func newBranchInfo(id string, value uint256.Int, isVar bool, varInfo *VarInfo, creation int) *branchInfo {
	return &branchInfo{
		id:        id,
		value:     value,
		isVar:     isVar,
		varInfo:   varInfo,
		predictor: initPredictor(),
		creation:  creation,
		regular:   false,
	}
}

func (br *branchInfo) isRegular() bool          { return br.regular }
func (br *branchInfo) getPredictor() *predictor { return br.predictor }
func (br *branchInfo) getCreationEpoch() int    { return br.creation }
func (br *branchInfo) getFrequency() int        { return br.getPredictor().observationCount }

// GenerateBranchID generates a unique ID for a branch
func GenerateBranchID(funcSig string, isVar bool, varInfo *VarInfo, value uint256.Int) string {
	var branchID string
	if isVar {
		slot := varInfo.slot
		offset := varInfo.offset
		branchID = funcSig + slot.String() + offset.String()
	} else {
		branchID = funcSig + value.String()
	}
	return branchID
}
