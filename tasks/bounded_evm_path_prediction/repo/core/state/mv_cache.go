package state

import (
	"container/list"
	"errors"
	"github.com/ethereum/go-ethereum/common"
	"github.com/holiman/uint256"
	"math/big"
	"strings"
)

// MVCache caches all read/written state variables in branches in a multi-version format
// Only used in pre-execution for ordering-based prediction and pre-execution repair
type MVCache struct {
	Cache   map[common.Address]map[common.Hash]*MVObject
	Dirties map[common.Address]map[common.Hash]struct{}
	Sweeper *Sweeper
	Epoch   int
}

// NewMVCache initiates a new multi-version cache
func NewMVCache(round int, threshold float32) *MVCache {
	return &MVCache{
		Cache:   make(map[common.Address]map[common.Hash]*MVObject),
		Dirties: make(map[common.Address]map[common.Hash]struct{}),
		Sweeper: newSweeper(round, threshold),
		Epoch:   0,
	}
}

// GetObject gets an existing multi-version object
func (mvc *MVCache) GetObject(contractAddr common.Address, slotInt uint256.Int) *MVObject {
	objectMap, ok := mvc.Cache[contractAddr]
	if ok {
		mvo, ok2 := objectMap[slotInt.Bytes32()]
		if ok2 {
			return mvo
		} else {
			return nil
		}
	}
	return nil
}

// GetOrCreateObject gets an existing multi-version object, if not, creates a new one
func (mvc *MVCache) GetOrCreateObject(contractAddr common.Address, slotInt uint256.Int, isCompact bool) *MVObject {
	var mvo *MVObject
	objectMap, ok := mvc.Cache[contractAddr]
	if ok {
		mvo = objectMap[slotInt.Bytes32()]
		if mvo == nil {
			mvo = NewMVObject(mvc.Epoch, isCompact)
			objectMap[slotInt.Bytes32()] = mvo
		}
	} else {
		newMap := make(map[common.Hash]*MVObject)
		mvo = NewMVObject(mvc.Epoch, isCompact)
		newMap[slotInt.Bytes32()] = mvo
		mvc.Cache[contractAddr] = newMap
	}
	return mvo
}

// ClearCache clears all cached multi-version updates at the end of each epoch
func (mvc *MVCache) ClearCache() {
	mvc.Epoch++
	for cAddr, slotMap := range mvc.Dirties {
		for slot := range slotMap {
			mvc.Cache[cAddr][slot].incrementAccesses()
			mvc.Cache[cAddr][slot].reset()
		}
	}
	mvc.Dirties = make(map[common.Address]map[common.Hash]struct{})
	// sweep the cache every $round$ epoch
	if mvc.Epoch%mvc.Sweeper.getRound() == 0 {
		mvc.Sweeper.sweep(mvc.Epoch, mvc.Cache)
	}
}

func (mvc *MVCache) SetStorageForRead(contractAddr common.Address, slotInt uint256.Int, txID common.Hash, tip *big.Int) error {
	obj := mvc.GetOrCreateObject(contractAddr, slotInt, false)
	if obj.slotStorage == nil {
		obj.slotStorage = newMVRecord()
	}
	if err := obj.setStorageForRead(txID, tip); err != nil {
		return err
	}
	mvc.markDirty(contractAddr, slotInt)
	return nil
}

func (mvc *MVCache) SetStorageForWrite(contractAddr common.Address, slotInt, value uint256.Int, txID common.Hash, tip *big.Int) (bool, error) {
	obj := mvc.GetOrCreateObject(contractAddr, slotInt, false)
	if obj.slotStorage == nil {
		obj.slotStorage = newMVRecord()
	}
	repair, err := obj.setStorageForWrite(txID, value, tip)
	if err == nil {
		mvc.markDirty(contractAddr, slotInt)
	}
	return repair, err
}

func (mvc *MVCache) SetCompactedStorageForRead(contractAddr common.Address, slotInt, offset uint256.Int, txID common.Hash, tip *big.Int) error {
	obj := mvc.GetOrCreateObject(contractAddr, slotInt, true)
	if obj.compactedStorage == nil {
		obj.compactedStorage = newCompactedStorage()
	}
	if err := obj.setCompactedStorageForRead(offset, txID, tip); err != nil {
		return err
	}
	mvc.markDirty(contractAddr, slotInt)
	return nil
}

func (mvc *MVCache) SetCompactedStorageForWrite(contractAddr common.Address, slotInt, offset, value uint256.Int, txID common.Hash, tip *big.Int) (bool, error) {
	obj := mvc.GetOrCreateObject(contractAddr, slotInt, true)
	if obj.compactedStorage == nil {
		obj.compactedStorage = newCompactedStorage()
	}
	repair, err := obj.setCompactedStorageForWrite(txID, offset, value, tip)
	if err != nil {
		mvc.markDirty(contractAddr, slotInt)
	}
	return repair, err
}

func (mvc *MVCache) GetStorageVersion(contractAddr common.Address, slotInt uint256.Int, tip *big.Int) (*WriteVersion, error) {
	obj := mvc.GetObject(contractAddr, slotInt)
	if obj == nil {
		return nil, errors.New("not found")
	}
	wVersion, err := obj.getStorageVersion(tip)
	return wVersion, err
}

func (mvc *MVCache) GetCompactedStorageVersion(contractAddr common.Address, slotInt, offset uint256.Int, tip *big.Int) (*WriteVersion, error) {
	obj := mvc.GetObject(contractAddr, slotInt)
	if obj == nil {
		return nil, errors.New("not found")
	}
	wVersion, err := obj.getCompactedStorageVersion(offset, tip)
	return wVersion, err
}

func (mvc *MVCache) GetRepairTXs(contractAddr common.Address, slotInt, offset uint256.Int) ([]common.Hash, error) {
	obj := mvc.GetObject(contractAddr, slotInt)
	if obj == nil {
		return nil, nil
	}
	output, err := obj.getRepairTXs(offset)
	if err != nil {
		return nil, err
	}
	return output, nil
}

// markDirty marks all updated or queried storage slot in a single epoch
func (mvc *MVCache) markDirty(contractAddr common.Address, slotInt uint256.Int) {
	slotMap, ok := mvc.Dirties[contractAddr]
	if !ok {
		slotMap = make(map[common.Hash]struct{})
		slotMap[slotInt.Bytes32()] = struct{}{}
		mvc.Dirties[contractAddr] = slotMap
	} else {
		if _, ok2 := slotMap[slotInt.Bytes32()]; !ok2 {
			slotMap[slotInt.Bytes32()] = struct{}{}
		}
	}
}

// Sweeper sweeps inactive multi-version objects in a round
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

// sweep defines the sweep function of a sweeper
func (s *Sweeper) sweep(epoch int, cache map[common.Address]map[common.Hash]*MVObject) {
	for cAddr, slotMap := range cache {
		if len(slotMap) == 0 {
			delete(cache, cAddr)
			continue
		}
		for slot, obj := range slotMap {
			// sweep the cache every $round$ epoch
			acc := obj.getAccesses()
			lifeCycle := epoch - obj.getCreationEpoch()
			frequency := float32(acc) / float32(lifeCycle)
			if frequency < s.threshold {
				delete(slotMap, slot)
			}
		}
	}
}

func (s *Sweeper) getRound() int         { return s.round }
func (s *Sweeper) getThreshold() float32 { return s.threshold }

type MVObject struct {
	slotStorage      *mvRecord
	compactedStorage *compactedStorage
	isCompact        bool
	accesses         int // record the frequency of accesses for cache sweeping
	creation         int // record the epoch of creation
}

func NewMVObject(epoch int, isCompact bool) *MVObject {
	mvo := &MVObject{}
	if !isCompact {
		mvo.slotStorage = newMVRecord()
	} else {
		mvo.compactedStorage = newCompactedStorage()
	}
	mvo.isCompact = isCompact
	mvo.creation = epoch
	mvo.accesses = 0
	return mvo
}

func (mvo *MVObject) reset() {
	if !mvo.isCompact {
		mvo.slotStorage = newMVRecord()
	} else {
		mvo.compactedStorage = newCompactedStorage()
	}
}

func (mvo *MVObject) getCompact() bool      { return mvo.isCompact }
func (mvo *MVObject) getAccesses() int      { return mvo.accesses }
func (mvo *MVObject) getCreationEpoch() int { return mvo.creation }
func (mvo *MVObject) incrementAccesses()    { mvo.accesses++ }

// setStorageForRead inserts a read record into the correct location of the list
func (mvo *MVObject) setStorageForRead(txID common.Hash, tip *big.Int) error {
	err := mvo.slotStorage.insertReadRecord(txID, tip)
	if err != nil {
		return errors.New("insert operation failed")
	}
	return nil
}

// setStorageForWrite inserts a write record into the correct location of the list
// return the true value of 'repair' indicates that the remaining txs after this tx requires repair operation
func (mvo *MVObject) setStorageForWrite(txID common.Hash, value uint256.Int, tip *big.Int) (bool, error) {
	repair, err := mvo.slotStorage.insertWriteRecord(txID, value, tip)
	return repair, err
}

// setCompactedStorageForRead inserts a read record into the corresponding location of the list (for compacted storage slot)
func (mvo *MVObject) setCompactedStorageForRead(offset uint256.Int, txID common.Hash, tip *big.Int) error {
	rd, ok := mvo.compactedStorage.offsetMap[offset.String()]
	if !ok {
		rd = newMVRecord()
		mvo.compactedStorage.offsetMap[offset.String()] = rd
	}
	err := rd.insertReadRecord(txID, tip)
	if err != nil {
		return errors.New("insert operation failed")
	}
	return nil
}

// setCompactedStorageForWrite inserts a write record into the corresponding location of the list (for compacted storage slot)
// return the true value of 'repair' indicates that the remaining txs after this tx requires repair operation
func (mvo *MVObject) setCompactedStorageForWrite(txID common.Hash, offset, value uint256.Int, tip *big.Int) (bool, error) {
	rd, ok := mvo.compactedStorage.offsetMap[offset.String()]
	if !ok {
		rd = newMVRecord()
		mvo.compactedStorage.offsetMap[offset.String()] = rd
	}
	repair, err := rd.insertWriteRecord(txID, value, tip)
	return repair, err
}

// getStorageVersion obtains the storage version at the corresponding location in the list
func (mvo *MVObject) getStorageVersion(tip *big.Int) (*WriteVersion, error) {
	if mvo.slotStorage == nil {
		return nil, errors.New("not found")
	}
	wVersion, err := mvo.slotStorage.getWriteVersion(tip)
	return wVersion, err
}

// getCompactedStorageVersion obtains the storage version at the corresponding location in the list (for compacted storage slot)
func (mvo *MVObject) getCompactedStorageVersion(offset uint256.Int, tip *big.Int) (*WriteVersion, error) {
	if mvo.compactedStorage == nil {
		return nil, errors.New("not found")
	}
	rd, ok := mvo.compactedStorage.offsetMap[offset.String()]
	if !ok {
		rd = newMVRecord()
		mvo.compactedStorage.offsetMap[offset.String()] = rd
	}
	wVersion, err := rd.getWriteVersion(tip)
	return wVersion, err
}

// getRepairTXs outputs transactions that need to be repaired due to changes in the transaction order
func (mvo *MVObject) getRepairTXs(offset uint256.Int) ([]common.Hash, error) {
	var identicalTXMap = make(map[common.Hash]struct{})
	if offset.Uint64() > 0 {
		rd, ok := mvo.compactedStorage.offsetMap[offset.String()]
		if !ok {
			return nil, errors.New("non-existent version")
		}
		output, err := rd.outputTXsFromRecord(identicalTXMap)
		if err != nil {
			return nil, err
		}
		return output, nil
	} else {
		output, err := mvo.slotStorage.outputTXsFromRecord(identicalTXMap)
		if err != nil {
			return nil, err
		}
		return output, nil
	}
}

type compactedStorage struct {
	offsetMap map[string]*mvRecord
}

func newCompactedStorage() *compactedStorage {
	return &compactedStorage{
		offsetMap: make(map[string]*mvRecord),
	}
}

type mvRecord struct {
	rRecord    *list.List
	wRecord    *list.List
	rLoc       *list.Element // latest location of read (itself)
	wLoc       *list.Element // latest location of write (itself)
	readRepair bool
}

func newMVRecord() *mvRecord {
	rRecord := list.New()
	wRecord := list.New()
	return &mvRecord{
		rRecord:    rRecord,
		wRecord:    wRecord,
		rLoc:       nil,
		wLoc:       nil,
		readRepair: false,
	}
}

func (r *mvRecord) appendReadRecord(txID common.Hash, tip *big.Int) {
	newVer := &ReadVersion{txID: txID, tip: tip}
	r.rRecord.PushBack(newVer)
}

func (r *mvRecord) appendWriteRecord(txID common.Hash, value uint256.Int, tip *big.Int) {
	newVer := &WriteVersion{txID: txID, value: value, tip: tip}
	r.wRecord.PushBack(newVer)
}

func (r *mvRecord) insertReadRecord(txID common.Hash, tip *big.Int) error {
	newVer := &ReadVersion{txID: txID, tip: tip}
	counter := r.rRecord.Len()
	if counter == 0 {
		r.rRecord.PushBack(newVer)
		return nil
	}
	for e := r.rRecord.Back(); e != nil; e = e.Prev() {
		ver, ok := e.Value.(*ReadVersion)
		if !ok {
			return errors.New("wrong version format")
		}
		if (tip.Cmp(ver.GetTip()) == -1) || (tip.Cmp(ver.GetTip()) == 0 && strings.Compare(txID.String(), ver.txID.String()) != 0) {
			// 待插入交易的交易费小于某个交易元素或者两个交易的交易费相同，插入到此交易元素之后
			r.rRecord.InsertAfter(newVer, e)
			break
		} else if counter == 1 && tip.Cmp(ver.GetTip()) == 1 {
			// 插入列表的第一位
			r.rRecord.InsertBefore(newVer, e)
			break
		} else if strings.Compare(txID.String(), ver.txID.String()) == 0 {
			// 发现列表中已经插入该交易元素，退出
			break
		}
		counter--
	}
	return nil
}

// checkReadRecord used for repair check
func (r *mvRecord) checkReadRecord(tip *big.Int) (bool, error) {
	counter := r.rRecord.Len()
	if counter == 0 {
		return false, nil
	}
	for e := r.rRecord.Back(); e != nil; e = e.Prev() {
		ver, ok := e.Value.(*ReadVersion)
		if !ok {
			return false, errors.New("wrong version format")
		}
		if (tip.Cmp(ver.GetTip()) == -1) || (tip.Cmp(ver.GetTip()) == 0) {
			r.rLoc = e.Next()
			if r.rLoc != nil {
				return true, nil
			}
			break
		} else if counter == 1 && tip.Cmp(ver.GetTip()) == 1 {
			r.rLoc = e
			return true, nil
		}
		counter--
	}
	return false, nil
}

// 插入写版本，队尾还有元素，修复队尾那些交易，并修复所有读了队尾那些交易的交易 (不考虑级联修复)；
// 并检查是否有交易读了比此版本小的版本，如果交易的排序更后的话，修复这些交易
// 综上，只需要检查读版本中是否有交易排序在插入写版本交易的后面，这些交易将被修复
func (r *mvRecord) insertWriteRecord(txID common.Hash, value uint256.Int, tip *big.Int) (bool, error) {
	readRepair, err := r.checkReadRecord(tip)
	r.readRepair = readRepair
	if err != nil {
		return false, err
	}
	newVer := &WriteVersion{txID: txID, value: value, tip: tip}
	counter := r.wRecord.Len()
	if counter == 0 {
		curE := r.wRecord.PushBack(newVer)
		r.wLoc = curE
		return false, nil
	}
	for e := r.wRecord.Back(); e != nil; e = e.Prev() {
		ver, ok := e.Value.(*WriteVersion)
		if !ok {
			return false, errors.New("wrong version format")
		}
		if (tip.Cmp(ver.GetTip()) == -1) || (tip.Cmp(ver.GetTip()) == 0 && strings.Compare(txID.String(), ver.txID.String()) != 0) {
			// 待插入交易的交易费小于某个交易元素或者两个交易的交易费相同，插入到此交易元素之后
			curE := r.wRecord.InsertAfter(newVer, e)
			r.wLoc = curE
			// 通知需要将后面的交易进行预执行修复
			if curE.Next() != nil || r.readRepair {
				return true, nil
			}
			break
		} else if counter == 1 && tip.Cmp(ver.GetTip()) == 1 {
			// 插入列表的第一位
			curE := r.wRecord.InsertBefore(newVer, e)
			r.wLoc = curE
			// 通知需要将后面的交易进行预执行修复
			return true, nil
		} else if strings.Compare(txID.String(), ver.txID.String()) == 0 {
			// 发现列表中已经插入该交易元素，修改版本信息（写入值）
			ver.value = value
			break
		}
		counter--
	}
	return false, nil
}

func (r *mvRecord) getWriteVersion(tip *big.Int) (*WriteVersion, error) {
	for e := r.wRecord.Back(); e != nil; e = e.Prev() {
		ver, ok := e.Value.(*WriteVersion)
		if !ok {
			return nil, errors.New("wrong version format")
		}
		if tip.Cmp(ver.GetTip()) == -1 || tip.Cmp(ver.GetTip()) == 0 {
			// 遇到费用比自己大的交易或者费用相等的交易，直接读取其写入的版本
			return ver, nil
		}
	}
	return nil, errors.New("not found")
}

func (r *mvRecord) getLatestReadVersion() (*ReadVersion, error) {
	element := r.rRecord.Back()
	if element == nil {
		return nil, errors.New("not found")
	}
	ver, ok := element.Value.(*ReadVersion)
	if !ok {
		return nil, errors.New("wrong version format")
	}
	return ver, nil
}

func (r *mvRecord) getLatestWriteVersion() (*WriteVersion, error) {
	element := r.wRecord.Back()
	if element == nil {
		return nil, errors.New("not found")
	}
	ver, ok := element.Value.(*WriteVersion)
	if !ok {
		return nil, errors.New("wrong version format")
	}
	return ver, nil
}

func (r *mvRecord) outputTXsFromRecord(txMap map[common.Hash]struct{}) ([]common.Hash, error) {
	var output []common.Hash
	//var sortedFees []int
	//feeMap := make(map[int][]common.Hash)

	if r.wLoc != nil {
		for e := r.wLoc.Next(); e != nil; e = e.Next() {
			ver, ok := e.Value.(*WriteVersion)
			if !ok {
				return nil, errors.New("wrong version format")
			}
			if _, exist := txMap[ver.GetID()]; !exist {
				//fee := int(ver.GetTip().Int64())
				//sortedFees = append(sortedFees, fee)
				//feeMap[fee] = append(feeMap[fee], ver.GetID())
				txMap[ver.GetID()] = struct{}{}
				output = append(output, ver.GetID())
			}
		}
	}

	if r.readRepair && r.rLoc != nil {
		r.readRepair = false
		for e := r.rLoc; e != nil; e = e.Next() {
			ver, ok := e.Value.(*ReadVersion)
			if !ok {
				return nil, errors.New("wrong version format")
			}
			if _, exist := txMap[ver.GetID()]; !exist {
				//fee := int(ver.GetTip().Int64())
				//sortedFees = append(sortedFees, fee)
				//feeMap[fee] = append(feeMap[fee], ver.GetID())
				txMap[ver.GetID()] = struct{}{}
				output = append(output, ver.GetID())
			}
		}
	}

	//sort.Ints(sortedFees)
	//for _, fee := range sortedFees {
	//	txs := feeMap[fee]
	//	output = append(output, txs...)
	//}
	return output, nil
}

type WriteVersion struct {
	txID  common.Hash
	value uint256.Int
	tip   *big.Int
}

func (wv *WriteVersion) GetID() common.Hash  { return wv.txID }
func (wv *WriteVersion) GetVal() uint256.Int { return wv.value }
func (wv *WriteVersion) GetTip() *big.Int    { return wv.tip } // Get the gas fee of transaction

type ReadVersion struct {
	txID common.Hash
	tip  *big.Int
}

func (rv *ReadVersion) GetID() common.Hash { return rv.txID }
func (rv *ReadVersion) GetTip() *big.Int   { return rv.tip } // Get the gas fee of transaction
