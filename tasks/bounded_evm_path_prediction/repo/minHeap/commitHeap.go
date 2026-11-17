package minHeap

import (
	"container/heap"
	"seerEVM/core/state"
	"sync"
)

type CommitItem struct {
	Index          int
	Incarnation    int
	StorageVersion int
	LastReads      []*state.ReadLoc
	LastWrites     state.WriteSets
}

func NewCommitItem(Index int, Incarnation int, StorageVersion int, ReadSet []*state.ReadLoc, WriteSet state.WriteSets) *CommitItem {
	return &CommitItem{
		Index:          Index,
		Incarnation:    Incarnation,
		StorageVersion: StorageVersion,
		LastReads:      ReadSet,
		LastWrites:     WriteSet,
	}
}

// CommitHeap 一个并发安全的最小堆，堆中元素为 CommitItem，Index越小排在越前面，Index相等时StorageVersion越小排在越前面
type CommitHeap struct {
	commitHeap *commitHeap
	mutex      sync.Mutex
}

func NewCommitHeap() *CommitHeap {
	h := &CommitHeap{
		commitHeap: new(commitHeap),
	}
	heap.Init(h.commitHeap)

	return h
}

func (h *CommitHeap) Push(nodeToPush *CommitItem) {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	heap.Push(h.commitHeap, nodeToPush)
}

func (h *CommitHeap) Pop() *CommitItem {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.commitHeap.Len() <= 0 {
		return nil
	}
	newNode := heap.Pop(h.commitHeap).(*CommitItem)
	return newNode
}

type commitHeap []*CommitItem

func (h commitHeap) Len() int {
	return len(h)
}

func (h commitHeap) Less(i, j int) bool {
	return h[i].Index < h[j].Index || (h[i].Index == h[j].Index && h[i].StorageVersion < h[j].StorageVersion)
}

func (h commitHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *commitHeap) Push(x interface{}) {
	item := x.(*CommitItem)
	*h = append(*h, item)
}

func (h *commitHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}
