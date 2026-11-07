package minHeap

import (
	"container/heap"
	"sync"
)

type ReadyItem struct {
	Index          int
	Incarnation    int
	StorageVersion int
	// 该任务是否是模拟生成读写集的任务（如果是的话，线程就会用publicStateDB2来执行交易）
	IsSimulation bool
}

// ReadyHeap 一个并发安全的最小堆，堆中元素为 ReadyItem，Index越小排在越前面，Index相等时StorageVersion越小排在越前面
type ReadyHeap struct {
	readyHeap *readyHeap
	mutex     sync.Mutex
}

func NewReadyHeap() *ReadyHeap {
	h := &ReadyHeap{
		readyHeap: new(readyHeap),
	}
	heap.Init(h.readyHeap)

	return h
}

func (h *ReadyHeap) Push(Index int, Incarnation int, StorageVersion int, IsSimulation bool) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	nodeToPush := &ReadyItem{
		Index:          Index,
		Incarnation:    Incarnation,
		StorageVersion: StorageVersion,
		IsSimulation:   IsSimulation,
	}
	heap.Push(h.readyHeap, nodeToPush)
}

func (h *ReadyHeap) Pop() *ReadyItem {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.readyHeap.Len() <= 0 {
		return nil
	}
	newNode := heap.Pop(h.readyHeap).(*ReadyItem)
	return newNode
}

type readyHeap []*ReadyItem

func (h readyHeap) Len() int {
	return len(h)
}

func (h readyHeap) Less(i, j int) bool {
	return h[i].Index < h[j].Index || (h[i].Index == h[j].Index && h[i].StorageVersion < h[j].StorageVersion)
}

func (h readyHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *readyHeap) Push(x interface{}) {
	item := x.(*ReadyItem)
	*h = append(*h, item)
}

func (h *readyHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}
