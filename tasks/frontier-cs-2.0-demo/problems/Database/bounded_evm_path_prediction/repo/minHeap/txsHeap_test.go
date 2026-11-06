package minHeap

import (
	"testing"
)

func Test(t *testing.T) {
	txsH := NewTxsHeap()

	go func() {
		for i := 0; i < 1000; i++ {
			txsH.Push(i, i, i)
		}
	}()
	go func() {
		for i := 0; i < 1000; i++ {
			txsH.Push(i, i, 1000-i)
		}
	}()
	go func() {
		for i := 0; i < 1000; i++ {
			txsH.Push(1000-i, 1000-i, i)
		}
	}()
	go func() {
		for i := 0; i < 1000; i++ {
			txsH.Push(1000-i, 1000-i, 1000-i)
		}
	}()

	for i := 0; ; i++ {
		item := txsH.Pop()
		if item == nil {
			break
		}
		t.Logf("get item %d: %+v\n", i, item)
	}
}
