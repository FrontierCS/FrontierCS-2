// Copyright 2014 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package vm

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/holiman/uint256"
	"sync"
)

type StackInterface interface {
	push(d *uint256.Int)
	peek() (*uint256.Int, TracingUnit)
	pop() (uint256.Int, TracingUnit)
	swap(n int)
	dup(n int)
	Back(n int) *uint256.Int
	len(isData bool) int
	Data() []uint256.Int
	returnStack()
	override(tu TracingUnit)
	updateUnit(t int, slot, offset, val, blockNum uint256.Int, blockEnv string, balAddr common.Address) *StateUnit
	BitLen(loc int) int
	Tracer() []TracingUnit
	UpdatePeek(e uint256.Int)
	copy(stackTracing bool) StackInterface
}

var stackPool = sync.Pool{
	New: func() interface{} {
		return &Stack{data: make([]uint256.Int, 0, 16)}
	},
}

// Stack is an object for basic stack operations. Items popped to the stack are
// expected to be changed and modified. stack does not take care of adding newly
// initialised objects.
type Stack struct {
	data []uint256.Int
}

func newstack() *Stack {
	//return stackPool.Get().(*Stack)
	return &Stack{data: make([]uint256.Int, 0, 16)}
}

func (st *Stack) returnStack() {
	st.data = st.data[:0]
	//stackPool.Put(st)
}

// Data returns the underlying uint256.Int array.
func (st *Stack) Data() []uint256.Int {
	return st.data
}

func (st *Stack) push(d *uint256.Int) {
	// NOTE push limit (1024) is checked in baseCheck
	st.data = append(st.data, *d)
}

func (st *Stack) pop() (uint256.Int, TracingUnit) {
	ret := st.data[st.len(true)-1]
	st.data = st.data[:st.len(true)-1]
	return ret, nil
}

func (st *Stack) len(isData bool) int {
	return len(st.data)
}

func (st *Stack) swap(n int) {
	st.data[st.len(true)-n], st.data[st.len(true)-1] = st.data[st.len(true)-1], st.data[st.len(true)-n]
}

func (st *Stack) dup(n int) {
	st.push(&st.data[st.len(true)-n])
}

func (st *Stack) peek() (*uint256.Int, TracingUnit) {
	return &st.data[st.len(true)-1], nil
}

func (st *Stack) Tracer() []TracingUnit {
	return []TracingUnit{}
}

func (st *Stack) updateUnit(t int, slot, offset, val, blockNum uint256.Int, blockEnv string, balAddr common.Address) *StateUnit {
	return nil
}

// UpdatePeek updates the value of the peek unit
func (st *Stack) UpdatePeek(e uint256.Int) {
	if st.len(true) > 0 {
		st.data[st.len(true)-1] = e
	}
}

func (st *Stack) copy(stackTracing bool) StackInterface {
	newStack := &Stack{
		data: make([]uint256.Int, len(st.data)),
	}
	copy(newStack.data, st.data)
	return newStack
}

func (st *Stack) override(tu TracingUnit) {}

// Back returns the n'th item in stack
func (st *Stack) Back(n int) *uint256.Int {
	return &st.data[st.len(true)-n-1]
}

func (st *Stack) BitLen(loc int) int {
	return st.data[loc].BitLen()
}

var stackPool2 = sync.Pool{
	New: func() interface{} {
		return &PreStack{
			data:   make([]uint256.Int, 0, 16),
			tracer: make([]TracingUnit, 0, 16),
		}
	},
}

// PreStack is an object for stack operations during pre-execution
// It contains tracing unit to track the state variable and relevant operations
type PreStack struct {
	data   []uint256.Int
	tracer []TracingUnit
}

func newPreStack() *PreStack {
	//return stackPool2.Get().(*PreStack)
	return &PreStack{
		data:   make([]uint256.Int, 0, 16),
		tracer: make([]TracingUnit, 0, 16),
	}
}

func (ps *PreStack) returnStack() {
	ps.data = ps.data[:0]
	ps.tracer = ps.tracer[:0]
	//stackPool2.Put(ps)
}

// Data returns the underlying uint256.Int array.
func (ps *PreStack) Data() []uint256.Int {
	return ps.data
}

func (ps *PreStack) Tracer() []TracingUnit {
	return ps.tracer
}

// UpdatePeek updates the value of the peek unit
func (ps *PreStack) UpdatePeek(e uint256.Int) {
	ps.data[ps.len(true)-1] = e
	ps.tracer[ps.len(false)-1].SetValue(e)
}

func (ps *PreStack) push(d *uint256.Int) {
	// NOTE push limit (1024) is checked in baseCheck
	// PUSH opcode normally pushes non-tracking units
	ps.data = append(ps.data, *d)
	normalUnit := NewNormalUnit(*d)
	ps.tracer = append(ps.tracer, normalUnit)
}

func (ps *PreStack) pop() (uint256.Int, TracingUnit) {
	reData := ps.data[ps.len(true)-1]
	reUnit := ps.tracer[ps.len(false)-1]
	ps.data = ps.data[:ps.len(true)-1]
	ps.tracer = ps.tracer[:ps.len(false)-1]
	return reData, reUnit
}

func (ps *PreStack) len(isData bool) int {
	if isData {
		return len(ps.data)
	}
	return len(ps.tracer)
}

func (ps *PreStack) swap(n int) {
	ps.data[ps.len(true)-n], ps.data[ps.len(true)-1] = ps.data[ps.len(true)-1], ps.data[ps.len(true)-n]
	ps.tracer[ps.len(false)-n], ps.tracer[ps.len(false)-1] = ps.tracer[ps.len(false)-1], ps.tracer[ps.len(false)-n]
}

func (ps *PreStack) dup(n int) {
	//st.push(&st.data[st.len()-n])
	d := ps.data[ps.len(true)-n]
	ps.data = append(ps.data, d)
	tu := ps.tracer[ps.len(false)-n]
	switch unit := tu.(type) {
	case *NormalUnit:
		newUnit := unit.Copy()
		ps.tracer = append(ps.tracer, newUnit)
	case *CallDataUnit:
		newUnit := unit.Copy()
		ps.tracer = append(ps.tracer, newUnit)
	case *StateUnit:
		newUnit := unit.Copy()
		ps.tracer = append(ps.tracer, newUnit)
	}
}

func (ps *PreStack) peek() (*uint256.Int, TracingUnit) {
	return &ps.data[ps.len(true)-1], ps.tracer[ps.len(false)-1]
}

func (ps *PreStack) updateUnit(t int, slot, offset, val, blockNum uint256.Int, blockEnv string, balAddr common.Address) *StateUnit {
	switch t {
	case INPUT:
		cu := NewCallDataUnit(val, offset)
		ps.tracer[ps.len(false)-1] = cu
		return nil
	case STATE:
		su := NewStateUnit(slot, val, blockNum, blockEnv, balAddr)
		ps.tracer[ps.len(false)-1] = su
		return su
	default:
		return nil
	}
}

func (ps *PreStack) copy(stackTracing bool) StackInterface {
	if stackTracing {
		newStack := &PreStack{
			data:   make([]uint256.Int, len(ps.data)),
			tracer: make([]TracingUnit, len(ps.tracer)),
		}
		copy(newStack.data, ps.data)
		copy(newStack.tracer, ps.tracer)
		return newStack
	} else {
		newStack := &Stack{
			data: make([]uint256.Int, len(ps.data)),
		}
		copy(newStack.data, ps.data)
		return newStack
	}
}

func (ps *PreStack) override(tu TracingUnit) {
	ps.tracer[ps.len(false)-1] = tu
}

// Back returns the n'th item in stack
func (ps *PreStack) Back(n int) *uint256.Int {
	return &ps.data[ps.len(true)-n-1]
}

func (ps *PreStack) BitLen(loc int) int {
	return ps.data[loc].BitLen()
}
