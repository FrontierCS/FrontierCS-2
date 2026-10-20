package types

import (
	"github.com/ethereum/go-ethereum/common"
)

type AccessSlot struct {
	IsRead  bool // 未读就是false
	IsWrite bool // 未写为false
}

type AccessSlotMap map[common.Hash]*AccessSlot

func NewAccessSlot() *AccessSlot {
	return &AccessSlot{
		IsRead:  false,
		IsWrite: false,
	}
}
func newAccessSlotMap() *AccessSlotMap {
	m := make(AccessSlotMap)
	return &m
}

type AccessAddress struct {
	Slots *AccessSlotMap
	//Slots   AccessSlotMap
	IsRead  bool // 未读就是false
	IsWrite bool // 未写为false
}

type AccessAddressMap map[common.Address]*AccessAddress

func NewAccessAddress() *AccessAddress {
	return &AccessAddress{
		Slots: newAccessSlotMap(),
		//Slots:   make(AccessSlotMap),
		IsRead:  false,
		IsWrite: false,
	}
}

func NewAccessAddressMap() *AccessAddressMap {
	m := make(AccessAddressMap)
	return &m
}
