package core

import (
	"github.com/ethereum/go-ethereum/common"
	"math"
	"seerEVM/core/types"
	"seerEVM/core/vm"
)

// Recorder defines a speedup recorder and calculator (for evaluation)
type Recorder struct {
	serialLatency      map[common.Hash]int64
	seerLatency        map[common.Hash]int64
	ctxPreLatency      int64
	ntxPreLatency      int64
	speedupNormal      map[int]int
	speedupComplicated map[int]int
	complicatedTxs     map[common.Hash]struct{}
	validTxNum         int
	normalTxNum        int
	complicatedTxNum   int
	performanceObserve bool
}

func NewRecorder(performanceObserve bool) *Recorder {
	return &Recorder{
		serialLatency:      make(map[common.Hash]int64),
		seerLatency:        make(map[common.Hash]int64),
		ctxPreLatency:      0,
		ntxPreLatency:      0,
		speedupNormal:      make(map[int]int),
		speedupComplicated: make(map[int]int),
		complicatedTxs:     make(map[common.Hash]struct{}),
		validTxNum:         0,
		normalTxNum:        0,
		complicatedTxNum:   0,
		performanceObserve: performanceObserve,
	}
}

func (r *Recorder) SerialRecord(tx *types.Transaction, stateDB vm.StateDB, latency int64) {
	if tx.To() != nil && IsContract(stateDB, tx.To()) {
		r.serialLatency[tx.Hash()] = latency
	}
}

func (r *Recorder) SeerRecord(tx *types.Transaction, latency int64) {
	r.seerLatency[tx.Hash()] = latency
}

func (r *Recorder) RecordCtxPreLatency(latency int64) {
	r.ctxPreLatency += latency
}

func (r *Recorder) RecordNtxPreLatency(latency int64) {
	r.ntxPreLatency += latency
}

func (r *Recorder) ChangeObserveIndicator() {
	r.performanceObserve = !r.performanceObserve
}

func (r *Recorder) ResetPreLatency() {
	r.ctxPreLatency = 0
	r.ntxPreLatency = 0
}

func (r *Recorder) SpeedupCalculation() (map[int]int, map[int]int) {
	for tx, lat := range r.seerLatency {
		serialLat := r.serialLatency[tx]
		// 排除因执行失败导致时延为0的交易
		if lat != 0 && serialLat != 0 {
			r.validTxNum++
			speedup := float64(serialLat) / float64(lat)
			if _, ok := r.complicatedTxs[tx]; ok {
				r.complicatedTxNum++
				if speedup < 1 {
					r.speedupComplicated[0]++
				} else {
					speedupInt := int(math.Floor(speedup + 0.5))
					r.speedupComplicated[speedupInt]++
				}
			} else {
				r.normalTxNum++
				if speedup < 1 {
					r.speedupNormal[0]++
				} else {
					speedupInt := int(math.Floor(speedup + 0.5))
					r.speedupNormal[speedupInt]++
				}
			}
		}
	}
	return r.speedupComplicated, r.speedupNormal
}

func (r *Recorder) GetCtxPreLatency() int64                     { return r.ctxPreLatency }
func (r *Recorder) GetNtxPreLatency() int64                     { return r.ntxPreLatency }
func (r *Recorder) GetObserveIndicator() bool                   { return r.performanceObserve }
func (r *Recorder) GetValidTxNum() int                          { return r.validTxNum }
func (r *Recorder) GetNormalTxNum() int                         { return r.normalTxNum }
func (r *Recorder) GetComplicatedTxNum() int                    { return r.complicatedTxNum }
func (r *Recorder) GetComplicatedMap() map[common.Hash]struct{} { return r.complicatedTxs }
func (r *Recorder) ResetComplicatedMap()                        { r.complicatedTxs = make(map[common.Hash]struct{}) }
func (r *Recorder) AppendComplicatedTx(txHash common.Hash) {
	r.complicatedTxs[txHash] = struct{}{}
}
