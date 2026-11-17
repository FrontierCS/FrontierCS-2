// Package refwire is the trusted, byte-locked definition of the reference
// client's server view: the request generator, which never touches database
// contents, and the information-tight encodings of the reference messages.
package refwire

import (
	"encoding/binary"
	"math"
	"math/bits"
	"math/rand"

	"example.com/util"
)

const (
	FailureProbLog2 = 40
	NoLabel         = uint64(999999999)
)

// historyReserve is the initial capacity of the generator's query history.
// The history holds the queries issued so far and grows with them, up to the
// planned budget; it is not preallocated for the whole budget.
const historyReserve = 1000

// Params are the public parameters of one experiment. OnlineQueries is the
// planned query budget PlannedQueries(): the most queries one preprocessing
// has to serve. It is not the number of queries a particular run issues.
type Params struct {
	DBSize        uint64
	ChunkSize     uint64
	SetSize       uint64
	SBSize        uint64
	SBSizeLog2    uint64
	SBNum         uint64
	Threads       uint64
	OnlineQueries uint64
}

func NewParams(dbSize, threads, onlineQueries uint64) Params {
	chunkSize, setSize, sbSize, sbNum := util.GenParams(dbSize)
	return Params{
		DBSize: dbSize, ChunkSize: chunkSize, SetSize: setSize, SBSize: sbSize,
		SBSizeLog2: uint64(math.Log2(float64(sbSize))), SBNum: sbNum,
		Threads: threads, OnlineQueries: onlineQueries,
	}
}

// NewPlannedParams are the parameters every harness process uses: the query
// count is the planned budget.
func NewPlannedParams(dbSize, threads uint64) Params {
	p := NewParams(dbSize, threads, 0)
	p.OnlineQueries = p.PlannedQueries()
	return p
}

// PlannedQueries is the original planned query budget of one preprocessing.
func (p Params) PlannedQueries() uint64 {
	return uint64(math.Sqrt(float64(p.DBSize)) * math.Log(float64(p.DBSize)))
}

// LocalSetNum is the number of primary hints.
func (p Params) LocalSetNum() uint64 {
	k := math.Ceil(math.Log(2)*(FailureProbLog2+1) + math.Log(float64(p.PlannedQueries())))
	localSetNum := uint64(k) * p.ChunkSize
	return (localSetNum + p.Threads - 1) / p.Threads * p.Threads
}

// BackupPerGroup is the number of backup hints (and replacements) per chunk.
func (p Params) BackupPerGroup() uint64 {
	backup := 3 * uint64(float64(p.PlannedQueries())/float64(p.SetSize))
	return (backup + p.Threads - 1) / p.Threads * p.Threads
}

func (p Params) TotalBackup() uint64 { return p.BackupPerGroup() * p.SetSize }

// OffsetBits is the exact width of one chunk offset.
func (p Params) OffsetBits() uint { return uint(bits.Len64(p.ChunkSize - 1)) }

// RequestBytes and ResponseBytes are the lengths of the reference messages.
func (p Params) RequestBytes() int {
	return int(p.SBNum)*util.KeyByteLength + (int(p.SBSize)*int(p.OffsetBits())+7)/8
}

func (p Params) ResponseBytes() int { return 2 * int(p.SBNum) * util.DBEntrySize }

// HintExportBytes is the length of the canonical hint export after preprocessing.
func (p Params) HintExportBytes() int {
	return int(p.LocalSetNum())*(4+util.DBEntrySize) + int(p.TotalBackup())*(4+3*util.DBEntrySize+8+util.DBEntrySize)
}

// EncodeRequest packs SBNum 16-byte keys followed by SBSize offsets of
// OffsetBits bits each, least-significant bit first.
func EncodeRequest(p Params, keys []byte, offsets []uint32) []byte {
	if len(keys) != int(p.SBNum)*util.KeyByteLength || len(offsets) != int(p.SBSize) {
		panic("refwire: malformed reference request")
	}
	out := make([]byte, p.RequestBytes())
	copy(out, keys)
	width := p.OffsetBits()
	position := uint(len(keys)) * 8
	for _, offset := range offsets {
		if uint64(offset) >= p.ChunkSize {
			panic("refwire: offset out of range")
		}
		for bit := uint(0); bit < width; bit++ {
			if offset>>bit&1 == 1 {
				out[position>>3] |= 1 << (position & 7)
			}
			position++
		}
	}
	return out
}

// DecodeRequest is the inverse of EncodeRequest.
func DecodeRequest(p Params, request []byte) ([]util.PrfKey, []uint32, bool) {
	if len(request) != p.RequestBytes() {
		return nil, nil, false
	}
	keys := make([]util.PrfKey, p.SBNum)
	for i := range keys {
		copy(keys[i][:], request[i*util.KeyByteLength:(i+1)*util.KeyByteLength])
	}
	offsets := make([]uint32, p.SBSize)
	width := p.OffsetBits()
	position := uint(p.SBNum) * util.KeyByteLength * 8
	for i := range offsets {
		for bit := uint(0); bit < width; bit++ {
			offsets[i] |= uint32(request[position>>3]>>(position&7)&1) << bit
			position++
		}
	}
	return keys, offsets, true
}

// EncodeResponse concatenates the two parity arrays as little-endian words.
func EncodeResponse(p Params, listParities, keyParities []uint64) []byte {
	words := int(p.SBNum) * util.DBEntryLength
	if len(listParities) != words || len(keyParities) != words {
		panic("refwire: malformed reference response")
	}
	out := make([]byte, p.ResponseBytes())
	for i, word := range listParities {
		binary.LittleEndian.PutUint64(out[8*i:], word)
	}
	for i, word := range keyParities {
		binary.LittleEndian.PutUint64(out[8*(words+i):], word)
	}
	return out
}

// DecodeResponse is the inverse of EncodeResponse.
func DecodeResponse(p Params, response []byte) ([]uint64, []uint64, bool) {
	if len(response) != p.ResponseBytes() {
		return nil, nil, false
	}
	words := int(p.SBNum) * util.DBEntryLength
	listParities := make([]uint64, words)
	keyParities := make([]uint64, words)
	for i := range listParities {
		listParities[i] = binary.LittleEndian.Uint64(response[8*i:])
		keyParities[i] = binary.LittleEndian.Uint64(response[8*(words+i):])
	}
	return listParities, keyParities, true
}

// LongKey is an expanded PRF key: the 11 AES round keys. The reference's PRF
// helpers take it as a pointer to this fixed-size array, never as a slice, so
// the assembly below them always sees a complete key.
type LongKey [44]uint32

// NewLongKey expands a 16-byte PRF key (util.GetLongKey).
func NewLongKey(key *[16]byte) *LongKey {
	return (*LongKey)(util.GetLongKey((*util.PrfKey128)(key)))
}

// Eval is the reference PRF on an expanded key (util.PRFEvalWithLongKey).
func Eval(longKey *LongKey, x uint64) uint64 {
	return util.PRFEvalWithLongKey(longKey[:], x)
}

// SubKey derives the 16-byte key of one tag and superblock from the master
// key (util.PRFEvalWithLongKeyAndTag128Output).
func SubKey(longKey *LongKey, tag uint32, x uint64) [16]byte {
	return util.PRFEvalWithLongKeyAndTag128Output(longKey[:], tag, x)
}

// SetOffset returns the offset of the chunkId-th chunk in the set with this tag.
func SetOffset(longKey *LongKey, tag uint32, chunkId uint64, p Params) uint64 {
	subkey := SubKey(longKey, tag, chunkId>>p.SBSizeLog2)
	return util.PRFEval((*util.PrfKey)(&subkey), chunkId&(p.SBSize-1)) & (p.ChunkSize - 1)
}

// LastQuery describes the hint bookkeeping of the most recent Query call.
type LastQuery struct {
	X          uint64
	HitSetId   uint64
	CaseId     int
	Y          uint64 // previous positive label of the hit set, or NoLabel
	ChunkId    uint64
	SBofX      uint64
	ReplSlot   uint64 // consumed replacement slot in ChunkId's group
	ReplUsed   bool   // false when the group had no replacement left
	BackupSlot uint64 // backup hint of ChunkId's group promoted by this query
}

// Generator reproduces every byte the reference client sends. It holds the
// coins, tags, labels, history and consumption counters; no database content.
type Generator struct {
	P              Params
	LongKey        *LongKey
	LocalSetNum    uint64
	BackupPerGroup uint64
	Tags           []uint32 // current tag of each primary hint
	Labels         []uint64 // positive constraint of each primary hint
	HistoryHit     []uint64 // hit hint of every query issued so far (length Count)
	HistoryQueries []uint64 // index of every query issued so far (length Count)
	Count          uint64
	BackupConsumed []uint64
	ReplConsumed   []uint64
	ReplIndices    []uint64 // SetSize*BackupPerGroup database indices
	Last           LastQuery
	rng            *rand.Rand
}

func NewGenerator(p Params, coinSeed int64) *Generator {
	rng := rand.New(rand.NewSource(coinSeed))
	masterKey := util.RandKey(rng)
	g := &Generator{
		P: p, rng: rng,
		LongKey:        NewLongKey((*[16]byte)(&masterKey)),
		LocalSetNum:    p.LocalSetNum(),
		BackupPerGroup: p.BackupPerGroup(),
	}
	g.Tags = make([]uint32, g.LocalSetNum)
	g.Labels = make([]uint64, g.LocalSetNum)
	for j := range g.Tags {
		g.Tags[j] = uint32(j)
		g.Labels[j] = NoLabel
	}
	g.HistoryHit = make([]uint64, 0, historyReserve)
	g.HistoryQueries = make([]uint64, 0, historyReserve)
	g.BackupConsumed = make([]uint64, p.SetSize)
	g.ReplConsumed = make([]uint64, p.SetSize)
	g.ReplIndices = make([]uint64, p.SetSize*g.BackupPerGroup)
	return g
}

// BackupTag is the tag of the k-th backup hint of a chunk's group.
func (g *Generator) BackupTag(chunkId, k uint64) uint32 {
	return uint32(g.LocalSetNum + chunkId*g.BackupPerGroup + k)
}

// DrawReplacements consumes the coins of one chunk's replacement indices.
// Chunks must be drawn in increasing order, each exactly once, before Query.
func (g *Generator) DrawReplacements(chunkId uint64) {
	for k := uint64(0); k < g.BackupPerGroup; k++ {
		offset := g.rng.Uint64() & (g.P.ChunkSize - 1)
		g.ReplIndices[chunkId*g.BackupPerGroup+k] = offset + chunkId*g.P.ChunkSize
	}
}

// Query returns the reference request for x and advances the bookkeeping.
func (g *Generator) Query(x uint64) ([]byte, []uint32) {
	p := g.P
	q := g.Count

	hitSetId := NoLabel
	queryOffset := x % p.ChunkSize
	chunkId := x / p.ChunkSize
	for i := uint64(0); i < g.LocalSetNum; i++ {
		if SetOffset(g.LongKey, g.Tags[i], chunkId, p) == queryOffset {
			if g.Labels[i] != NoLabel && chunkId == (g.Labels[i]/p.ChunkSize) {
				continue
			}
			hitSetId = i
			break
		}
	}
	if hitSetId == NoLabel {
		panic("refwire: no hit set found")
	}
	g.HistoryQueries = append(g.HistoryQueries, x)
	g.HistoryHit = append(g.HistoryHit, hitSetId)
	y := g.Labels[hitSetId]

	caseId := 0
	if y != NoLabel {
		if y/(p.SBSize*p.ChunkSize) == x/(p.SBSize*p.ChunkSize) {
			caseId = 1
		} else {
			caseId = 2
		}
	}

	subkeyBytes := make([]byte, p.SBNum*util.KeyByteLength)
	offsets := make([]uint64, p.SBSize)
	SBofX := x / (p.SBSize * p.ChunkSize)
	for i := uint64(0); i < p.SBNum; i++ {
		subkey := SubKey(g.LongKey, g.Tags[hitSetId], i)
		copy(subkeyBytes[i*util.KeyByteLength:(i+1)*util.KeyByteLength], subkey[:])
	}
	for j := uint64(0); j < p.SBSize; j++ {
		offsets[j] = SetOffset(g.LongKey, g.Tags[hitSetId], SBofX*p.SBSize+j, p)
	}

	nxtAvailable := g.ReplConsumed[chunkId]
	repOffset := g.rng.Uint64() % p.ChunkSize // a dummy random offset
	replUsed := false
	if nxtAvailable != g.BackupPerGroup {
		repOffset = g.ReplIndices[chunkId*g.BackupPerGroup+nxtAvailable] % p.ChunkSize
		g.ReplConsumed[chunkId]++
		replUsed = true
	}
	offsets[chunkId%p.SBSize] = repOffset
	randKey := util.RandKey(g.rng)
	copy(subkeyBytes[SBofX*util.KeyByteLength:(SBofX+1)*util.KeyByteLength], randKey[:])

	if caseId == 1 {
		offsets[(y/p.ChunkSize)%p.SBSize] = y % p.ChunkSize
	}

	if caseId == 2 {
		negConst := make([]uint64, 0)
		for i := int(q) - 1; i >= 0; i-- {
			if g.HistoryHit[i] == hitSetId {
				break
			}
			if g.HistoryHit[i] > hitSetId {
				if g.HistoryQueries[i]/(p.SBSize*p.ChunkSize) == y/(p.SBSize*p.ChunkSize) {
					negConst = append(negConst, g.HistoryQueries[i])
				}
			}
		}
		randKey := util.RandKey(g.rng)
		yOffset := y % p.ChunkSize
		yChunkId := y / p.ChunkSize
		yRelativeChunkId := yChunkId % p.SBSize
		for true {
			newOffset := util.PRFEval(&randKey, yRelativeChunkId) & (p.ChunkSize - 1)
			if newOffset != yOffset {
				randKey = util.RandKey(g.rng)
				continue
			}
			ok := true
			for i := 0; i < len(negConst); i++ {
				tmpOffset := util.PRFEval(&randKey, (negConst[i]/p.ChunkSize)%p.SBSize) % p.ChunkSize
				if tmpOffset == negConst[i]%p.ChunkSize {
					ok = false
					break
				}
			}
			if ok == true {
				break
			} else {
				randKey = util.RandKey(g.rng)
			}
		}
		copy(subkeyBytes[(yChunkId/p.SBSize)*util.KeyByteLength:(yChunkId/p.SBSize+1)*util.KeyByteLength], randKey[:])
	}
	offsetsuint32 := make([]uint32, len(offsets))
	for i := 0; i < len(offsets); i++ {
		offsetsuint32[i] = uint32(offsets[i])
	}

	backupSlot := g.BackupConsumed[chunkId]
	if backupSlot == g.BackupPerGroup {
		panic("refwire: no backup set available")
	}
	g.Last = LastQuery{X: x, HitSetId: hitSetId, CaseId: caseId, Y: y, ChunkId: chunkId,
		SBofX: SBofX, ReplSlot: nxtAvailable, ReplUsed: replUsed, BackupSlot: backupSlot}
	g.Tags[hitSetId] = g.BackupTag(chunkId, backupSlot)
	g.BackupConsumed[chunkId]++
	g.Labels[hitSetId] = x
	g.Count++
	return subkeyBytes, offsetsuint32
}
