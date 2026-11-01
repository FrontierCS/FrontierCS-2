// Coverage probe: the additional untimed queries the conductor issues after
// the measured ones. They are chosen from the trusted generator's own
// bookkeeping (tags, labels, consumption counters), never from anything the
// client reports, so the candidate and the paired reference run are asked the
// identical sequence for identical seeds.
//
// Three kinds of queries are planned, in this order.
//
//   - Primary hints. A hint index is drawn uniformly from all primary hints
//     and the query is a database index whose first containing hint, under
//     the reference's selection rule at that moment, is that hint. Most deep
//     hints are the first containing hint of no index at all, so the query
//     for a drawn index h is the index whose first containing hint is the
//     nearest one at or below h among the scanned chunks.
//   - Backup hints and replacements. A few chunks are queried until their
//     whole per-chunk allotment (BackupPerGroup) is consumed, which is exactly
//     as far as the reference serves a chunk; inside a chunk the indices with
//     the deepest first containing hints are taken first.
//   - Volume. Uniformly random distinct indices, outside the chunks whose
//     allotment is used up, until the queries issued in total reach the
//     target, the planned query budget. Their hit hint is not planned.
//
// The plan is computed on a private copy of the bookkeeping. The conductor
// then issues it through the real generator and counts every query whose hit
// hint differs from the planned one (there must be none).
package main

import (
	"encoding/binary"
	"math/rand"
	"sort"
	"sync"

	"easypir/refwire"
	"easypir/safeaes"
)

const (
	// probeScanChunks bounds the chunks whose first-containing-hint table is
	// computed; probePrimaryTargets is the number of drawn primary hints;
	// probeBackupChunks is the number of chunks consumed to the full allotment.
	probeScanChunks     = 1024
	probePrimaryTargets = 256
	probeBackupChunks   = 4
	probeAuditHints     = 512

	probeNone   = ^uint32(0)      // no hint contains this index
	probeIssued = ^uint32(0) - 1  // already queried, never repeated
	probeStale  = uint32(1) << 31 // first containing hint no longer known
)

type probeQuery struct {
	X      uint64
	Hit    uint64 // primary hint the reference uses for X at that moment
	Backup bool   // part of a chunk's allotment run
	Volume bool   // part of the volume phase; Hit is not planned
}

// prober is the private copy of the generator's bookkeeping plus the tables
// of the scanned chunks.
type prober struct {
	p        refwire.Params
	master   safeaes.RoundKeys
	primary  uint64 // LocalSetNum
	perGroup uint64 // BackupPerGroup
	tags     []uint32
	labels   []uint64
	consumed []uint64
	issued   map[uint64]bool
	chunks   []uint64   // scanned chunk ids
	first    [][]uint32 // per scanned chunk: first containing hint of each offset
	live     []uint32   // per hint: scanned indices whose first containing hint it is
}

// offset is refwire.SetOffset without allocation.
func (s *prober) offset(tag uint32, chunk uint64) uint64 {
	var block, subkey, out [16]byte
	var keys safeaes.RoundKeys
	binary.LittleEndian.PutUint64(block[:], uint64(tag)<<35+chunk>>s.p.SBSizeLog2)
	safeaes.MMO(&s.master, &subkey, &block)
	safeaes.ExpandKey(&subkey, &keys)
	block = [16]byte{}
	binary.LittleEndian.PutUint64(block[:], chunk&(s.p.SBSize-1))
	safeaes.MMO(&keys, &out, &block)
	return binary.LittleEndian.Uint64(out[:]) & (s.p.ChunkSize - 1)
}

// skipped is the reference's rule: a hint whose promoted point lies in the
// queried chunk is not used for that chunk.
func (s *prober) skipped(hint, chunk uint64) bool {
	return s.labels[hint] != refwire.NoLabel && s.labels[hint]/s.p.ChunkSize == chunk
}

// table computes the first containing hint of every offset of one chunk.
func (s *prober) table(chunk uint64) []uint32 {
	first := make([]uint32, s.p.ChunkSize)
	for i := range first {
		first[i] = probeNone
	}
	missing := s.p.ChunkSize
	for hint := uint64(0); hint < s.primary && missing > 0; hint++ {
		if s.skipped(hint, chunk) {
			continue
		}
		if o := s.offset(s.tags[hint], chunk); first[o] == probeNone {
			first[o] = uint32(hint)
			missing--
		}
	}
	for o := range first {
		if s.issued[chunk*s.p.ChunkSize+uint64(o)] {
			first[o] = probeIssued
		}
	}
	return first
}

func newProber(g *refwire.Generator, issued map[uint64]bool, rng *rand.Rand, workers int) *prober {
	s := &prober{p: g.P, master: safeaes.RoundKeys(*g.LongKey), primary: g.LocalSetNum, perGroup: g.BackupPerGroup,
		tags: append([]uint32(nil), g.Tags...), labels: append([]uint64(nil), g.Labels...),
		consumed: append([]uint64(nil), g.BackupConsumed...), issued: map[uint64]bool{}}
	for x := range issued {
		s.issued[x] = true
	}
	scan := s.p.SetSize
	if scan > probeScanChunks {
		scan = probeScanChunks
	}
	for _, chunk := range rng.Perm(int(s.p.SetSize))[:scan] {
		s.chunks = append(s.chunks, uint64(chunk))
	}
	s.first = make([][]uint32, scan)
	var wait sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wait.Add(1)
		go func(worker int) {
			defer wait.Done()
			for index := worker; index < len(s.chunks); index += workers {
				s.first[index] = s.table(s.chunks[index])
			}
		}(worker)
	}
	wait.Wait()
	s.live = make([]uint32, s.primary)
	for _, first := range s.first {
		for _, hint := range first {
			if uint64(hint) < s.primary {
				s.live[hint]++
			}
		}
	}
	return s
}

// consume applies one query to the private bookkeeping exactly as
// refwire.Generator.Query does and keeps the scanned tables exact: an entry
// whose first containing hint was the consumed one becomes stale (its next
// containing hint is not known), and the promoted backup hint becomes the
// first containing hint wherever it now precedes the recorded one.
func (s *prober) consume(x, hint uint64) {
	chunk := x / s.p.ChunkSize
	oldTag, newTag := s.tags[hint], uint32(s.primary+chunk*s.perGroup+s.consumed[chunk])
	for index, other := range s.chunks {
		first := s.first[index]
		if other == chunk {
			o := x % s.p.ChunkSize
			if uint64(first[o]) < s.primary {
				s.live[first[o]]--
			}
			first[o] = probeIssued
			continue
		}
		if !s.skipped(hint, other) {
			if o := s.offset(oldTag, other); uint64(first[o]) == hint {
				first[o] = uint32(hint) | probeStale
				s.live[hint]--
			}
		}
		o := s.offset(newTag, other)
		switch current := first[o]; {
		case current == probeIssued:
		case current == probeNone, uint64(current) < s.primary && uint64(current) > hint,
			current&probeStale != 0 && uint64(current&^probeStale) >= hint:
			if uint64(current) < s.primary {
				s.live[current]--
			}
			first[o] = uint32(hint)
			s.live[hint]++
		}
	}
	s.tags[hint] = newTag
	s.labels[hint] = x
	s.consumed[chunk]++
	s.issued[x] = true
}

// at returns a scanned, not yet issued index whose first containing hint is
// hint, in a chunk that still has a backup hint left.
func (s *prober) at(hint uint64, start int) (uint64, bool) {
	for step := range s.chunks {
		index := (start + step) % len(s.chunks)
		chunk := s.chunks[index]
		if s.consumed[chunk] >= s.perGroup || s.skipped(hint, chunk) {
			continue
		}
		if o := s.offset(s.tags[hint], chunk); uint64(s.first[index][o]) == hint {
			return chunk*s.p.ChunkSize + o, true
		}
	}
	return 0, false
}

// planAudit returns the primary hints of the audit after the online phase:
// probeAuditHints distinct hints, uniform over the primary hints that no
// query has consumed (their tag is still their index, so their canonical
// record is the one the hint export contained).
func planAudit(g *refwire.Generator, seed int64) []uint64 {
	rng := rand.New(rand.NewSource(seed ^ 0x5DEECE66D))
	chosen := map[uint64]bool{}
	var hints []uint64
	for attempts := 0; len(hints) < probeAuditHints && attempts < 64*probeAuditHints; attempts++ {
		hint := rng.Uint64() % g.LocalSetNum
		if chosen[hint] || g.Labels[hint] != refwire.NoLabel || uint64(g.Tags[hint]) != hint {
			continue
		}
		chosen[hint] = true
		hints = append(hints, hint)
	}
	return hints
}

// planProbe returns the probe queries for the generator's current state.
// It does not modify the generator or the issued set.
func planProbe(g *refwire.Generator, issued map[uint64]bool, seed int64, workers int, target uint64) []probeQuery {
	rng := rand.New(rand.NewSource(seed))
	s := newProber(g, issued, rng, workers)
	var plan []probeQuery

	for target := 0; target < probePrimaryTargets; target++ {
		drawn, start := rng.Uint64()%s.primary, rng.Intn(len(s.chunks))
		for hint := int64(drawn); hint >= 0; hint-- {
			if s.live[hint] == 0 {
				continue
			}
			if x, ok := s.at(uint64(hint), start); ok {
				s.consume(x, uint64(hint))
				plan = append(plan, probeQuery{X: x, Hit: uint64(hint)})
				break
			}
		}
	}

	runs := 0
	for _, chunk := range s.chunks {
		if runs == probeBackupChunks {
			break
		}
		if s.consumed[chunk] >= s.perGroup {
			continue
		}
		runs++
		// A fresh table: no stale entries, and exact for the whole run, because
		// a hint consumed inside this chunk is afterwards skipped for it.
		first := s.table(chunk)
		offsets := make([]int, 0, len(first))
		for o, hint := range first {
			if uint64(hint) < s.primary {
				offsets = append(offsets, o)
			}
		}
		sort.Slice(offsets, func(a, b int) bool { return first[offsets[a]] > first[offsets[b]] })
		for _, o := range offsets {
			if s.consumed[chunk] >= s.perGroup {
				break
			}
			x, hint := chunk*s.p.ChunkSize+uint64(o), uint64(first[o])
			s.consume(x, hint)
			plan = append(plan, probeQuery{X: x, Hit: hint, Backup: true})
		}
	}

	for total := g.Count + uint64(len(plan)); total < target; total++ {
		x := rng.Uint64() % s.p.DBSize
		for s.issued[x] || s.consumed[x/s.p.ChunkSize] >= s.perGroup {
			x = rng.Uint64() % s.p.DBSize
		}
		s.issued[x] = true
		s.consumed[x/s.p.ChunkSize]++
		plan = append(plan, probeQuery{X: x, Volume: true})
	}
	return plan
}
