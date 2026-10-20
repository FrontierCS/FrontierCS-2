package main

import (
	"math/rand"
	"reflect"
	"testing"

	"easypir/refwire"
)

// measured reproduces the conductor's state after the measured queries.
func measured(size uint64, coinSeed, querySeed int64, queries int) (*refwire.Generator, map[uint64]bool) {
	params := refwire.NewPlannedParams(size, 4)
	generator := refwire.NewGenerator(params, coinSeed)
	for chunk := uint64(0); chunk < params.SetSize; chunk++ {
		generator.DrawReplacements(chunk)
	}
	rng := rand.New(rand.NewSource(querySeed))
	issued := map[uint64]bool{}
	for q := 0; q < queries; q++ {
		x := rng.Uint64() % size
		for issued[x] {
			x = rng.Uint64() % size
		}
		issued[x] = true
		generator.Query(x)
	}
	return generator, issued
}

// firstHit is the reference's selection rule, written against refwire only.
func firstHit(g *refwire.Generator, x uint64) uint64 {
	chunk, offset := x/g.P.ChunkSize, x%g.P.ChunkSize
	for hint := uint64(0); hint < g.LocalSetNum; hint++ {
		if refwire.SetOffset(g.LongKey, g.Tags[hint], chunk, g.P) != offset {
			continue
		}
		if g.Labels[hint] != refwire.NoLabel && g.Labels[hint]/g.P.ChunkSize == chunk {
			continue
		}
		return hint
	}
	return refwire.NoLabel
}

func TestProbeOffsetMatchesTheReferencePRF(t *testing.T) {
	for _, size := range []uint64{65536, 67108864} {
		generator, _ := measured(size, 7, 9, 0)
		s := newProber(generator, nil, rand.New(rand.NewSource(1)), 1)
		rng := rand.New(rand.NewSource(int64(size)))
		for trial := 0; trial < 2000; trial++ {
			tag := uint32(rng.Uint64() % (generator.LocalSetNum + generator.P.TotalBackup()))
			chunk := rng.Uint64() % generator.P.SetSize
			if got, want := s.offset(tag, chunk), refwire.SetOffset(generator.LongKey, tag, chunk, generator.P); got != want {
				t.Fatalf("size %d tag %d chunk %d: offset %d, reference %d", size, tag, chunk, got, want)
			}
		}
		if size > 65536 {
			break // one table of the large size is enough; its scan is the expensive part
		}
	}
}

func checkPlan(t *testing.T, size uint64, coinSeed, querySeed, probeSeed int64) {
	generator, issued := measured(size, coinSeed, querySeed, 1000)
	params := generator.P
	before := generator.Count
	plan := planProbe(generator, issued, probeSeed, 4)
	if generator.Count != before || len(issued) != 1000 {
		t.Fatal("planning modified the generator or the issued set")
	}
	if again := planProbe(generator, issued, probeSeed, 2); !reflect.DeepEqual(plan, again) {
		t.Fatal("the plan is not a deterministic function of the state and the seed")
	}
	if other := planProbe(generator, issued, probeSeed+1, 4); reflect.DeepEqual(plan, other) {
		t.Fatal("the plan does not depend on the probe seed")
	}
	if uint64(1000+len(plan)) > params.PlannedQueries() {
		t.Fatalf("%d queries exceed the planned budget %d", 1000+len(plan), params.PlannedQueries())
	}

	primary, deepest := 0, uint64(0)
	runs := map[uint64]bool{}
	for index, query := range plan {
		if issued[query.X] {
			t.Fatalf("probe query %d repeats index %d", index, query.X)
		}
		issued[query.X] = true
		chunk := query.X / params.ChunkSize
		if generator.BackupConsumed[chunk] >= generator.BackupPerGroup {
			t.Fatalf("probe query %d exceeds the allotment of chunk %d", index, chunk)
		}
		if want := firstHit(generator, query.X); want != query.Hit {
			t.Fatalf("probe query %d: planned hint %d, reference rule gives %d", index, query.Hit, want)
		}
		generator.Query(query.X) // panics if the reference could not serve it
		if generator.Last.HitSetId != query.Hit || !generator.Last.ReplUsed {
			t.Fatalf("probe query %d: generator used hint %d (replacement %v), planned %d",
				index, generator.Last.HitSetId, generator.Last.ReplUsed, query.Hit)
		}
		if query.Backup {
			runs[chunk] = true
		} else {
			primary++
			if len(runs) != 0 {
				t.Fatal("primary probe queries must precede the allotment runs")
			}
		}
		if query.Hit > deepest {
			deepest = query.Hit
		}
	}
	if primary != probePrimaryTargets {
		t.Fatalf("%d primary probe queries, want %d", primary, probePrimaryTargets)
	}
	if len(runs) != probeBackupChunks {
		t.Fatalf("%d allotment runs, want %d", len(runs), probeBackupChunks)
	}
	for chunk := range runs {
		// Exactly the whole allotment: the last backup hint and the last
		// replacement of the chunk are used, and nothing beyond them is asked.
		if generator.BackupConsumed[chunk] != generator.BackupPerGroup || generator.ReplConsumed[chunk] != generator.BackupPerGroup {
			t.Fatalf("chunk %d consumed %d backups and %d replacements of %d", chunk,
				generator.BackupConsumed[chunk], generator.ReplConsumed[chunk], generator.BackupPerGroup)
		}
	}
	// 1,000 uniform queries reach about ChunkSize*ln(1000) = 6.9 ChunkSize deep.
	if deepest < 9*params.ChunkSize {
		t.Fatalf("deepest probed hint %d is below 9 ChunkSize", deepest)
	}
	if uint64(len(generator.HistoryHit)) != generator.Count || generator.Count != uint64(1000+len(plan)) {
		t.Fatal("generator history does not cover every issued query")
	}
}

func TestProbePlanIsExactAndWithinTheReference(t *testing.T) {
	for seed := int64(1); seed <= 3; seed++ {
		checkPlan(t, 65536, 100+seed, 200+seed, 300+seed)
	}
}

// A larger size where only part of the chunks is scanned.
func TestProbePlanWithAScanLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	checkPlan(t, 16777216, 11, 12, 13)
}

// The generator's history is not preallocated for the planned budget and
// its reservation does not depend on Params.OnlineQueries.
func TestGeneratorHistoryReservation(t *testing.T) {
	params := refwire.NewPlannedParams(67108864, 4)
	if params.OnlineQueries != params.PlannedQueries() || params.OnlineQueries != 147634 {
		t.Fatalf("OnlineQueries = %d, want the planned budget", params.OnlineQueries)
	}
	generator := refwire.NewGenerator(params, 1)
	if len(generator.HistoryHit) != 0 || cap(generator.HistoryHit) != 1000 || cap(generator.HistoryQueries) != 1000 {
		t.Fatal("unexpected history reservation")
	}
}

// The audit sample is deterministic, distinct and contains only primary
// hints that no query has consumed.
func TestAuditSampleIsUnconsumedPrimaryHints(t *testing.T) {
	generator, issued := measured(65536, 5, 6, 1000)
	for _, query := range planProbe(generator, issued, 7, 4) {
		generator.Query(query.X)
	}
	hints := planAudit(generator, 7)
	if len(hints) != probeAuditHints || !reflect.DeepEqual(hints, planAudit(generator, 7)) ||
		reflect.DeepEqual(hints, planAudit(generator, 8)) {
		t.Fatal("audit sample has the wrong size or is not a function of the seed")
	}
	seen, tail := map[uint64]bool{}, 0
	for _, hint := range hints {
		if seen[hint] || hint >= generator.LocalSetNum || generator.Labels[hint] != refwire.NoLabel || uint64(generator.Tags[hint]) != hint {
			t.Fatalf("hint %d is repeated, out of range or consumed", hint)
		}
		seen[hint] = true
		if hint >= 20*generator.P.ChunkSize {
			tail++
		}
	}
	if tail < probeAuditHints/4 {
		t.Fatalf("only %d sampled hints lie beyond 20 ChunkSize", tail)
	}
}
