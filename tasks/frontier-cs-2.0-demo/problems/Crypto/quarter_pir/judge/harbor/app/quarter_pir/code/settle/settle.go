// Package settle is the trusted per-call completion boundary for submission
// code. Every trusted caller of package pir records the goroutine count
// before the call and, directly after it returns and before any clock stops
// or any result is used, waits here until the count is back. Work continued
// in the background is therefore on the clock, and a goroutine that never
// ends fails the run.
package settle

import (
	"runtime"
	"time"
)

// Limit is how long a goroutine may outlive the call that started it.
const Limit = 10 * time.Second

// Wait returns true once at most baseline goroutines exist, false if more
// are still alive after Limit. Goroutines that were joined inside the call
// cost one goroutine count.
func Wait(baseline int) bool {
	if runtime.NumGoroutine() <= baseline {
		return true
	}
	deadline := time.Now().Add(Limit)
	for runtime.NumGoroutine() > baseline {
		if time.Now().After(deadline) {
			return false
		}
		runtime.Gosched()
	}
	return true
}
