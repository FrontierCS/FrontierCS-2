// Trusted, byte-locked client driver. It has no network and never learns the
// database seed: the harness conductor streams the database, issues query
// indices one at a time and carries the online messages over this process's
// stdin and stdout. Only package pir is submission code.
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"easypir/frame"
	"easypir/pir"
	"easypir/refwire"
	"easypir/settle"
)

const frameLimit = 64 << 20

func fail(code int) {
	os.Exit(code)
}

// settled is the completion boundary of one entry-point call. It runs inside
// the clocks, directly after the entry point returns, so work continued in
// the background is never free: the clock does not stop before it ends, and a
// goroutine that never ends fails the run.
func settled(baseline int) {
	if !settle.Wait(baseline) {
		fail(4)
	}
}

// peakResidentBytes is the process's resident-set high-water mark, used only
// for a loose safety limit.
func peakResidentBytes() uint64 {
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(status), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 3 && fields[0] == "VmHWM:" {
			kilobytes, _ := strconv.ParseUint(fields[1], 10, 64)
			return kilobytes * 1024
		}
	}
	return 0
}

// liveBytes is the live Go heap after two complete collections. It is taken
// only when settled has shown that no submission goroutine exists, so no data
// can sit on another goroutine's stack; without package variables, cgo or
// unsafe, the heap is the only place the client can keep state.
func liveBytes() uint64 {
	var stats runtime.MemStats
	runtime.GC()
	runtime.GC()
	runtime.ReadMemStats(&stats)
	return stats.HeapAlloc
}

func main() {
	threads := flag.Uint64("thread", 4, "number of threads")
	size := flag.Uint64("size", 0, "database entries")
	flag.Parse()
	params := refwire.NewPlannedParams(*size, *threads)

	in := bufio.NewReaderSize(os.Stdin, 1<<20)
	out := bufio.NewWriterSize(os.Stdout, 1<<16)

	// The export buffer is allocated and touched before any clock starts.
	export := make([]byte, params.HintExportBytes())
	for i := 0; i < len(export); i += 4096 {
		export[i] = 1
	}
	scratch := make([]byte, 8*2*params.ChunkSize)
	baseline := liveBytes()
	goroutines := runtime.NumGoroutine()
	if frame.Write(out, 'R', nil) != nil {
		fail(3)
	}
	tag, payload, err := frame.Read(in, frameLimit, nil)
	if err != nil || tag != 'S' || len(payload) != 8 {
		fail(3)
	}
	coinSeed := int64(binary.LittleEndian.Uint64(payload))

	// Preprocessing clock (diagnostic copy; the conductor holds the official one).
	start := time.Now()
	client := pir.NewClient(params, coinSeed)
	settled(goroutines)
	var stream sync.Mutex
	streamOpen, streamEnded := true, false
	receive := func() []uint64 {
		tag, payload, err := frame.Read(in, frameLimit, scratch)
		if err != nil {
			fail(3)
		}
		if tag == 'D' {
			streamEnded = true
			return nil
		}
		if tag != 'C' || len(payload)%8 != 0 {
			fail(3)
		}
		chunk := make([]uint64, len(payload)/8)
		for i := range chunk {
			chunk[i] = binary.LittleEndian.Uint64(payload[8*i:])
		}
		return chunk
	}
	client.Preprocess(func() []uint64 {
		stream.Lock()
		defer stream.Unlock()
		if !streamOpen || streamEnded {
			return nil
		}
		return receive()
	})
	settled(goroutines)
	stream.Lock()
	streamOpen = false
	for !streamEnded {
		receive()
	}
	stream.Unlock()

	// The canonical hint export is copied inside the clock.
	var exportLock sync.Mutex
	exportOpen, exported, overflow := true, 0, false
	client.ExportHints(func(part []byte) {
		exportLock.Lock()
		defer exportLock.Unlock()
		if !exportOpen || overflow {
			return
		}
		if exported+len(part) > len(export) {
			overflow = true
			return
		}
		copy(export[exported:], part)
		exported += len(part)
	})
	settled(goroutines)
	exportLock.Lock()
	exportOpen = false
	exportLock.Unlock()
	offline := time.Since(start)
	if frame.Write(out, 'p', nil) != nil {
		fail(3)
	}

	digest := sha256.Sum256(export[:exported])
	report := make([]byte, 32+8+8+8+1)
	copy(report, digest[:])
	binary.LittleEndian.PutUint64(report[32:], uint64(offline.Nanoseconds()))
	live := liveBytes()
	if live < baseline {
		live = baseline
	}
	binary.LittleEndian.PutUint64(report[40:], live-baseline)
	binary.LittleEndian.PutUint64(report[48:], uint64(exported))
	if overflow {
		report[56] = 1
	}
	if frame.Write(out, 'H', report) != nil {
		fail(3)
	}

	var online time.Duration
	for {
		tag, payload, err := frame.Read(in, frameLimit, nil)
		if err != nil {
			fail(3)
		}
		if tag == 'E' {
			break
		}
		if tag == 'M' {
			// Client-storage reading between two queries, outside every clock.
			live := liveBytes()
			if live < baseline {
				live = baseline
			}
			var storage [8]byte
			binary.LittleEndian.PutUint64(storage[:], live-baseline)
			if frame.Write(out, 'm', storage[:]) != nil {
				fail(3)
			}
			continue
		}
		if tag == 'U' {
			// Primary-hint audit, outside every clock: the client is asked for
			// the canonical record of harness-chosen primary hints that no query
			// has consumed. Such a record is the one the hint export contained,
			// so it is compared with the export buffer held here.
			if len(payload)%8 != 0 {
				fail(3)
			}
			count := len(payload) / 8
			reply := make([]byte, count+32)
			records := sha256.New()
			var record [4 + 16]byte
			for i := 0; i < count; i++ {
				hint := binary.LittleEndian.Uint64(payload[8*i:])
				if hint >= params.LocalSetNum() || (hint+1)*uint64(len(record)) > uint64(exported) {
					fail(3)
				}
				hintTag, parity := client.PrimaryHint(hint)
				settled(goroutines)
				binary.LittleEndian.PutUint32(record[0:], hintTag)
				binary.LittleEndian.PutUint64(record[4:], parity[0])
				binary.LittleEndian.PutUint64(record[12:], parity[1])
				records.Write(record[:])
				if bytes.Equal(record[:], export[hint*uint64(len(record)):(hint+1)*uint64(len(record))]) {
					reply[i] = 1
				}
			}
			copy(reply[count:], records.Sum(nil))
			if frame.Write(out, 'u', reply) != nil {
				fail(3)
			}
			continue
		}
		if tag != 'Q' || len(payload) != 8 {
			fail(3)
		}
		begin := time.Now()
		request := client.Query(binary.LittleEndian.Uint64(payload))
		settled(goroutines)
		online += time.Since(begin)
		if frame.Write(out, 'q', request) != nil {
			fail(3)
		}
		tag, payload, err = frame.Read(in, frameLimit, nil)
		if err != nil || tag != 'A' {
			fail(3)
		}
		begin = time.Now()
		value := client.Reconstruct(payload)
		settled(goroutines)
		online += time.Since(begin)
		var answer [16]byte
		binary.LittleEndian.PutUint64(answer[0:], value[0])
		binary.LittleEndian.PutUint64(answer[8:], value[1])
		if frame.Write(out, 'V', answer[:]) != nil {
			fail(3)
		}
	}
	var done [16]byte
	binary.LittleEndian.PutUint64(done[:], uint64(online.Nanoseconds()))
	binary.LittleEndian.PutUint64(done[8:], peakResidentBytes())
	if frame.Write(out, 'F', done[:]) != nil {
		fail(3)
	}
	runtime.KeepAlive(export)
	runtime.KeepAlive(client)
}
