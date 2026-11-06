// Trusted, byte-locked request-reduction process. It receives nothing but
// reference requests, so pir.ReduceRequest cannot depend on the query index,
// the client state, the coins or the database.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"os"
	"runtime"

	"easypir/frame"
	"easypir/pir"
	"easypir/refwire"
	"easypir/settle"
)

// reduce is one ReduceRequest call with its completion boundary: the result
// is taken only after every goroutine the call started has exited. If one is
// still alive after settle.Limit the reducer exits without answering, which
// the conductor records as a failed reduction.
func reduce(params refwire.Params, reference []byte) ([]byte, bool) {
	goroutines := runtime.NumGoroutine()
	reduced := pir.ReduceRequest(params, append([]byte(nil), reference...))
	if !settle.Wait(goroutines) {
		return nil, false
	}
	return append([]byte(nil), reduced...), true
}

func main() {
	threads := flag.Uint64("thread", 4, "number of threads")
	size := flag.Uint64("size", 0, "database entries")
	flag.Parse()
	params := refwire.NewPlannedParams(*size, *threads)
	in := bufio.NewReaderSize(os.Stdin, 1<<16)
	out := bufio.NewWriterSize(os.Stdout, 1<<16)
	for {
		tag, reference, err := frame.Read(in, 1<<20, nil)
		if err != nil || tag != 'r' {
			return
		}
		first, ok := reduce(params, reference)
		if !ok {
			return
		}
		second, ok := reduce(params, reference)
		if !ok {
			return
		}
		if !bytes.Equal(first, second) {
			frame.Write(out, 'x', nil)
			return
		}
		if frame.Write(out, 'c', first) != nil {
			return
		}
	}
}
