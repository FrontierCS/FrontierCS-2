// truncate copies a go-ethereum chain database without the blocks above a height.
//
//	truncate SRC DST CUTOFF [WANTED_DIGEST]
//
// DST is a new LevelDB with every key of SRC but the block records of heights above CUTOFF: the
// canonical hash (h<number>n), the header (h<number><hash>), the total difficulty
// (h<number><hash>t), the body (b<number><hash>), the receipts (r<number><hash>) and the number
// of a hash (H<hash>). It prints the SHA-256 of the kept keys and values in key order, and fails
// if WANTED_DIGEST is given and differs.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"strconv"

	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/opt"
	"github.com/syndtr/goleveldb/leveldb/util"
)

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "truncate: "+format+"\n", args...)
	os.Exit(1)
}

// height is the block number a block record is about, if the key is one.
func height(key, value []byte) (uint64, bool) {
	number := func() uint64 { return binary.BigEndian.Uint64(key[1:9]) }
	switch {
	case key[0] == 'h' && len(key) == 10 && key[9] == 'n':
		return number(), true
	case key[0] == 'h' && len(key) == 41, key[0] == 'b' && len(key) == 41, key[0] == 'r' && len(key) == 41:
		return number(), true
	case key[0] == 'h' && len(key) == 42 && key[41] == 't':
		return number(), true
	case key[0] == 'H' && len(key) == 33 && len(value) == 8:
		return binary.BigEndian.Uint64(value), true
	}
	return 0, false
}

func main() {
	if len(os.Args) != 4 && len(os.Args) != 5 {
		fail("give SRC DST CUTOFF [WANTED_DIGEST]")
	}
	cutoff, err := strconv.ParseUint(os.Args[3], 10, 64)
	if err != nil {
		fail("the cutoff is not a number: %v", err)
	}
	src, err := leveldb.OpenFile(os.Args[1], &opt.Options{ReadOnly: true})
	if err != nil {
		fail("opening %s: %v", os.Args[1], err)
	}
	defer src.Close()
	dst, err := leveldb.OpenFile(os.Args[2], nil)
	if err != nil {
		fail("creating %s: %v", os.Args[2], err)
	}
	digest := sha256.New()
	kept, dropped := 0, 0
	var maxKept, maxDropped uint64
	batch := new(leveldb.Batch)
	it := src.NewIterator(nil, nil)
	for it.Next() {
		key, value := it.Key(), it.Value()
		if n, ok := height(key, value); ok && n > cutoff {
			dropped++
			if n > maxDropped {
				maxDropped = n
			}
			continue
		} else if ok && n > maxKept {
			maxKept = n
		}
		kept++
		var length [8]byte
		binary.BigEndian.PutUint32(length[:4], uint32(len(key)))
		binary.BigEndian.PutUint32(length[4:], uint32(len(value)))
		digest.Write(length[:])
		digest.Write(key)
		digest.Write(value)
		batch.Put(bytes.Clone(key), bytes.Clone(value))
		if batch.Len() >= 10000 {
			if err := dst.Write(batch, nil); err != nil {
				fail("writing: %v", err)
			}
			batch.Reset()
		}
	}
	if err := it.Error(); err != nil {
		fail("reading: %v", err)
	}
	it.Release()
	if err := dst.Write(batch, nil); err != nil {
		fail("writing: %v", err)
	}
	if err := dst.CompactRange(util.Range{}); err != nil {
		fail("compacting: %v", err)
	}
	if err := dst.Close(); err != nil {
		fail("closing: %v", err)
	}
	found := fmt.Sprintf("%x", digest.Sum(nil))
	fmt.Printf("kept %d keys (block records up to %d), dropped %d (up to %d), digest %s\n", kept, maxKept, dropped, maxDropped, found)
	if len(os.Args) == 5 && os.Args[4] != found {
		fail("the kept keys have the digest %s, not %s: pin the digest you mean", found, os.Args[4])
	}
}
