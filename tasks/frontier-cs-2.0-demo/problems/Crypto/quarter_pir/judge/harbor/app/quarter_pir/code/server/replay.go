// Trusted server-time replay, part of the byte-locked server entrypoint. With
// FRONTIER_REPLAY=1 the binary runs
//
//	server <db size> <db seed> <transcript.json> <output.json> validate|measure
//
// instead of serving, and calls the trusted server clock in process.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strconv"

	"easypir/refwire"

	util "example.com/util"
)

type frontierRecord struct {
	Request    []byte `json:"request"`
	Response   []byte `json:"response"`
	RefRequest []byte `json:"ref_request"`
}

func frontierOracle(server *QueryServiceServer, keys []util.PrfKey, offsets []uint32) ([]uint64, []uint64) {
	candidates := make([]uint64, SBNum*util.DBEntryLength)
	auxiliary := make([]uint64, SBNum*util.DBEntryLength)
	for candidate := uint64(0); candidate < SBNum; candidate++ {
		for block := uint64(0); block < SBNum; block++ {
			longKey := util.GetLongKey((*util.PrfKey128)(&keys[block]))
			for chunk := uint64(0); chunk < SBSize; chunk++ {
				offset := uint64(offsets[chunk])
				if block != candidate { offset = util.PRFEvalWithLongKey(longKey, chunk) & (ChunkSize-1) }
				entry := server.DBAccess((block*SBSize+chunk)*ChunkSize+offset)
				for word := 0; word < util.DBEntryLength; word++ { candidates[candidate*util.DBEntryLength+uint64(word)] ^= entry[word] }
			}
		}
	}
	for block := uint64(0); block < SBNum; block++ {
		longKey := util.GetLongKey((*util.PrfKey128)(&keys[block]))
		for chunk := uint64(0); chunk < SBSize; chunk++ {
			offset := util.PRFEvalWithLongKey(longKey, chunk) & (ChunkSize-1)
			entry := server.DBAccess((block*SBSize+chunk)*ChunkSize+offset)
			for word := 0; word < util.DBEntryLength; word++ { auxiliary[block*util.DBEntryLength+uint64(word)] ^= entry[word] }
		}
	}
	return candidates, auxiliary
}

func frontierAbort(reason string) {
	fmt.Fprintln(os.Stderr, "protocol validity failed:", reason)
	os.Exit(3)
}

// frontierGuard perturbs database entries that the reference request
// addresses through its explicit offsets. A response that does not change, or
// does not change back, was not computed from the resident database.
func frontierGuard(server *QueryServiceServer, records []frontierRecord, digest [32]byte) {
	for guard := 0; guard < 8; guard++ {
		digest = sha256.Sum256(digest[:])
		record := records[binary.LittleEndian.Uint64(digest[:8])%uint64(len(records))]
		_, offsets, ok := refwire.DecodeRequest(Params, record.RefRequest)
		if !ok { frontierAbort("reference request") }
		block := binary.LittleEndian.Uint64(digest[8:16])%SBNum
		chunk := binary.LittleEndian.Uint64(digest[16:24])%SBSize
		entryIndex := ((block*SBSize+chunk)*ChunkSize+uint64(offsets[chunk]))*util.DBEntryLength
		mask := binary.LittleEndian.Uint64(digest[24:32])|1
		server.DB[entryIndex] ^= mask
		changed, _ := server.TimedQuery(append([]byte(nil), record.Request...))
		server.DB[entryIndex] ^= mask
		restored, _ := server.TimedQuery(append([]byte(nil), record.Request...))
		if bytes.Equal(changed, record.Response) || !bytes.Equal(restored, record.Response) { frontierAbort("response does not follow the resident database") }
	}
}

func init() {
	if os.Getenv("FRONTIER_REPLAY") != "1" { return }
	if len(os.Args) != 6 { frontierAbort("arguments") }
	var err error
	DBSize, err = strconv.ParseUint(os.Args[1],10,64); if err != nil { frontierAbort("arguments") }
	DBSeed, err = strconv.ParseUint(os.Args[2],10,64); if err != nil { frontierAbort("arguments") }
	Params = refwire.NewPlannedParams(DBSize, 4)
	ChunkSize, SetSize, SBSize, SBNum = Params.ChunkSize, Params.SetSize, Params.SBSize, Params.SBNum
	server := &QueryServiceServer{DB: make([]uint64, ChunkSize*SetSize*util.DBEntryLength)}
	for index := uint64(0); index < ChunkSize*SetSize; index++ {
		entry := util.GenDBEntry(DBSeed,index)
		copy(server.DB[index*util.DBEntryLength:(index+1)*util.DBEntryLength],entry[:])
	}
	file, err := os.Open(os.Args[3]); if err != nil { frontierAbort("transcript") }
	var records []frontierRecord
	err = json.NewDecoder(file).Decode(&records); file.Close()
	if err != nil || len(records) != 1000 { frontierAbort("transcript") }
	for _, record := range records {
		if len(record.Request) == 0 || len(record.Response) == 0 || len(record.RefRequest) != Params.RequestBytes() { frontierAbort("transcript") }
	}
	upload := make([]int, len(records))
	download := make([]int, len(records))
	observations := make([][]uint64, len(records))
	original, err := json.Marshal(records); if err != nil { frontierAbort("transcript") }
	originalDigest := sha256.Sum256(original)
	if os.Args[5] == "validate" {
		// Reference-format transcripts only: every word against the expanded-set oracle.
		for index, record := range records {
			if !bytes.Equal(record.Request, record.RefRequest) { frontierAbort("not a reference transcript") }
			keys, offsets, _ := refwire.DecodeRequest(Params, record.Request)
			candidate, auxiliary := frontierOracle(server,keys,offsets)
			got, gotAuxiliary, ok := refwire.DecodeResponse(Params, record.Response)
			if !ok || !reflect.DeepEqual(candidate,got) || !reflect.DeepEqual(auxiliary,gotAuxiliary) { frontierAbort("reference response differs from the oracle") }
			upload[index], download[index] = len(record.Request), len(record.Response)
		}
	} else {
		frontierGuard(server, records, originalDigest)
		for pass := 0; pass < 6; pass++ {
			for index, record := range records {
				request := append([]byte(nil), record.Request...)
				response, elapsed := server.TimedQuery(request)
				if elapsed == 0 || !bytes.Equal(response, record.Response) { frontierAbort("response differs from the client-verified experiment") }
				if !bytes.Equal(request, record.Request) { frontierAbort("request mutated") }
				upload[index], download[index] = len(request), len(response)
				if pass > 0 { observations[index] = append(observations[index],elapsed) }
			}
		}
		guardDigest := sha256.Sum256(originalDigest[:])
		frontierGuard(server, records, guardDigest)
	}
	final, err := json.Marshal(records); if err != nil || sha256.Sum256(final) != originalDigest { frontierAbort("transcript mutated") }
	for index := uint64(0); index < ChunkSize*SetSize; index++ {
		entry := util.GenDBEntry(DBSeed,index)
		for word := 0; word < util.DBEntryLength; word++ { if server.DB[index*util.DBEntryLength+uint64(word)] != entry[word] { frontierAbort("database mutated") } }
	}
	if runtime.NumGoroutine() > 1 { frontierAbort("goroutines outlive the handler") }
	output, err := os.Create(os.Args[4]); if err != nil { frontierAbort("output") }
	err = json.NewEncoder(output).Encode(map[string]interface{}{"times":observations,"upload":upload,"download":download})
	output.Close(); if err != nil { frontierAbort("output") }
	os.Exit(0)
}
