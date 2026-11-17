package vm

import (
	"encoding/binary"
	"os"
)

// The submitted predictor's answers, written as they are given and each with a write of its own,
// so that a later call cannot revise one. A record is a kind byte, the number of the branch info
// asked (var_table.go numbers them in creation order) as a uvarint and the answer as a varint:
// 'P' for what predict(false) returned and 'C' for the observationCount that the sweep read. The
// judge's own build replays them (replay.go).
var answerFile = openLoggedLog()

func openLoggedLog() *os.File {
	file, err := os.OpenFile("predictions.log", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o666)
	if err != nil {
		panic(err)
	}
	return file
}

func writeLogged(kind byte, seq, value int) {
	record := binary.AppendUvarint([]byte{kind}, uint64(seq))
	record = binary.AppendVarint(record, int64(value))
	if _, err := answerFile.Write(record); err != nil {
		panic(err)
	}
}

func (br *branchInfo) queryPredictor() int {
	res := br.predictor.predict(false)
	writeLogged('P', br.seq, res)
	return res
}

func (br *branchInfo) readObservationCount() int {
	count := br.predictor.observationCount
	writeLogged('C', br.seq, count)
	return count
}
