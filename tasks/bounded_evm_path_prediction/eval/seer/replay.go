package vm

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// The answers of the submitted predictor, in the format of record.go, which this build returns
// to the same queries: for each branch info, its n-th query gets the n-th prediction loggedByBranch for
// it, and its n-th sweep the n-th count. A query the log has no answer for ends the replay.
type loggedBranch struct {
	predictions []int
	counts      []int
}

var loggedByBranch = loadLogged()

func loadLogged() map[int]*loggedBranch {
	file, err := os.Open("../predictions.log")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	byBranch := make(map[int]*loggedBranch)
	reader := bufio.NewReader(file)
	for {
		kind, err := reader.ReadByte()
		if err == io.EOF {
			return byBranch
		} else if err != nil {
			panic(err)
		}
		seq, err := binary.ReadUvarint(reader)
		if err != nil {
			panic(err)
		}
		value, err := binary.ReadVarint(reader)
		if err != nil {
			panic(err)
		}
		branch := byBranch[int(seq)]
		if branch == nil {
			branch = &loggedBranch{}
			byBranch[int(seq)] = branch
		}
		switch kind {
		case 'P':
			branch.predictions = append(branch.predictions, int(value))
		case 'C':
			branch.counts = append(branch.counts, int(value))
		default:
			panic(fmt.Sprintf("unknown record kind %q", kind))
		}
	}
}

func takeLogged(list *[]int) int {
	if len(*list) == 0 {
		fmt.Fprintln(os.Stderr, "the log holds no answer for a query of the replay")
		os.Exit(3)
	}
	value := (*list)[0]
	*list = (*list)[1:]
	return value
}

func (br *branchInfo) loggedFor() *loggedBranch {
	if loggedByBranch[br.seq] == nil {
		loggedByBranch[br.seq] = &loggedBranch{}
	}
	return loggedByBranch[br.seq]
}

func (br *branchInfo) queryPredictor() int       { return takeLogged(&br.loggedFor().predictions) }
func (br *branchInfo) readObservationCount() int { return takeLogged(&br.loggedFor().counts) }
