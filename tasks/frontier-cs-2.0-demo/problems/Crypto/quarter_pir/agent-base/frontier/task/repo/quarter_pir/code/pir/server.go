package pir

import (
	"easypir/refwire"

	"example.com/util"
)

func dbAccess(db []uint64, id uint64) util.DBEntry {
	if id < uint64(len(db)) {
		if id*util.DBEntryLength+util.DBEntryLength > uint64(len(db)) {
			panic("dbAccess: id out of range")
		}
		return util.DBEntryFromSlice(db[id*util.DBEntryLength : (id+1)*util.DBEntryLength])
	} else {
		var ret util.DBEntry
		for i := 0; i < util.DBEntryLength; i++ {
			ret[i] = 0
		}
		return ret
	}
}

// HandleQuery answers one online request. db is the resident database, two
// words per entry; request is exactly what ReduceRequest produced. The
// returned bytes go to the client's Reconstruct unchanged. The trusted server
// timer wraps this whole call, including request decoding and response
// encoding.
func HandleQuery(db []uint64, p refwire.Params, request []byte) []byte {
	panic("list decoding is not implemented")
}
