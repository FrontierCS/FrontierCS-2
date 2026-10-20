package pir

import (
	"encoding/binary"
	"sync"

	"easypir/refwire"

	"example.com/util"
)

type LocalSetEC24 struct {
	Tag          uint32
	SetParity    util.DBEntry
	BlockParity  util.DBEntry
	ProgPointVal util.DBEntry
}

type LocalBackupSetGroupEC24 struct {
	consumed uint64
	sets     []LocalSetEC24
}

type LocalReplacementGroup struct {
	consumed uint64
	indices  []uint64
	value    []util.DBEntry
}

// Client is the reference client: the original preprocessing, query,
// reconstruction and hint-refresh code behind the harness interface.
type Client struct {
	p   refwire.Params
	gen *refwire.Generator

	localSetNum          uint64
	backupSetNumPerGroup uint64
	totalBackupSetNum    uint64

	localSets              []LocalSetEC24
	localBackupSets        []LocalSetEC24
	localCache             map[uint64]util.DBEntry
	localMissElements      map[uint64]util.DBEntry
	LocalBackupSetGroups   []LocalBackupSetGroupEC24
	LocalReplacementGroups []LocalReplacementGroup
}

// NewClient starts the preprocessing clock's work: it derives the client
// keys from the coin seed and allocates the hint tables.
func NewClient(p refwire.Params, coinSeed int64) *Client {
	gen := refwire.NewGenerator(p, coinSeed)
	c := &Client{p: p, gen: gen}
	localSetNum := gen.LocalSetNum
	backupSetNumPerGroup := gen.BackupPerGroup
	totalBackupSetNum := backupSetNumPerGroup * p.SetSize
	c.localSetNum = localSetNum
	c.backupSetNumPerGroup = backupSetNumPerGroup
	c.totalBackupSetNum = totalBackupSetNum

	// Initialize local sets and backup sets

	localSets := make([]LocalSetEC24, localSetNum)
	localBackupSets := make([]LocalSetEC24, totalBackupSetNum)
	c.localCache = make(map[uint64]util.DBEntry)
	c.localMissElements = make(map[uint64]util.DBEntry)
	tagCounter := uint32(0)

	for j := uint64(0); j < localSetNum; j++ {
		localSets[j] = LocalSetEC24{
			Tag:          tagCounter,
			SetParity:    util.ZeroEntry(),
			BlockParity:  util.ZeroEntry(),
			ProgPointVal: util.ZeroEntry(),
		}
		tagCounter += 1
	}

	LocalBackupSetGroups := make([]LocalBackupSetGroupEC24, p.SetSize)
	LocalReplacementGroups := make([]LocalReplacementGroup, p.SetSize)

	for i := uint64(0); i < p.SetSize; i++ {
		LocalBackupSetGroups[i].consumed = 0
		LocalBackupSetGroups[i].sets = localBackupSets[i*backupSetNumPerGroup : (i+1)*backupSetNumPerGroup]

		LocalReplacementGroups[i].consumed = 0
		LocalReplacementGroups[i].indices = make([]uint64, backupSetNumPerGroup)
		LocalReplacementGroups[i].value = make([]util.DBEntry, backupSetNumPerGroup)
	}

	for j := uint64(0); j < p.SetSize; j++ {
		for k := uint64(0); k < backupSetNumPerGroup; k++ {
			LocalBackupSetGroups[j].sets[k] = LocalSetEC24{
				Tag:          tagCounter,
				SetParity:    util.ZeroEntry(),
				BlockParity:  util.ZeroEntry(),
				ProgPointVal: util.ZeroEntry(),
			}
			tagCounter += 1
		}
	}
	c.localSets = localSets
	c.localBackupSets = localBackupSets
	c.LocalBackupSetGroups = LocalBackupSetGroups
	c.LocalReplacementGroups = LocalReplacementGroups
	return c
}

// Preprocess consumes the streamed database. next returns the chunks in
// order, two words per entry, and nil after the last chunk.
//
//	When the client gets i-th chunk, it updates all local sets' parities
//	It also needs to update the backup sets' parities
//	It also needs to update the i-th group's punct point parity
func (c *Client) Preprocess(next func() []uint64) {
	ChunkSize, SetSize, SBSize, SBSizeLog2 := c.p.ChunkSize, c.p.SetSize, c.p.SBSize, c.p.SBSizeLog2
	threadNum := c.p.Threads
	longKey := c.gen.LongKey
	localSetNum, backupSetNumPerGroup, totalBackupSetNum := c.localSetNum, c.backupSetNumPerGroup, c.totalBackupSetNum
	localSets, localBackupSets := c.localSets, c.localBackupSets
	LocalBackupSetGroups, LocalReplacementGroups := c.LocalBackupSetGroups, c.LocalReplacementGroups
	localMissElements := c.localMissElements

	// temporal storage
	tempLongKey := make([]*refwire.LongKey, localSetNum)
	tempLongKeyBackup := make([]*refwire.LongKey, totalBackupSetNum)

	for i := uint64(0); i < SetSize; i++ {
		if i%SBSize == 0 {
			for j := uint64(0); j < localSetNum; j++ {
				tempSubKey := refwire.SubKey(longKey, localSets[j].Tag, i>>SBSizeLog2)
				tempLongKey[j] = refwire.NewLongKey(&tempSubKey)
			}
			for j := uint64(0); j < totalBackupSetNum; j++ {
				tempSubKey := refwire.SubKey(longKey, localBackupSets[j].Tag, i>>SBSizeLog2)
				tempLongKeyBackup[j] = refwire.NewLongKey(&tempSubKey)
			}
		}

		chunk := next()
		if chunk == nil {
			break
		}
		hitMap := make([]bool, ChunkSize)

		// use multiple threads to parallelize the computation for the chunk

		var wg sync.WaitGroup
		wg.Add(int(threadNum))

		perTheadSetNum := (localSetNum+threadNum-1)/threadNum + 1 // make sure all sets are covered
		perThreadBackupNum := (totalBackupSetNum+threadNum-1)/threadNum + 1

		for tid := uint64(0); tid < threadNum; tid++ {
			startIndex := uint64(tid) * uint64(perTheadSetNum)
			endIndex := startIndex + uint64(perTheadSetNum)
			if endIndex > localSetNum {
				endIndex = localSetNum
			}

			startIndexBackup := uint64(tid) * uint64(perThreadBackupNum)
			endIndexBackup := startIndexBackup + uint64(perThreadBackupNum)
			if endIndexBackup > totalBackupSetNum {
				endIndexBackup = totalBackupSetNum
			}

			go func(start, end, start1, end1 uint64) {
				defer wg.Done()
				for j := uint64(start); j < uint64(end); j++ {
					offset := refwire.Eval(tempLongKey[j], i&(SBSize-1)) & (ChunkSize - 1)
					hitMap[offset] = true
					util.DBEntryXorFromRaw(&localSets[j].SetParity, chunk[offset*util.DBEntryLength:(offset+1)*util.DBEntryLength])
				}
				for j := uint64(start1); j < uint64(end1); j++ {
					offset := refwire.Eval(tempLongKeyBackup[j], i&(SBSize-1)) & (ChunkSize - 1)
					util.DBEntryXorFromRaw(&localBackupSets[j].SetParity, chunk[offset*util.DBEntryLength:(offset+1)*util.DBEntryLength])
				}
			}(startIndex, endIndex, startIndexBackup, endIndexBackup)
		}

		wg.Wait()

		for j := uint64(0); j < ChunkSize; j++ {
			if hitMap[j] == false {
				entry := util.DBEntryFromSlice(chunk[j*util.DBEntryLength : (j+1)*util.DBEntryLength])
				localMissElements[j+i*ChunkSize] = entry
			}
		}

		// for the i-th group of backups, set the ProgPointVal to the i-th chunk
		for k := uint64(0); k < backupSetNumPerGroup; k++ {
			offset := refwire.SetOffset(longKey, LocalBackupSetGroups[i].sets[k].Tag, i, c.p)
			util.DBEntryXorFromRaw(&LocalBackupSetGroups[i].sets[k].ProgPointVal, chunk[offset*util.DBEntryLength:(offset+1)*util.DBEntryLength])
		}

		// for the groups in the same superblock, update their block parities
		SBId := i / SBSize
		for j := SBId * SBSize; j < (SBId+1)*SBSize; j++ {
			for k := uint64(0); k < backupSetNumPerGroup; k++ {
				offset := refwire.SetOffset(longKey, LocalBackupSetGroups[j].sets[k].Tag, i, c.p)
				util.DBEntryXorFromRaw(&LocalBackupSetGroups[j].sets[k].BlockParity, chunk[offset*util.DBEntryLength:(offset+1)*util.DBEntryLength])
			}
		}

		// store the replacement: random offsets between 0 and ChunkSize - 1
		c.gen.DrawReplacements(i)
		for k := uint64(0); k < backupSetNumPerGroup; k++ {
			index := c.gen.ReplIndices[i*backupSetNumPerGroup+k]
			offset := index - i*ChunkSize
			LocalReplacementGroups[i].indices[k] = index
			LocalReplacementGroups[i].value[k] = util.DBEntryFromSlice(chunk[offset*util.DBEntryLength : (offset+1)*util.DBEntryLength])
		}
	}
}

// ExportHints emits the canonical hint export. The harness calls it once,
// inside the preprocessing clock, directly after Preprocess; an
// implementation that stores hints differently must derive the same bytes.
//
//	for each primary hint j, in order:      tag (uint32), set parity
//	for each backup hint (chunk g, slot k): tag (uint32), set parity,
//	                                        block parity, programmed-point value
//	for each replacement (chunk g, slot k): database index (uint64), value
//
// Integers are little-endian; a parity or value is two uint64 words.
func (c *Client) ExportHints(emit func([]byte)) {
	buffer := make([]byte, 0, 1<<16)
	put := func(entry *util.DBEntry) {
		for i := 0; i < util.DBEntryLength; i++ {
			buffer = binary.LittleEndian.AppendUint64(buffer, entry[i])
		}
	}
	flush := func(force bool) {
		if force || len(buffer) > 1<<16-128 {
			emit(buffer)
			buffer = buffer[:0]
		}
	}
	for j := range c.localSets {
		buffer = binary.LittleEndian.AppendUint32(buffer, c.localSets[j].Tag)
		put(&c.localSets[j].SetParity)
		flush(false)
	}
	for j := range c.localBackupSets {
		set := &c.localBackupSets[j]
		buffer = binary.LittleEndian.AppendUint32(buffer, set.Tag)
		put(&set.SetParity)
		put(&set.BlockParity)
		put(&set.ProgPointVal)
		flush(false)
	}
	for g := range c.LocalReplacementGroups {
		group := &c.LocalReplacementGroups[g]
		for k := range group.indices {
			buffer = binary.LittleEndian.AppendUint64(buffer, group.indices[k])
			put(&group.value[k])
			flush(false)
		}
	}
	flush(true)
}

// Query returns the bytes this client sends for database index x. The
// harness requires them to equal ReduceRequest applied to the reference
// request for the same coins and query sequence.
func (c *Client) Query(x uint64) []byte {
	// find a local set that contains x, expand it and edit it
	subkeyBytes, offsets := c.gen.Query(x)
	return refwire.EncodeRequest(c.p, subkeyBytes, offsets)
}

// Reconstruct turns the server's response to the last Query into the
// database entry and refreshes the consumed hint.
func (c *Client) Reconstruct(response []byte) util.DBEntry {
	ChunkSize, SBSize := c.p.ChunkSize, c.p.SBSize
	last := c.gen.Last
	x, hitSetId, caseId, y, chunkId, SBofX := last.X, last.HitSetId, last.CaseId, last.Y, last.ChunkId, last.SBofX
	localSets := c.localSets

	paritiesFromListDecoding, paritiesFromKeys, ok := refwire.DecodeResponse(c.p, response)
	if !ok {
		panic("malformed response")
	}

	repVal := util.ZeroEntry()
	if last.ReplUsed {
		// consume one replacement
		repVal = c.LocalReplacementGroups[chunkId].value[last.ReplSlot]
		c.LocalReplacementGroups[chunkId].consumed++
	}

	var xVal util.DBEntry
	// this is the common case 0
	xVal = localSets[hitSetId].SetParity                                                                           // the parity of the hit set
	util.DBEntryXorFromRaw(&xVal, paritiesFromListDecoding[SBofX*util.DBEntryLength:(SBofX+1)*util.DBEntryLength]) // xor the parity of the edited set
	util.DBEntryXor(&xVal, &repVal)                                                                                // xor the replacement value

	if caseId == 1 {
		// in this case, we just need to xor the parity of y and the original point in chunk(y)
		util.DBEntryXor(&xVal, &localSets[hitSetId].ProgPointVal)
	}

	if caseId == 2 {
		// first xor the alpha_superblock(y)
		util.DBEntryXorFromRaw(&xVal, paritiesFromKeys[y/(ChunkSize*SBSize)*util.DBEntryLength:(y/(ChunkSize*SBSize)+1)*util.DBEntryLength])
		// second, xor the original block's parity
		util.DBEntryXor(&xVal, &localSets[hitSetId].BlockParity)
	}

	// update the local cache
	c.localCache[x] = xVal

	// Refresh Phase:
	// The client picks one set from the x's backup group
	// adds the x to the set and adds the set to the local set list
	consumed := c.LocalBackupSetGroups[chunkId].consumed
	localSets[hitSetId].Tag = c.LocalBackupSetGroups[chunkId].sets[consumed].Tag
	localSets[hitSetId].SetParity = c.LocalBackupSetGroups[chunkId].sets[consumed].SetParity
	localSets[hitSetId].BlockParity = c.LocalBackupSetGroups[chunkId].sets[consumed].BlockParity
	localSets[hitSetId].ProgPointVal = c.LocalBackupSetGroups[chunkId].sets[consumed].ProgPointVal
	// edit the progpointval
	util.DBEntryXor(&localSets[hitSetId].ProgPointVal, &xVal)
	c.LocalBackupSetGroups[chunkId].consumed++
	return xVal
}

// PrimaryHint returns the canonical record of primary hint j, as ExportHints
// emits it: its tag and its set parity. The harness calls it after the online
// queries, and only for hints that no query has consumed.
func (c *Client) PrimaryHint(j uint64) (uint32, util.DBEntry) {
	return c.localSets[j].Tag, c.localSets[j].SetParity
}
