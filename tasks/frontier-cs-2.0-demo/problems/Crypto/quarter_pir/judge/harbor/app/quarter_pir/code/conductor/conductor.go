// Trusted conductor for one complete client experiment. The evaluator builds
// it from its own trusted copy of this tree; it contains no submission code
// (it does not import package pir) and holds everything a submission must not influence: the
// query stream, the reference request generator, the transport to the server,
// the clocks, the value oracle and the result record.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"easypir/frame"
	"easypir/rawrpc"
	"easypir/refwire"

	pb "example.com/query"
	util "example.com/util"
	"google.golang.org/grpc"
)

type config struct {
	Upstream   string `json:"upstream"`
	Client     string `json:"client"`
	Reducer    string `json:"reducer"`
	Size       uint64 `json:"size"`
	DBSeed     uint64 `json:"db_seed"`
	CoinSeed   int64  `json:"coin_seed"`
	QuerySeed  int64  `json:"query_seed"`
	ProbeSeed  int64  `json:"probe_seed"`
	Queries    uint64 `json:"queries"`
	Threads    uint64 `json:"threads"`
	Transcript string `json:"transcript"`
	Result     string `json:"result"`
}

type record struct {
	Request    []byte `json:"request"`
	Response   []byte `json:"response"`
	RefRequest []byte `json:"ref_request"`
}

type result struct {
	Valid               bool   `json:"valid"`
	Failure             string `json:"failure"`
	Size                uint64 `json:"db_size"`
	Queries             uint64 `json:"actual_queries"`
	PlannedQueries      uint64 `json:"planned_queries"`
	Threads             uint64 `json:"threads"`
	OfflineNs           int64  `json:"offline_elapsed_ns"`
	OnlineNs            int64  `json:"online_elapsed_ns"`
	OnlineQueryNs       int64  `json:"online_query_ns"`
	OnlineTransportNs   int64  `json:"online_transport_ns"`
	OnlineReconstructNs int64  `json:"online_reconstruct_ns"`
	DriverOfflineNs     uint64 `json:"driver_offline_ns"`
	DriverOnlineNs      uint64 `json:"driver_online_ns"`
	StorageBytes        uint64 `json:"client_storage_bytes"`
	PeakRSSBytes        uint64 `json:"client_peak_rss_bytes"`
	HintBytes           uint64 `json:"hint_export_bytes"`
	HintSHA256          string `json:"hint_export_sha256"`
	TranscriptSHA256    string `json:"transcript_sha256"`
	UploadBytes         []int  `json:"upload_bytes"`
	DownloadBytes       []int  `json:"download_bytes"`
	OfflineCommBytes    uint64 `json:"offline_communication_bytes"`
	// Coverage probe: the untimed queries after the measured ones. They enter
	// no clock, no byte count and no transcript. ProbeAnswered is the number of
	// leading probe queries that passed every check; ProbeFailure names the
	// check the next one failed, or is empty when all passed.
	ProbeQueries      uint64 `json:"probe_queries"`
	ProbeAnswered     uint64 `json:"probe_answered"`
	ProbeFailure      string `json:"probe_failure"`
	ProbeSHA256       string `json:"probe_sha256"`
	ProbeBackup       uint64 `json:"probe_backup_queries"`
	ProbeDeepestHint  uint64 `json:"probe_deepest_hint"`
	ProbeMispredicted uint64 `json:"probe_mispredicted"`
	ProbePlanNs       int64  `json:"probe_plan_ns"`
	ProbeNs           int64  `json:"probe_elapsed_ns"`
	// Primary-hint audit after the online phase (only when the probe passed).
	// AuditMismatched lists the positions of the asked hints whose record was
	// not the canonical one; AuditFailure is set when the client did not answer.
	AuditHints      uint64 `json:"audit_hints"`
	AuditMismatched []int  `json:"audit_mismatched"`
	AuditFailure    string `json:"audit_failure"`
	AuditSHA256     string `json:"audit_sha256"`
	AuditRecords    string `json:"audit_records_sha256"`
	AuditNs         int64  `json:"audit_elapsed_ns"`
}

const requestLimit = 1 << 20

var outcome result
var settings config
var children []*exec.Cmd

func finish(failure string) {
	outcome.Valid = failure == ""
	outcome.Failure = failure
	for _, child := range children {
		if child.Process != nil {
			child.Process.Kill()
		}
	}
	body, err := json.Marshal(outcome)
	if err != nil || os.WriteFile(settings.Result, body, 0o600) != nil {
		os.Exit(2)
	}
	os.Exit(0)
}

func harnessError(message string) {
	fmt.Fprintln(os.Stderr, "conductor:", message)
	for _, child := range children {
		if child.Process != nil {
			child.Process.Kill()
		}
	}
	os.Exit(2)
}

type limitedWriter struct {
	file  *os.File
	space int
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	if w.space > 0 {
		part := p
		if len(part) > w.space {
			part = part[:w.space]
		}
		w.file.Write(part)
		w.space -= len(part)
	}
	return len(p), nil
}

func spawn(path string, log string) (*bufio.Writer, *bufio.Reader) {
	// The processes that run submission code are told the database size and the
	// thread count only: their query count is the planned budget, never the
	// number of queries this run measures or issues.
	command := exec.Command(path, "-size", strconv.FormatUint(settings.Size, 10),
		"-thread", strconv.FormatUint(settings.Threads, 10))
	stdin, err := command.StdinPipe()
	if err != nil {
		harnessError("pipe")
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		harnessError("pipe")
	}
	if file, err := os.OpenFile(log, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600); err == nil {
		command.Stderr = &limitedWriter{file: file, space: 1 << 20}
	}
	if command.Start() != nil {
		harnessError("cannot start " + path)
	}
	children = append(children, command)
	return bufio.NewWriterSize(stdin, 1<<20), bufio.NewReaderSize(stdout, 1<<16)
}

// peakResident reads a child's resident-set high-water mark from /proc; it is
// used only when the client can no longer report it itself.
func peakResident(child *exec.Cmd) uint64 {
	if child.Process == nil {
		return 0
	}
	status, err := os.ReadFile("/proc/" + strconv.Itoa(child.Process.Pid) + "/status")
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

func main() {
	if len(os.Args) != 2 {
		os.Exit(2)
	}
	body, err := os.ReadFile(os.Args[1])
	if err != nil || json.Unmarshal(body, &settings) != nil {
		os.Exit(2)
	}
	params := refwire.NewPlannedParams(settings.Size, settings.Threads)
	outcome = result{Size: settings.Size, Queries: settings.Queries, Threads: settings.Threads,
		PlannedQueries: params.PlannedQueries(), OfflineCommBytes: settings.Size * util.DBEntrySize}

	startup, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	connection, err := grpc.DialContext(startup, settings.Upstream, grpc.WithBlock(), grpc.WithInsecure(),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(12*1024*1024), grpc.MaxCallSendMsgSize(12*1024*1024)))
	if err != nil {
		harnessError("server unavailable")
	}
	defer connection.Close()
	server := pb.NewQueryServiceClient(connection)
	ctx := context.Background()

	// The trusted generator never sees the submission and runs outside every clock.
	generator := refwire.NewGenerator(params, settings.CoinSeed)
	for chunk := uint64(0); chunk < params.SetSize; chunk++ {
		generator.DrawReplacements(chunk)
	}
	queryRng := rand.New(rand.NewSource(settings.QuerySeed))
	issued := map[uint64]bool{}

	toClient, fromClient := spawn(settings.Client, settings.Result+".client.log")
	toReducer, fromReducer := spawn(settings.Reducer, settings.Result+".reducer.log")

	receive := func(want byte, limit int) ([]byte, bool) {
		tag, payload, err := frame.Read(fromClient, limit, nil)
		return payload, err == nil && tag == want
	}
	expect := func(want byte, limit int) []byte {
		payload, ok := receive(want, limit)
		if !ok {
			finish("client_crashed_or_protocol")
		}
		return payload
	}
	expect('R', 0)

	var seed [8]byte
	binary.LittleEndian.PutUint64(seed[:], uint64(settings.CoinSeed))
	pumped := make(chan bool, 1)
	offlineStart := time.Now()
	if frame.Write(toClient, 'S', seed[:]) != nil {
		finish("client_crashed_or_protocol")
	}
	// Exactly one FetchFullDB, issued by the conductor itself.
	stream, err := server.FetchFullDB(ctx, &pb.FetchFullDBMsg{Dummy: 1})
	if err != nil {
		harnessError("FetchFullDB failed")
	}
	go func() {
		buffer := make([]byte, 8*2*params.ChunkSize)
		for chunks := uint64(0); ; chunks++ {
			chunk, err := stream.Recv()
			if err == io.EOF {
				pumped <- chunks == params.SetSize && frame.Write(toClient, 'D', nil) == nil
				return
			}
			if err != nil || len(chunk.Chunk) != len(buffer)/8 {
				pumped <- false
				return
			}
			for i, word := range chunk.Chunk {
				binary.LittleEndian.PutUint64(buffer[8*i:], word)
			}
			if frame.Write(toClient, 'C', buffer) != nil {
				pumped <- false
				return
			}
		}
	}()
	expect('p', 0)
	outcome.OfflineNs = time.Since(offlineStart).Nanoseconds()
	select {
	case ok := <-pumped:
		if !ok {
			finish("client_crashed_or_protocol")
		}
	case <-time.After(30 * time.Second):
		finish("client_crashed_or_protocol")
	}
	report := expect('H', 64)
	if len(report) != 57 {
		finish("client_crashed_or_protocol")
	}
	outcome.HintSHA256 = hex.EncodeToString(report[:32])
	outcome.DriverOfflineNs = binary.LittleEndian.Uint64(report[32:])
	outcome.StorageBytes = binary.LittleEndian.Uint64(report[40:])
	outcome.HintBytes = binary.LittleEndian.Uint64(report[48:])
	if report[56] != 0 || outcome.HintBytes != uint64(params.HintExportBytes()) {
		finish("hint_export_length")
	}

	// exchange is one complete online query: the reference request, its
	// reduction, the client's request, the server round trip and the checked
	// value. It returns the failed check, or "" with the carried messages and
	// the three clocked intervals.
	var word [8]byte
	exchange := func(x uint64) (failure string, entry record, intervals [3]int64) {
		issued[x] = true
		keys, offsets := generator.Query(x)
		reference := refwire.EncodeRequest(params, keys, offsets)
		if frame.Write(toReducer, 'r', reference) != nil {
			return "reducer_crashed", entry, intervals
		}
		tag, reduced, err := frame.Read(fromReducer, requestLimit, nil)
		if err != nil || (tag != 'c' && tag != 'x') {
			return "reducer_crashed", entry, intervals
		}
		if tag == 'x' {
			return "reduction_nondeterministic", entry, intervals
		}

		binary.LittleEndian.PutUint64(word[:], x)
		begin := time.Now()
		if frame.Write(toClient, 'Q', word[:]) != nil {
			return "client_crashed_or_protocol", entry, intervals
		}
		request, ok := receive('q', requestLimit)
		if !ok {
			return "client_crashed_or_protocol", entry, intervals
		}
		intervals[0] = time.Since(begin).Nanoseconds()
		if !bytes.Equal(request, reduced) {
			return "request_not_reduction", entry, intervals
		}

		// The wire carries the trusted reduction, never client-written bytes.
		begin = time.Now()
		response, err := rawrpc.Call(ctx, connection, reduced)
		intervals[1] = time.Since(begin).Nanoseconds()
		if err != nil || len(response) == 0 {
			return "server_rpc_failed", entry, intervals
		}

		begin = time.Now()
		if frame.Write(toClient, 'A', response) != nil {
			return "client_crashed_or_protocol", entry, intervals
		}
		answer, ok := receive('V', 16)
		if !ok {
			return "client_crashed_or_protocol", entry, intervals
		}
		intervals[2] = time.Since(begin).Nanoseconds()
		truth := util.GenDBEntry(settings.DBSeed, x)
		if len(answer) != 16 || binary.LittleEndian.Uint64(answer[0:]) != truth[0] || binary.LittleEndian.Uint64(answer[8:]) != truth[1] {
			return "wrong_value", entry, intervals
		}
		return "", record{Request: reduced, Response: response, RefRequest: reference}, intervals
	}

	transcript := make([]record, 0, settings.Queries)
	digest := sha256.New()
	longestRequest, longestResponse := 0, 0
	for q := uint64(0); q < settings.Queries; q++ {
		x := queryRng.Uint64() % settings.Size
		for issued[x] {
			x = queryRng.Uint64() % settings.Size
		}
		failure, entry, intervals := exchange(x)
		if failure != "" {
			finish(failure)
		}
		outcome.OnlineQueryNs += intervals[0]
		outcome.OnlineTransportNs += intervals[1]
		outcome.OnlineReconstructNs += intervals[2]

		outcome.UploadBytes = append(outcome.UploadBytes, len(entry.Request))
		outcome.DownloadBytes = append(outcome.DownloadBytes, len(entry.Response))
		if len(entry.Request) > longestRequest {
			longestRequest = len(entry.Request)
		}
		if len(entry.Response) > longestResponse {
			longestResponse = len(entry.Response)
		}
		transcript = append(transcript, entry)
		for _, part := range [][]byte{entry.Request, entry.Response} {
			binary.LittleEndian.PutUint64(word[:], uint64(len(part)))
			digest.Write(word[:])
			digest.Write(part)
		}
	}
	outcome.OnlineNs = outcome.OnlineQueryNs + outcome.OnlineTransportNs + outcome.OnlineReconstructNs
	outcome.TranscriptSHA256 = hex.EncodeToString(digest.Sum(nil))

	// Coverage probe. Every clock, byte count and the transcript are complete
	// at this point; nothing below changes a measured quantity. A probe query
	// is checked exactly like a measured one and, in addition, neither of its
	// messages may be longer than the longest measured message of its
	// direction. The probe stops at the first failed check and the record
	// stays a completed run: the evaluator decides, against the paired
	// reference run on the same sequence, whether that query counts.
	probeStart := time.Now()
	plan := planProbe(generator, issued, settings.ProbeSeed, int(settings.Threads))
	outcome.ProbePlanNs = time.Since(probeStart).Nanoseconds()
	outcome.ProbeQueries = uint64(len(plan))
	planDigest := sha256.New()
	for _, query := range plan {
		binary.LittleEndian.PutUint64(word[:], query.X)
		planDigest.Write(word[:])
		if query.Backup {
			outcome.ProbeBackup++
		}
		if query.Hit > outcome.ProbeDeepestHint {
			outcome.ProbeDeepestHint = query.Hit
		}
	}
	outcome.ProbeSHA256 = hex.EncodeToString(planDigest.Sum(nil))
	for _, query := range plan {
		failure, entry, _ := exchange(query.X)
		if generator.Last.X != query.X || generator.Last.HitSetId != query.Hit {
			outcome.ProbeMispredicted++
		}
		if failure == "" && (len(entry.Request) > longestRequest || len(entry.Response) > longestResponse) {
			failure = "message_longer_than_measured"
		}
		if failure != "" {
			outcome.ProbeFailure = failure
			break
		}
		outcome.ProbeAnswered++
	}
	outcome.ProbeNs = time.Since(probeStart).Nanoseconds()

	// Primary-hint audit, untimed. The client driver compares each record
	// with the canonical hint export it holds; the evaluator pairs the result
	// with the reference run, which is asked for the same hints.
	outcome.AuditMismatched = []int{}
	if outcome.ProbeFailure == "" {
		auditStart := time.Now()
		hints := planAudit(generator, settings.ProbeSeed)
		outcome.AuditHints = uint64(len(hints))
		ask := make([]byte, 8*len(hints))
		for i, hint := range hints {
			binary.LittleEndian.PutUint64(ask[8*i:], hint)
		}
		asked := sha256.Sum256(ask)
		outcome.AuditSHA256 = hex.EncodeToString(asked[:])
		reply, ok := []byte(nil), frame.Write(toClient, 'U', ask) == nil
		if ok {
			reply, ok = receive('u', len(hints)+32)
		}
		if !ok || len(reply) != len(hints)+32 {
			outcome.AuditFailure = "client_crashed_or_protocol"
		} else {
			for i := range hints {
				if reply[i] != 1 {
					outcome.AuditMismatched = append(outcome.AuditMismatched, i)
				}
			}
			outcome.AuditRecords = hex.EncodeToString(reply[len(hints):])
		}
		outcome.AuditNs = time.Since(auditStart).Nanoseconds()
	}

	if outcome.ProbeFailure == "" && outcome.AuditFailure == "" {
		if frame.Write(toClient, 'E', nil) != nil {
			finish("client_crashed_or_protocol")
		}
		done := expect('F', 16)
		if len(done) != 16 {
			finish("client_crashed_or_protocol")
		}
		outcome.DriverOnlineNs = binary.LittleEndian.Uint64(done)
		outcome.PeakRSSBytes = binary.LittleEndian.Uint64(done[8:])
	} else {
		// The client may be gone or out of step; its own report is not asked for.
		outcome.PeakRSSBytes = peakResident(children[0])
	}
	body, err = json.Marshal(transcript)
	if err != nil || os.WriteFile(settings.Transcript, body, 0o600) != nil {
		harnessError("cannot write transcript")
	}
	finish("")
}
