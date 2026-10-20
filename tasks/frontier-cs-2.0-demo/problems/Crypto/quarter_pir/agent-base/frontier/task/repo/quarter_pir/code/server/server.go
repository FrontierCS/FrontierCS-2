// Trusted, byte-locked server entrypoint. It owns database generation, the
// transport and the server timer; only pir.HandleQuery is submission code.
package main

import (
	"bufio"
	"flag"
	"log"
	"net"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"easypir/pir"
	"easypir/rawrpc"
	"easypir/refwire"
	"easypir/settle"

	pb "example.com/query"
	util "example.com/util"
	"google.golang.org/grpc"
)

var DBSize uint64
var DBSeed uint64
var ChunkSize uint64
var SetSize uint64
var SBSize uint64
var SBNum uint64
var Params refwire.Params
var port string

type QueryServiceServer struct {
	pb.UnimplementedQueryServiceServer
	DB []uint64 // the database, for every DBEntrySize/8 uint64s, we store a DBEntry
}

func (s *QueryServiceServer) DBAccess(id uint64) util.DBEntry {
	if id < uint64(len(s.DB)) {
		if id*util.DBEntryLength+util.DBEntryLength > uint64(len(s.DB)) {
			log.Fatalf("DBAccess: id %d out of range", id)
		}
		return util.DBEntryFromSlice(s.DB[id*util.DBEntryLength : (id+1)*util.DBEntryLength])
	} else {
		var ret util.DBEntry
		for i := 0; i < util.DBEntryLength; i++ {
			ret[i] = 0
		}
		return ret
	}
}

func (s *QueryServiceServer) FetchFullDB(in *pb.FetchFullDBMsg, stream pb.QueryService_FetchFullDBServer) error {
	for i := uint64(0); i < SetSize; i++ {
		down := i * ChunkSize
		up := (i + 1) * ChunkSize
		var chunk []uint64
		chunk = s.DB[down*util.DBEntryLength : up*util.DBEntryLength]

		ret := &pb.DBChunk{ChunkId: i, ChunkSize: ChunkSize, Chunk: chunk}
		if err := stream.Send(ret); err != nil {
			log.Printf("Failed to send a chunk: %v", err)
			return err
		}
	}
	return nil
}

func ReadConfigInfo() (uint64, uint64) {
	file, err := os.Open("config.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	line, _, err := reader.ReadLine()
	if err != nil {
		log.Fatal(err)
	}
	split := strings.Split(string(line), " ")
	var DBSize uint64
	var DBSeed uint64

	if DBSize, err = strconv.ParseUint(split[0], 10, 64); err != nil {
		log.Fatal(err)
	}
	if DBSeed, err = strconv.ParseUint(split[1], 10, 64); err != nil {
		log.Fatal(err)
	}
	return uint64(DBSize), uint64(DBSeed)
}

// TimedQuery is the server clock, used by the live server and by the replay.
// It starts before the submission code sees the request bytes and stops only
// when the complete response bytes exist and every goroutine the handler
// started has exited, so request decoding, response encoding and any work the
// handler left running are inside the measurement. A goroutine that is still
// alive after settle.Limit ends the process: the call has failed.
func (s *QueryServiceServer) TimedQuery(request []byte) ([]byte, uint64) {
	goroutines := runtime.NumGoroutine()
	start := time.Now()
	response := pir.HandleQuery(s.DB, Params, request)
	if !settle.Wait(goroutines) {
		log.Fatal("handler goroutines outlive the call")
	}
	if len(response) == 0 {
		log.Fatal("response is empty")
	}
	since := time.Since(start)
	return response, uint64(since.Nanoseconds())
}

func (s *QueryServiceServer) RawQuery(request []byte) []byte {
	response, _ := s.TimedQuery(request)
	return response
}

func main() {
	portPtr := flag.String("port", "50051", "port number")
	ignoreInitPtr := flag.Bool("ignoreInit", false, "ignore initialization of the DB")
	flag.Parse()

	port = *portPtr
	ignoreInit := *ignoreInitPtr
	port = "127.0.0.1:" + port

	DBSize, DBSeed = ReadConfigInfo()
	Params = refwire.NewPlannedParams(DBSize, 4)
	ChunkSize, SetSize, SBSize, SBNum = Params.ChunkSize, Params.SetSize, Params.SBSize, Params.SBNum

	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("Failed to listen %v", err)
	}

	DB := make([]uint64, ChunkSize*SetSize*util.DBEntryLength)
	if !ignoreInit {
		for i := uint64(0); i < uint64(len(DB))/util.DBEntryLength; i++ {
			entry := util.GenDBEntry(DBSeed, i)
			copy(DB[i*util.DBEntryLength:(i+1)*util.DBEntryLength], entry[:])
		}
	}

	maxMsgSize := 12 * 1024 * 1024

	s := grpc.NewServer(
		grpc.MaxMsgSize(maxMsgSize),
		grpc.MaxRecvMsgSize(maxMsgSize),
		grpc.MaxSendMsgSize(maxMsgSize),
	)

	server := &QueryServiceServer{DB: DB[:]}
	pb.RegisterQueryServiceServer(s, server)
	rawrpc.Register(s, server)
	log.Printf("server listening at %v", lis.Addr())

	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to server %v", err)
	}
}
