package main

import (
	"github.com/urfave/cli/v2"
	"log"
	"os"
	"runtime"
	"seerEVM/experiments"
)

const (
	TestPreExecutionLarge = iota
	TestPrediction
	TestSpeedup
	TestConcurrent
	TestAbort
	TestBreakDown
	TestMemoryCPUBaseline
	TestMemoryCPU
	BranchStatistics
	TestMemoryCPUSweep
	TestIOBaseline
	TestIO
	TestMemoryCPUConcurrent
)

type SeerRun struct {
	// flags to determine which to run
	TestIndicator   int
	StartingBlock   int64
	BlockNum        int64
	TxNum           int
	Threads         int
	DisorderRatio   float64
	EnableRepair    bool
	EnablePredictor bool
	EnableFast      bool
	StoreCheckpoint bool
	FullStorage     bool
}

func (sr *SeerRun) Run() {
	app := &cli.App{
		Flags: []cli.Flag{
			&cli.IntFlag{
				Name:        "indicator",
				Aliases:     []string{"in"},
				Usage:       "Specify the test content",
				Value:       TestPreExecutionLarge,
				Destination: &sr.TestIndicator,
			},
			&cli.Int64Flag{
				Name:        "start",
				Aliases:     []string{"st"},
				Usage:       "Starting block height",
				Value:       14650000,
				Destination: &sr.StartingBlock,
			},
			&cli.Int64Flag{
				Name:        "blockNum",
				Aliases:     []string{"bn"},
				Usage:       "Number of blocks to replay",
				Value:       100,
				Destination: &sr.BlockNum,
			},
			&cli.IntFlag{
				Name:        "txNum",
				Aliases:     []string{"tn"},
				Usage:       "Number of transactions to replay (in large scale)",
				Value:       2000,
				Destination: &sr.TxNum,
			},
			&cli.IntFlag{
				Name:        "threads",
				Aliases:     []string{"th"},
				Usage:       "Number of threads for concurrent execution",
				Value:       4,
				Destination: &sr.Threads,
			},
			&cli.Float64Flag{
				Name:        "ratio",
				Aliases:     []string{"r"},
				Usage:       "Ratio of out-of-order transactions",
				Value:       0,
				Destination: &sr.DisorderRatio,
			},
			&cli.BoolFlag{
				Name:        "repair",
				Aliases:     []string{"rp"},
				Value:       false,
				Usage:       "Whether to enable pre-execution repair",
				Destination: &sr.EnableRepair,
			},
			&cli.BoolFlag{
				Name:        "predictor",
				Aliases:     []string{"pc"},
				Value:       false,
				Usage:       "Whether to enable predictor model",
				Destination: &sr.EnablePredictor,
			},
			&cli.BoolFlag{
				Name:        "fast",
				Aliases:     []string{"f"},
				Value:       false,
				Usage:       "Whether to enable fast-path execution",
				Destination: &sr.EnableFast,
			},
			&cli.BoolFlag{
				Name:        "checkpoint",
				Aliases:     []string{"cp"},
				Value:       false,
				Usage:       "Whether to enable checkpoint caching",
				Destination: &sr.StoreCheckpoint,
			},
			&cli.BoolFlag{
				Name:        "fullStorage",
				Aliases:     []string{"fs"},
				Value:       false,
				Usage:       "Whether to store all state variable versions",
				Destination: &sr.FullStorage,
			},
		},
	}

	err := app.Run(os.Args)
	if err != nil {
		log.Panic(err)
	}
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	seer := &SeerRun{}
	seer.Run()

	switch seer.TestIndicator {
	case TestPreExecutionLarge:
		experiments.TestPreExecutionLarge(seer.TxNum, seer.StartingBlock, seer.BlockNum, seer.DisorderRatio, seer.EnableRepair)
	case TestPrediction:
		experiments.TestPredictionSuccess(seer.StartingBlock, seer.BlockNum, 0)
	case TestSpeedup:
		experiments.TestSpeedupPerTx(seer.StartingBlock, seer.BlockNum, seer.DisorderRatio, seer.EnableRepair, seer.EnablePredictor, seer.EnableFast)
	case TestConcurrent:
		experiments.TestSeerConcurrentLarge(seer.Threads, seer.TxNum, seer.StartingBlock, seer.BlockNum)
	case TestAbort:
		experiments.TestSeerConcurrentAbort(seer.Threads, seer.TxNum, seer.StartingBlock, seer.BlockNum)
	case TestBreakDown:
		experiments.TestSeerBreakDown(seer.StartingBlock, seer.BlockNum, seer.DisorderRatio, seer.EnableRepair, seer.EnablePredictor, seer.EnableFast)
	case TestMemoryCPUBaseline:
		experiments.TestMemoryCPUBaseline(seer.StartingBlock, seer.BlockNum)
	case TestMemoryCPU:
		experiments.TestMemoryCPUBreakDown(seer.StartingBlock, seer.BlockNum, seer.EnableRepair, seer.EnablePredictor, seer.EnableFast, seer.StoreCheckpoint, seer.FullStorage)
	case BranchStatistics:
		experiments.TestGasUsed(seer.StartingBlock, seer.BlockNum)
	case TestMemoryCPUSweep:
		experiments.TestMemoryCPUSweep(seer.StartingBlock, seer.BlockNum)
	case TestIOBaseline:
		experiments.TestIOBaseline(seer.StartingBlock, seer.BlockNum)
	case TestIO:
		experiments.TestIOSeer(seer.StartingBlock, seer.BlockNum)
	case TestMemoryCPUConcurrent:
		experiments.TestMemoryCPUConcurrent(seer.Threads, seer.StartingBlock, seer.BlockNum)
	}
}
