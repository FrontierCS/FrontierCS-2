package config

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/params"
	"math/big"
)

func newUint64(val uint64) *uint64 { return &val }

var MainnetTerminalTotalDifficulty, _ = new(big.Int).SetString("58_750_000_000_000_000_000_000", 0)

// MainnetChainConfig is the chain parameters to run a node on the main network.
var MainnetChainConfig = &params.ChainConfig{
	ChainID:                       big.NewInt(1),
	HomesteadBlock:                big.NewInt(1_150_000),
	DAOForkBlock:                  big.NewInt(1_920_000),
	DAOForkSupport:                true,
	EIP150Block:                   big.NewInt(2_463_000),
	EIP150Hash:                    common.HexToHash("0x2086799aeebeae135c246c65021c82b4e15a2c451340993aacfd2751886514f0"),
	EIP155Block:                   big.NewInt(2_675_000),
	EIP158Block:                   big.NewInt(2_675_000),
	ByzantiumBlock:                big.NewInt(4_370_000),
	ConstantinopleBlock:           big.NewInt(7_280_000),
	PetersburgBlock:               big.NewInt(7_280_000),
	IstanbulBlock:                 big.NewInt(9_069_000),
	MuirGlacierBlock:              big.NewInt(9_200_000),
	BerlinBlock:                   big.NewInt(12_244_000),
	LondonBlock:                   big.NewInt(12_965_000),
	ArrowGlacierBlock:             big.NewInt(13_773_000),
	GrayGlacierBlock:              big.NewInt(15_050_000),
	TerminalTotalDifficulty:       MainnetTerminalTotalDifficulty, // 58_750_000_000_000_000_000_000
	TerminalTotalDifficultyPassed: true,
	ShanghaiTime:                  newUint64(1681338455),
	Ethash:                        new(params.EthashConfig),
}
