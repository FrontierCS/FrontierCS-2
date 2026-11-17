package database

import "runtime"

type RawConfig struct {
	Path      string
	Cache     int
	Handles   int
	Ancient   string
	Namespace string
	ReadOnly  bool
}

func DefaultRawConfig() *RawConfig {
	if runtime.GOOS == "darwin" { // MacOS
		return &RawConfig{
			Path:      "../ethereumdata_1465_small/copychain",
			Cache:     2048,
			Handles:   5120,
			Ancient:   "../ethereumdata_1465_small/copychain/ancient",
			Namespace: "eth/db/chaindata/",
			ReadOnly:  false,
		}
	} else {
		return &RawConfig{
			Path:      "../ethereumdata_1465_small/copychain",
			Cache:     2048,
			Handles:   5120,
			Ancient:   "../ethereumdata_1465_small/copychain/ancient",
			Namespace: "eth/db/chaindata/",
			ReadOnly:  false,
		}
	}
}

type StateDBConfig struct {
	Cache     int
	Journal   string
	Preimages bool
}

func defaultStateDBConfig() *StateDBConfig {
	if runtime.GOOS == "darwin" { // MacOS
		return &StateDBConfig{
			Cache:     614,
			Journal:   "/Volumes/ETH_DATA/ethereum/geth/triecache",
			Preimages: false,
		}
	} else {
		return &StateDBConfig{
			Cache:     614,
			Journal:   "/experiment/ethereum/geth/triecache",
			Preimages: false,
		}
	}
}
