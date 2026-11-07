#!/bin/bash

go run ../main.go --indicator 4 --txNum 2000 --blockNum 1000 --threads 4
go run ../main.go --indicator 4 --txNum 4000 --blockNum 1000 --threads 4
go run ../main.go --indicator 4 --txNum 6000 --blockNum 1000 --threads 4
go run ../main.go --indicator 4 --txNum 8000 --blockNum 1000 --threads 4
go run ../main.go --indicator 4 --txNum 10000 --blockNum 1000 --threads 4
