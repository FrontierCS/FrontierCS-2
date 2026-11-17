#!/bin/bash

go run ../main.go --indicator 0 --txNum 2000 --blockNum 1000 --ratio 0
go run ../main.go --indicator 0 --txNum 2000 --blockNum 1000 --ratio 0.4 --repair=true
go run ../main.go --indicator 0 --txNum 2000 --blockNum 1000 --ratio 0.8 --repair=true

go run ../main.go --indicator 0 --txNum 4000 --blockNum 1000 --ratio 0
go run ../main.go --indicator 0 --txNum 4000 --blockNum 1000 --ratio 0.4 --repair=true
go run ../main.go --indicator 0 --txNum 4000 --blockNum 1000 --ratio 0.8 --repair=true

go run ../main.go --indicator 0 --txNum 6000 --blockNum 1000 --ratio 0
go run ../main.go --indicator 0 --txNum 6000 --blockNum 1000 --ratio 0.4 --repair=true
go run ../main.go --indicator 0 --txNum 6000 --blockNum 1000 --ratio 0.8 --repair=true

go run ../main.go --indicator 0 --txNum 8000 --blockNum 1000 --ratio 0
go run ../main.go --indicator 0 --txNum 8000 --blockNum 1000 --ratio 0.4 --repair=true
go run ../main.go --indicator 0 --txNum 8000 --blockNum 1000 --ratio 0.8 --repair=true

go run ../main.go --indicator 0 --txNum 10000 --blockNum 1000 --ratio 0
go run ../main.go --indicator 0 --txNum 10000 --blockNum 1000 --ratio 0.4 --repair=true
go run ../main.go --indicator 0 --txNum 10000 --blockNum 1000 --ratio 0.8 --repair=true
