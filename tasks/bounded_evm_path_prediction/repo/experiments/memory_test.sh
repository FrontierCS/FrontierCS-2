#!/bin/bash

go run ../main.go --indicator 6 --blockNum 1000

sleep 2

go run ../main.go --indicator 7 --blockNum 1000 --repair=false --predictor=false --fast=false --checkpoint=false

sleep 2

go run ../main.go --indicator 7 --blockNum 1000 --repair=false --predictor=false --fast=false --checkpoint=true

sleep 2

go run ../main.go --indicator 7 --blockNum 1000 --repair=true --predictor=false --fast=false --checkpoint=true

sleep 2

go run ../main.go --indicator 7 --blockNum 1000 --repair=true --predictor=true --fast=false --checkpoint=true

sleep 2

go run ../main.go --indicator 7 --blockNum 1000 --repair=true --predictor=true --fast=true --checkpoint=true

sleep 2

go run ../main.go --indicator 7 --blockNum 1000 --repair=true --predictor=true --fast=true --checkpoint=true --fullStorage=true

sleep 2

go run ../main.go --indicator 9 --blockNum 1000

sleep 2

go run ../main.go --indicator 12 -blockNum 100 --threads 16

