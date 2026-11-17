#!/bin/bash

go run ../main.go --indicator 2 --ratio 0.2 --blockNum 1000 --repair=false --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.2 --blockNum 1000 --repair=true  --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.2 --blockNum 1000 --repair=true  --predictor=true  --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.2 --blockNum 1000 --repair=true  --predictor=true  --fast=true
sleep 2


go run ../main.go --indicator 2 --ratio 0.4 --blockNum 1000 --repair=false --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.4 --blockNum 1000 --repair=true  --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.4 --blockNum 1000 --repair=true  --predictor=true  --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.4 --blockNum 1000 --repair=true  --predictor=true  --fast=true
sleep 2


go run ../main.go --indicator 2 --ratio 0.6 --blockNum 1000 --repair=false --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.6 --blockNum 1000 --repair=true  --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.6 --blockNum 1000 --repair=true  --predictor=true  --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.6 --blockNum 1000 --repair=true  --predictor=true  --fast=true
sleep 2


go run ../main.go --indicator 2 --ratio 0.8 --blockNum 1000 --repair=false --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.8 --blockNum 1000 --repair=true  --predictor=false --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.8 --blockNum 1000 --repair=true  --predictor=true  --fast=false
sleep 2

go run ../main.go --indicator 2 --ratio 0.8 --blockNum 1000 --repair=true  --predictor=true  --fast=true
