package experiments

import (
	"bufio"
	"fmt"
	"github.com/shirou/gopsutil/v4/cpu"
	"github.com/shirou/gopsutil/v4/mem"
	"io/ioutil"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func GetCPUPercent() float64 {
	var percentSum float64
	percent, err := cpu.Percent(time.Second, true)
	if err != nil {
		log.Fatalln(err.Error())
		return -1
	}
	for _, p := range percent {
		percentSum += p
	}
	avg := percentSum / float64(len(percent))
	return avg
}

func GetCPUPercent2() float64 {
	var largest float64
	percent, err := cpu.Percent(time.Second, true)
	if err != nil {
		log.Fatalln(err.Error())
		return -1
	}
	for _, p := range percent {
		if p > largest {
			largest = p
		}
	}
	return largest
}

func GetMemPercent() float64 {
	memInfo, err := mem.VirtualMemory()
	if err != nil {
		log.Fatalln(err.Error())
		return -1
	}
	return memInfo.UsedPercent
}

func GetIOOperations() (float64, float64) {
	cmd := exec.Command("bash", "-c", "iostat -dx | grep 'nvme0n1' | awk '{print $2}'")
	read, err := cmd.Output()
	if err != nil {
		log.Fatalln("Error executing command:", err)
	}
	cmd = exec.Command("bash", "-c", "iostat -dx | grep 'nvme0n1' | awk '{print $8}'")
	write, err := cmd.Output()
	if err != nil {
		log.Fatalln("Error executing command:", err)
	}
	readString := string(read)
	writeString := string(write)
	readString = strings.TrimSpace(readString)
	writeString = strings.TrimSpace(writeString)
	readOp, _ := strconv.ParseFloat(readString, 64)
	writeOp, _ := strconv.ParseFloat(writeString, 64)
	return readOp, writeOp
}

func GetIOWait() float64 {
	cmd := exec.Command("bash", "-c", "iostat | grep -A 1 'avg-cpu' | tail -n 1 | awk '{print $4}'")
	wait, err := cmd.Output()
	if err != nil {
		log.Fatalln("Error executing command:", err)
	}
	waitString := string(wait)
	waitString = strings.TrimSpace(waitString)
	waitRatio, _ := strconv.ParseFloat(waitString, 64)
	return waitRatio
}

func GetIOStats() (readSysCalls, writeSysCalls int, err error) {
	// 读取/proc/self/io文件，获取当前进程的I/O统计
	data, err := ioutil.ReadFile("/proc/self/io")
	if err != nil {
		return 0, 0, err
	}

	// 将文件内容转换为字符串
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		// 找到以 syscr 和 syscw 开头的行
		if strings.HasPrefix(line, "syscr:") {
			fmt.Sscanf(line, "syscr: %d", &readSysCalls)
		} else if strings.HasPrefix(line, "syscw:") {
			fmt.Sscanf(line, "syscw: %d", &writeSysCalls)
		}
	}

	return readSysCalls, writeSysCalls, nil
}

func RunBpftrace(pid int, duration time.Duration) ([]int, []int, error) {
	// 构建bpftrace脚本
	bpftraceScript := `
	tracepoint:block:block_rq_issue /pid == ` + strconv.Itoa(pid) + `/ {
		if (strncmp(args->rwbs, "R", 1) == 0) {
			@reads = count();
		} else if (strncmp(args->rwbs, "W", 1) == 0) {
			@writes = count();
		}
	}
	interval:s:1 {
		print(@reads);
		print(@writes);
		clear(@reads);
		clear(@writes);
	}
	`
	// 创建bpftrace命令
	cmd := exec.Command("sudo", "bpftrace", "-e", bpftraceScript)

	// 获取bpftrace输出管道
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("error creating stdout pipe: %v", err)
	}

	// 启动bpftrace命令
	if err = cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("error starting bpftrace: %v", err)
	}

	// 使用切片收集reads和writes数据
	var readsList, writesList []int
	var currentReads, currentWrites int
	var gotReads, gotWrites bool
	var killed bool

	// 使用Scanner逐行读取bpftrace的输出
	scanner := bufio.NewScanner(stdout)
	done := make(chan struct{})

	// 启动一个goroutine处理bpftrace的输出
	go func() {
		for scanner.Scan() {
			line := scanner.Text()
			fmt.Println("Raw output:", line) // 打印原始输出

			// 解析 reads 或 writes 的值
			value, kind, err2 := parseReadsOrWrites(line)
			if err2 == nil {
				if kind == "reads" {
					currentReads = value
					gotReads = true
				} else if kind == "writes" {
					currentWrites = value
					gotWrites = true
				}

				// 如果同时得到了 reads 和 writes，记录并清空状态
				if gotReads && gotWrites {
					readsList = append(readsList, currentReads)
					writesList = append(writesList, currentWrites)
					gotReads = false
					gotWrites = false
				}
			}
		}
		done <- struct{}{}
	}()

	// 运行指定的持续时间
	select {
	case <-time.After(duration):
		// 时间结束，停止 bpftrace 进程
		if err = cmd.Process.Kill(); err != nil {
			return nil, nil, fmt.Errorf("failed to kill bpftrace process: %v", err)
		}
		killed = true
	case <-done:
	}

	// 等待bpftrace进程结束
	if !killed {
		if err = cmd.Wait(); err != nil {
			return nil, nil, fmt.Errorf("error waiting for bpftrace to finish: %v", err)
		}
	}

	return readsList, writesList, nil
}

func parseReadsOrWrites(line string) (int, string, error) {
	if strings.Contains(line, "@reads:") {
		readsStr := strings.TrimSpace(strings.Split(line, ":")[1])
		reads, err := strconv.Atoi(readsStr)
		if err != nil {
			return 0, "", fmt.Errorf("failed to parse reads from line: %s", line)
		}
		return reads, "reads", nil
	} else if strings.Contains(line, "@writes:") {
		writesStr := strings.TrimSpace(strings.Split(line, ":")[1])
		writes, err := strconv.Atoi(writesStr)
		if err != nil {
			return 0, "", fmt.Errorf("failed to parse writes from line: %s", line)
		}
		return writes, "writes", nil
	}
	return 0, "", fmt.Errorf("failed to parse reads/writes from line: %s", line)
}

func CalculateAverage(readsList, writesList []int) (float64, float64) {
	var sumReads, sumWrites int
	var rCount, wCount int
	var averageReads, averageWrites float64

	for i := 0; i < len(readsList); i++ {
		if readsList[i] != 0 {
			sumReads += readsList[i]
			rCount++
		}
	}

	for j := 0; j < len(writesList); j++ {
		if writesList[j] != 0 {
			sumWrites += writesList[j]
			wCount++
		}
	}

	if rCount == 0 {
		averageReads = 0
	} else {
		averageReads = float64(sumReads) / float64(rCount)
	}

	if wCount == 0 {
		averageWrites = 0
	} else {
		averageWrites = float64(sumWrites) / float64(wCount)
	}

	return averageReads, averageWrites
}
