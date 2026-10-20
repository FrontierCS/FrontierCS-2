package vm

import (
	"math"
)

const (
	HISTORY_LENGTH_MIN      = 10
	LHR_LENGTH              = HISTORY_LENGTH_MIN
	WT_LENGTH               = HISTORY_LENGTH_MIN + 1
	MAX_WT                  = 127
	MIN_WT                  = -128
	OBSERVER_MODE_THRESHOLD = 10
)

var THETA = math.Floor(1.93*LHR_LENGTH + 14)

type perceptron struct {
	historyLength int
	weights       []int
	// 加入一轮预执行的分支预测结果
	lhr []int
	// 缓存上次预测结果，加速下次更新过程
	lastPred []int
	// 是否启用观察者模式，即每隔#round检查一次看其是否可以使用perceptron预测
	enabled bool
	// 观察者模式的轮次
	round int
	// 记录连续uncertain的次数
	uncertainNum int
}

func initPerceptron() *perceptron {
	p := &perceptron{}
	p.weights = make([]int, WT_LENGTH)
	p.lhr = make([]int, LHR_LENGTH)

	// init lhr to Uncertain first
	for i := 0; i < LHR_LENGTH; i++ {
		p.lhr[i] = uncertain
	}
	p.historyLength = 0
	p.lastPred = make([]int, 0)
	p.enabled = false
	p.round = 0
	p.uncertainNum = 0
	return p
}

func (p *perceptron) run() int {
	prediction := 0
	prediction += p.weights[0]
	// n point to weight, i point to lastPre array
	n := LHR_LENGTH
	i := len(p.lastPred) - 1

	for i >= 0 && n > 0 {
		if p.lastPred[i] == uncertain {
			n--
			i--
			continue
		}
		if p.lastPred[i] == taken {
			prediction += p.weights[n]
		} else {
			prediction -= p.weights[n]
		}
		n--
		i--
	}
	// j point to lhr
	j := len(p.lhr) - 1
	for n > 0 {
		if p.lhr[j] == uncertain {
			n--
			i--
			continue
		}
		if p.lhr[j] == taken {
			prediction += p.weights[n]
		} else {
			prediction -= p.weights[n]
		}
		n--
		j--
	}
	return prediction
}

func (p *perceptron) predict(train bool) int {
	// train情况下，不需要走这个分支
	if !train && p.historyLength < HISTORY_LENGTH_MIN {
		//fmt.Printf("History length is less than %d\n", HISTORY_LENGTH_MIN)
		return uncertain
	}

	if p.enabled {
		p.round++
		if p.round%10 > 0 {
			return uncertain
		}
		prediction := p.run()
		if math.Abs(float64(prediction)) < THETA {
			return uncertain
		} else {
			p.enabled = false
			p.round = 0
			if prediction > 0 {
				return taken
			} else {
				return notTaken
			}
		}
	} else {
		prediction := p.run()
		if math.Abs(float64(prediction)) < THETA {
			p.uncertainNum++
			// 当连续#round轮预测的结果为uncertain时，切换至观察者模式
			if p.uncertainNum == OBSERVER_MODE_THRESHOLD {
				p.enabled = true
				p.uncertainNum = 0
			}
			return uncertain
		} else if prediction > 0 {
			p.uncertainNum = 0
			return taken
		} else {
			p.uncertainNum = 0
			return notTaken
		}
	}
}

func (p *perceptron) update(dir, predDir int) {
	p.historyLength += 1
	if predDir == uncertain || predDir != dir {
		if dir == taken {
			p.weights[0] += 1
		} else if dir == notTaken {
			p.weights[0] -= 1
		}
		for i := 0; i < LHR_LENGTH; i++ {
			// 不更改Uncertain对应的权重
			if p.lhr[i] == uncertain {
				continue
			}
			if dir == p.lhr[i] {
				if p.weights[i+1] < MAX_WT {
					p.weights[i+1] += 1
				}
			} else {
				if p.weights[i+1] > MIN_WT {
					p.weights[i+1] -= 1
				}
			}
		}
	}
	// 添加新历史数据
	p.lhr = append(p.lhr, dir)
	if p.historyLength > HISTORY_LENGTH_MIN {
		// 删除最早的历史
		p.historyLength -= 1
		p.lhr = p.lhr[1:]
	}
	// 清除更新过的lastPred
	if len(p.lastPred) > 1 {
		p.lastPred = p.lastPred[1:]
	} else {
		p.lastPred = make([]int, 0)
	}
}

func Bool2branchRes(dir bool) int {
	if dir {
		return taken
	}
	return notTaken
}

func BranchRes2bool(dir int) bool {
	if dir == taken {
		return true
	}
	return false
}
