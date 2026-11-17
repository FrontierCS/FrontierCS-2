package vm

const WARMUP_OBSERVATIONS = 1

type predictor struct {
	observationCount int
	lastPred         []int
}

func initPredictor() *predictor {
	return &predictor{
		lastPred: make([]int, 0),
	}
}

func (p *predictor) predict(train bool) int { return uncertain }

func (p *predictor) update(dir, predDir int) {
	p.observationCount++
	if p.observationCount > WARMUP_OBSERVATIONS {
		p.observationCount = WARMUP_OBSERVATIONS
	}
	if len(p.lastPred) > 1 {
		p.lastPred = p.lastPred[1:]
	} else {
		p.lastPred = p.lastPred[:0]
	}
}

func Bool2branchRes(dir bool) int {
	if dir {
		return taken
	}
	return notTaken
}

func BranchRes2bool(dir int) bool { return dir == taken }
