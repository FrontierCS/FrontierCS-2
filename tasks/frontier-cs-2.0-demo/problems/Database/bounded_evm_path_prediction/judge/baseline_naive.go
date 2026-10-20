package vm

const (
	WARMUP_OBSERVATIONS     = 10
	LHR_LENGTH              = 10
	WT_LENGTH               = 11
	MAX_WT                  = 127
	MIN_WT                  = -128
	OBSERVER_MODE_THRESHOLD = 10
)

type predictor struct {
	observationCount int
	weights          []int
	lhr              []int
	lastPred         []int
	enabled          bool
	round            int
	uncertainNum     int
}

func initPredictor() *predictor              { return &predictor{} }
func (p *predictor) predict(train bool) int  { return taken }
func (p *predictor) update(dir, predDir int) {}
func Bool2branchRes(v bool) int {
	if v {
		return taken
	}
	return notTaken
}
func BranchRes2bool(v int) bool { return v == taken }
