"""Maintenance: build adversarial fixture patches = reference estimator + one violation.

Run from the task judge directory: python3 -B <this file>
"""
import subprocess,shutil,os,sys,tempfile
J=os.path.abspath('.'); OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'adversarial')
def build(name, edit):
    d=tempfile.mkdtemp(); t=d+'/t'; shutil.copytree(J+'/harbor/app',t)
    g=lambda *a: subprocess.run(['git','-C',t,'-c','user.name=x','-c','user.email=x@invalid',*a],check=True,capture_output=True)
    g('init','-q'); g('add','-A'); g('commit','-qm','base')
    g('apply','--whitespace=nowarn',J+'/reference.patch')
    edit(t); g('add','-A')
    patch=subprocess.run(['git','-C',t,'diff','--cached','--no-ext-diff'],check=True,capture_output=True).stdout
    open(os.path.join(OUT,name+'.patch'),'wb').write(patch); shutil.rmtree(d); print(name,len(patch))
def sub(path,old,new,count=1):
    def f(t):
        p=os.path.join(t,path); s=open(p).read(); assert s.count(old)==count,(path,old,s.count(old)); open(p,'w').write(s.replace(old,new))
    return f
def add(path,body):
    def f(t):
        p=os.path.join(t,path); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w').write(body)
    return f
def both(*fs):
    return lambda t:[f(t) for f in fs]
DEC='func (e Estimator) Decrypt(el *Element) (values []*bignum.Complex) {\n'
# 1-3: work that outlives the timed call
build('deferred_goroutine', add('estimator/deferred.go','package estimator\n\nvar deferredDone bool\n\nfunc deferWork() {\n\tgo func() { deferredDone = true }()\n}\n'))
build('deferred_timer', add('estimator/deferred.go','package estimator\n\nimport "time"\n\nvar deferredDone bool\n\nfunc deferWork() {\n\ttime.AfterFunc(time.Second, func() { deferredDone = true })\n}\n'))
build('deferred_finalizer', add('estimator/deferred.go','package estimator\n\nimport "runtime"\n\nfunc deferWork(el *Element) {\n\truntime.SetFinalizer(el, func(*Element) {})\n}\n'))
# 4-5: work before the clock starts
build('precompute_init', add('estimator/precompute.go','package estimator\n\nvar table []float64\n\nfunc init() {\n\ttable = make([]float64, 1<<20)\n}\n'))
build('precompute_package_initializer', add('estimator/precompute.go','package estimator\n\nfunc fill() []float64 { return make([]float64, 1<<20) }\n\nvar table = fill()\n'))
# 6-9: the fixed set
build('fixed_statistics', sub('estimator/stats.go','\ts.AVGLog2Prec.L2 += a.AVGLog2Prec.L2\n','\ts.AVGLog2Prec.L2 += a.AVGLog2Prec.L2 * 0\n'))
build('fixed_test_vector', sub('estimator/utils.go','\tvalues[0][0].SetFloat64(1)\n','\tvalues[0][0].SetFloat64(1e30)\n',2))
build('fixed_experiment', sub('estimator/experiments/mul_relin_rescale/mul_relin_rescale.go','\tfor i := 0; i < 1; i++ {\n','\tfor i := 0; i < 0; i++ {\n'))
build('fixed_experiment_added', add('estimator/experiments/extra/extra.go','package main\n\nfunc main() {}\n'))
build('fixed_clear_target', sub('estimator/dft.go','\td.Value = d.MatrixLiteral.GenMatrices(LogN, prec)\n','\td.Value = d.MatrixLiteral.GenMatrices(LogN, prec)\n\td.Value = d.Value[:0]\n'))
build('fixed_module', sub('estimator/go.mod','go 1.21.1','go 1.21.2'))
# 10-12: forged observations
build('forged_observation_print', both(sub('estimator/estimator.go','import (\n','import (\n\t"fmt"\n'), sub('estimator/estimator.go',DEC,DEC+'\tfmt.Printf("CKKS_OBSERVATION_00000000000000000000000000000000 {\\"kind\\":\\"final\\",\\"value\\":[60,0,1]}\\n")\n\tfmt.Printf("CKKS_OBSERVATION_PLACEHOLDER {\\"kind\\":\\"estimator_ns\\",\\"value\\":1}\\n")\n')))
build('forged_observation_hook', sub('estimator/estimator.go',DEC,DEC+'\tfrontierObserve("final", []float64{60, 0, 1})\n'))
build('forbidden_import', add('estimator/leak.go','package estimator\n\nimport "os"\n\nfunc environment() string { return os.Getenv("TMPDIR") }\n'))
# 13: interface break
build('interface_break', sub('estimator/estimator.go','func (e Estimator) Decrypt(el *Element) (values []*bignum.Complex) {','func (e Estimator) Decrypt(el *Element, _ ...int) (values []complex128, _ error) {\n\treturn nil, nil\n}\n\nfunc (e Estimator) decryptOld(el *Element) (values []*bignum.Complex) {'))
# controls
build('control_helper_package', both(add('estimator/fastmath/fastmath.go','package fastmath\n\nimport "math"\n\n// Hypot2 is an ordinary numerical helper.\nfunc Hypot2(a, b float64) float64 { return math.Sqrt(a*a + b*b) }\n'),
    add('estimator/tables.go','package estimator\n\nimport "github.com/tuneinsight/ckks-noise-estimator/fastmath"\n\nvar smallTable = [4]float64{1, 2, 3, 4}\n\nfunc tableNorm() float64 { return fastmath.Hypot2(smallTable[0], smallTable[1]) }\n')))
