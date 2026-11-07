"""Approved LogN12/Table 1 configuration successor, without numerical kernels.

The nine predecessor rows keep their names and exact main bytes. Added rows
change only initial scale, matching rescaling-prime bit lengths, and secret
distribution. VII/VIII never acquire a scale55 row. Runtime qualification and
fresh reference calibration are separate from this finite source definition.
"""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASES = json.loads((ROOT/'judge/configuration_bases.json').read_text())
ORIGINAL_FINAL = ('mul_relin_rescale', 'power_basis', 'poly_eval', 'goldschmidt',
                  'coeffs_to_slots', 'slots_to_coeffs', 'iterative_step', 'mod1', 'bootstrapping')
ORIGINAL_DEV = ORIGINAL_FINAL[:6]
RELEASE_ID = 'composition-scaled-logn12-configurations-v2'
SECRETS = ('dense', 'h192')


def require(value, message):
    if not value:
        raise ValueError(message)


def configuration(circuit, scale, secret):
    require(circuit in ORIGINAL_FINAL, 'unknown circuit')
    base = BASES[circuit]
    scales = (45,) if circuit in ('mul_relin_rescale', 'mod1', 'bootstrapping') else (45, 55)
    require(type(scale) is int and scale in scales and secret in SECRETS,
            'unsupported scale/secret or VII/VIII scale exclusion')
    require(circuit != 'mul_relin_rescale' or secret == base['secret'],
            'primitive anchor must remain unchanged')
    original = (scale, secret) == (base['log_scale'], base['secret'])
    # Preserve Q0, depth and P. Other Q primes track the initial scale only in
    # sets I-VI; EvalMod's internal60/53-bit chain and bootstrap stay unchanged.
    q = list(base['log_q'])
    if scale != base['log_scale']:
        require(all(v == base['log_scale'] for v in q[1:]), 'unknown rescaling chain')
        q = [q[0], *([scale]*(len(q)-1))]
    return {'id': circuit if original else f'{circuit}__logn12_scale{scale}_{secret}',
            'circuit': circuit, 'paper_set': base['paper_set'], 'log_n': 12,
            'log_scale': scale, 'secret': secret, 'log_q': q, 'log_p': base['log_p'],
            'original_row': original, 'source_invocations': 1,
            'main_path': f'estimator/experiments/{circuit}/{circuit}.go',
            'predecessor_main_sha256': base['main_sha256']}


def configurations():
    rows = [configuration(c, BASES[c]['log_scale'], BASES[c]['secret']) for c in ORIGINAL_FINAL]
    for circuit in ORIGINAL_FINAL[1:]:
        for scale in ((45,) if circuit in ('mod1', 'bootstrapping') else (45, 55)):
            for secret in SECRETS:
                row = configuration(circuit, scale, secret)
                if not row['original_row']:
                    rows.append(row)
    return rows


def manifests():
    rows = configurations()
    return ([r['id'] for r in rows if r['circuit'] in ORIGINAL_DEV], [r['id'] for r in rows])


def validate_manifest(document, release):
    development, final = manifests()
    require(document.get('schema') == 'ckks-workload-configurations/v1' and
            document.get('release') == RELEASE_ID and document.get('rows') == configurations(),
            'configuration membership, order or parameters differ')
    require(release['release'] == RELEASE_ID and release['development'] == development and
            release['final'] == final, 'release workload membership/order differs')
    require(len(final) == len(set(final)) == 29 and len(development) == 21,
            'configuration counts differ')
    return {row['id']: row for row in document['rows']}


def configured_main(body, row):
    """Derive a trusted Go overlay; no edits to protected/submitted source files."""
    require(len(re.findall(r'\bLogN\s*:=\s*12\b', body)) == 1, 'LogN12 binding missing')
    if row['original_row']:
        return body
    base = BASES[row['circuit']]
    body, n = re.subn(r'\bLogScale := '+str(base['log_scale'])+r'\b',
                     'LogScale := '+str(row['log_scale']), body)
    require(n == 1, 'initial scale literal not unique')
    body, n = re.subn(r'LogQ:\s*\[\]int\{[^}]+\}',
                     'LogQ: []int{'+', '.join(map(str, row['log_q']))+'}', body)
    require(n == 1, 'modulus chain literal not unique')
    secret = 'ring.Ternary{H: 192}' if row['secret'] == 'h192' else 'ring.Ternary{P: 2.0/3.0}'
    body, n = re.subn(r'(?m)^\s*(?://)?Xs:\s*ring.Ternary\{H:\s*192\},',
                     '\n\t\tXs: '+secret+',', body)
    require(n == 1, 'secret literal not unique')
    import_path = '"github.com/tuneinsight/lattigo/v6/ring"'
    if import_path not in body:
        require(body.count('import (') == 1, 'unexpected import block')
        body = body.replace('import (', 'import (\n\t'+import_path, 1)
    return body


# Statements of a fixed main that can reach the estimator package. Each is
# bracketed by the trusted clock; the clear target, key generation, the
# encrypted circuit and the statistics of the encrypted result are not.
ESTIMATOR_STATEMENT = re.compile(
    r'\best\.|\bestParamsResidual\.|\bevalEst\.|\bbootEst\.NewEvaluator\b|\bestimator\.NewEstimator\b'
    r'|\bpbEl\.|\bestimator\.NewPowerBasis\b|, est\)|\bNewTestVector\w*\(est\b')
LAST_ESTIMATOR_OUTPUT = re.compile(r'\bDecrypt\(')
OBSERVATION_PLACEHOLDER = 'CKKS_OBSERVATION_PLACEHOLDER '
TIMED_MAIN_SHA256 = json.loads((ROOT/'judge/timed_main_sha256.json').read_text())


def timed_main(body, row):
    """Derive the measured main: statements are inserted, no source byte changes.

    The derived main owns every reported line. Nothing in package estimator can
    name the observation prefix, the clock or the reporting function.
    """
    lines = body.split('\n'); out = []; index = 0; inside = False; timed = 0
    require(body.count('import (') == 1 and body.count('\nfunc main() {\n') == 1 and
            body.count('\tstatsHave.Finalize()\n') == 1 and 'frontier' not in body,
            'experiment main shape differs')
    parameters = 'btpParams' if row['circuit'] == 'bootstrapping' else 'params'
    while index < len(lines):
        line = lines[index]
        if line.startswith('import ('):
            out += [line, '\tfrontierjson "encoding/json"', '\tfrontierruntime "runtime"',
                    '\tfrontiertime "time"']
            index += 1; continue
        if line == 'func main() {':
            inside = True
            out += [line,
                    '\tvar frontierEstimator frontiertime.Duration',
                    '\tvar frontierStart frontiertime.Time',
                    '\tfrontierReport := func(kind string, value interface{}) {',
                    '\t\tb, err := frontierjson.Marshal(map[string]interface{}{"kind": kind, "value": value})',
                    '\t\tif err != nil {', '\t\t\tpanic(err)', '\t\t}',
                    '\t\tfmt.Printf("' + OBSERVATION_PLACEHOLDER + '%s\\n", b)', '\t}']
            index += 1; continue
        if inside and line == '}':
            inside = False
        stripped = line.strip()
        if inside and ESTIMATOR_STATEMENT.search(line) and not stripped.startswith('//'):
            indent = line[:len(line)-len(line.lstrip())]; end = index
            require(not stripped.startswith(('for ', '} else', 'else', 'func ', 'return')),
                    'estimator statement shape differs')
            if line.rstrip().endswith('{'):
                end = index+1
                while end < len(lines) and lines[end] != indent+'}':
                    require(lines[end].startswith(indent+'\t') or not lines[end].strip(),
                            'estimator block shape differs')
                    end += 1
                require(end < len(lines), 'estimator block is not closed')
            block = lines[index:end+1]
            out.append(indent+'frontierStart = frontiertime.Now()')
            out += block
            if any(LAST_ESTIMATOR_OUTPUT.search(text) for text in block):
                # The estimator's own garbage is collected on its own clock.
                out.append(indent+'frontierruntime.GC()')
            out.append(indent+'frontierEstimator += frontiertime.Since(frontierStart)')
            timed += 1; index = end+1; continue
        out.append(line)
        if inside and line == '\tstatsHave.Finalize()':
            out += ['\tfrontierReport("parameters", '+parameters+')',
                    '\tfrontierReport("final", []float64{statsWant.AVGLog2Prec.L2, statsWant.STDLog2Prec.L2, statsWant.N})',
                    '\tfrontierReport("final", []float64{statsHave.AVGLog2Prec.L2, statsHave.STDLog2Prec.L2, statsHave.N})',
                    '\tfrontierReport("estimator_ns", frontierEstimator.Nanoseconds())']
        index += 1
    require(timed >= 3 and any('frontierruntime.GC()' in text for text in out), 'no timed estimator statements')
    derived = '\n'.join(out)
    binding = main_source_binding_kind(body, row)
    require(hashlib.sha256(derived.encode()).hexdigest() == TIMED_MAIN_SHA256.get(row['id'], {}).get(binding),
            'derived measured main differs from its frozen form: '+row['id'])
    return derived


def main_source_binding_kind(configured, row):
    """authored or canonical_restored, from the comment block the private restoration appends
    to the reference's mains. Each kind has its own frozen derived main."""
    return 'canonical_restored' if configured.rstrip().endswith('*/') else 'authored'


def measured_main(body, row, prefix):
    """Frozen derived main with this run's unpredictable observation prefix."""
    require(re.fullmatch(r'CKKS_OBSERVATION_[0-9a-f]{32} ', prefix) is not None, 'invalid observation prefix')
    derived = timed_main(configured_main(body, row), row)
    require(derived.count(OBSERVATION_PLACEHOLDER) == 1, 'observation placeholder is not unique')
    return derived.replace(OBSERVATION_PLACEHOLDER, prefix)


def prepare_overlay(tree, dest, prefix, row):
    """Overlay only the experiment main; no file of package estimator is replaced."""
    source = Path(tree)/row['main_path']
    original = source.read_text()
    derived = measured_main(original, row, prefix)
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    main = dest/(row['circuit']+'.go'); main.write_text(derived)
    path = dest/'overlay.json'
    path.write_text(json.dumps({'Replace': {str(source): str(main)}}, sort_keys=True))
    return path, {'configuration': row, 'frozen_measured_main_sha256': hashlib.sha256(derived.replace(prefix, OBSERVATION_PLACEHOLDER).encode()).hexdigest(),
                  'input_main_binding': {'kind': main_source_binding_kind(original, row),
                                         'sha256': hashlib.sha256(original.encode()).hexdigest()},
                  'instrumentation': 'trusted derived main: estimator clock and all reporting; package estimator not overlaid'}


def verify_parameters(observations, row):
    """Check the actual Lattigo literal emitted by the retained instrumentation.

    Bootstrapping reports its residual literal separately from its larger
    internal chain. The expected input chain applies to that residual literal.
    Dense means P=2/3 sampling, not a falsely exact fixed weight floor(2N/3).
    """
    require(isinstance(observations, list) and observations, 'missing resolved parameters')
    if row['circuit'] == 'bootstrapping':
        literals = [p['ResidualParameters'] for p in observations if 'ResidualParameters' in p]
    else:
        literals = [p for p in observations if 'LogN' in p]
    require(len(literals) == 1, 'ambiguous resolved input parameters')
    p = literals[0]
    require(type(p.get('LogN')) is int and p['LogN'] == 12 and
            type(p.get('LogDefaultScale')) is int and p['LogDefaultScale'] == row['log_scale'],
            'resolved ring/initial scale differs')
    secret = p.get('Xs', {})
    require(secret.get('Type') == 'Ternary', 'resolved secret distribution differs')
    if row['secret'] == 'h192':
        require(type(secret.get('H')) is int and secret['H'] == 192 and not secret.get('P', 0),
                'resolved H192 secret differs')
    else:
        require(type(secret.get('P')) in (int, float) and secret['P'] == 2/3 and not secret.get('H', 0),
                'resolved dense secret differs')
    for key, expected in (('Q', row['log_q']), ('P', row['log_p'])):
        primes = p.get(key, [])
        require(len(primes) == len(expected) and all(type(q) is int and q > 0 for q in primes),
                'resolved modulus chain missing')
        require([round(math.log2(q)) for q in primes] == expected, 'resolved modulus sizes differ')
    return True
