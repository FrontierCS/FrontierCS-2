"""Run one original public IVF workload using the baked naive oracle."""
import json
from pathlib import Path
import sys

sys.path.insert(0, '/opt/frontier/development')
from ivfpq_worker import measure

config = json.loads(Path('development_config.json').read_text())
workloads = {w['id']: w for w in config['workloads']}
selected = sys.argv[1] if len(sys.argv) > 1 else next(iter(workloads))
if len(sys.argv) > 2 or selected not in workloads:
    raise SystemExit('Choose one public workload: ' + ', '.join(workloads))
result = measure(Path.cwd(), Path('/opt/frontier/development'), workloads[selected], config)
print(json.dumps(result, allow_nan=False, indent=2))
