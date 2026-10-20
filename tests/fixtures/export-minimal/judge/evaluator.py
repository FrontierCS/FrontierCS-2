def evaluate(value):
    passed = float(value) > 42.0
    return {"beats_reference": int(passed), "beat_spec_workload_passes": [
        {"workload": "synthetic", "metrics": {"value": passed}}]}
