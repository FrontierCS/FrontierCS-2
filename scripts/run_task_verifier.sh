#!/bin/sh
# Run a task's Verifier image on one payload the way the platform wrapper does: offline, as
# uid 65532, with the five FRONTIER_* variables and a writable output directory.
#
#   scripts/run_task_verifier.sh IMAGE PAYLOAD.tar DATASET MODE OUTDIR [CPUS] [MEMORY]
#
# MODE is train or eval; DATASET is the task's datasets/<mode> file. Writes OUTDIR/result.json,
# OUTDIR/evidence.tar.zst, OUTDIR/stdout.log and OUTDIR/exit.
set -u
[ $# -ge 5 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
image=$1 payload=$2 dataset=$3 mode=$4 out=$(realpath -m "$5") cpus=${6:-2} mem=${7:-4g}
mkdir -p "$out/output" && chmod 0777 "$out/output"
input=$(mktemp -d) && cp "$payload" "$input/payload.tar" && cp "$dataset" "$input/dataset" \
  && chmod -R a+rX "$input"
docker run --rm --network none --cpus "$cpus" --memory "$mem" --user 65532:65532 \
  -v "$input:/frontier/input:ro" -v "$out/output:/frontier/output" \
  -e FRONTIER_PAYLOAD_PATH=/frontier/input/payload.tar \
  -e FRONTIER_DATASET_PATH=/frontier/input/dataset \
  -e FRONTIER_VERIFICATION_MODE="$mode" \
  -e FRONTIER_RESULT_PATH=/frontier/output/result.json \
  -e FRONTIER_EVIDENCE_PATH=/frontier/output/evidence.tar.zst \
  "$image" /opt/frontier/verifier > "$out/stdout.log" 2>&1
echo $? > "$out/exit"
cp "$out/output/result.json" "$out/result.json" 2>/dev/null || echo '{}' > "$out/result.json"
cp "$out/output/evidence.tar.zst" "$out/evidence.tar.zst" 2>/dev/null || true
rm -rf "$input"
cat "$out/result.json"; echo
