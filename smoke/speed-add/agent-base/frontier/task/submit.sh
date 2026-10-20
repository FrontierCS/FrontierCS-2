#!/bin/sh
set -eu

payload_dir="${1:-/workspace/payload}"
api_url="${FRONTIER_DRIVER_API_URL:?FRONTIER_DRIVER_API_URL is required}"
token_file="${FRONTIER_DRIVER_TOKEN_FILE:?FRONTIER_DRIVER_TOKEN_FILE is required}"

if [ ! -x "${payload_dir}/run.sh" ]; then
    echo "${payload_dir}/run.sh must exist and be executable" >&2
    exit 2
fi

archive="/workspace/submission.tar"
tar -C "$(dirname "${payload_dir}")" \
    --format=ustar \
    --sort=name \
    --mtime='UTC 1970-01-01' \
    --owner=0 \
    --group=0 \
    -cf "${archive}" \
    "$(basename "${payload_dir}")"

size="$(wc -c < "${archive}" | tr -d ' ')"
digest="sha256:$(sha256sum "${archive}" | cut -d ' ' -f 1)"
token="$(cat "${token_file}")"
upload_key="speed-add-final-${digest#sha256:}"

upload_response="$(
    curl --fail-with-body --silent --show-error \
        -X POST "${api_url}/v1/payloads" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/x-tar" \
        -H "Content-Length: ${size}" \
        -H "X-Content-SHA256: ${digest}" \
        -H "X-Frontier-Purpose: final" \
        -H "Idempotency-Key: ${upload_key}" \
        --data-binary "@${archive}"
)"
payload_id="$(printf '%s' "${upload_response}" | jq -er '.payload_id')"

curl --fail-with-body --silent --show-error \
    -X POST "${api_url}/v1/submissions" \
    -H "Authorization: Bearer ${token}" \
    -H "Content-Type: application/json" \
    -H "Idempotency-Key: speed-add-submit-${payload_id}" \
    --data "{\"payload_id\":\"${payload_id}\"}"
printf '\n'

