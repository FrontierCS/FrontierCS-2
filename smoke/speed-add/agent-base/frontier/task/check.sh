#!/bin/sh
set -eu

payload_dir="${1:-/workspace/payload}"
api_url="${FRONTIER_DRIVER_API_URL:?FRONTIER_DRIVER_API_URL is required}"
token_file="${FRONTIER_DRIVER_TOKEN_FILE:?FRONTIER_DRIVER_TOKEN_FILE is required}"

if [ ! -x "${payload_dir}/run.sh" ]; then
    echo "${payload_dir}/run.sh must exist and be executable" >&2
    exit 2
fi

archive="/workspace/train-payload.tar"
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
upload_key="speed-add-train-${digest#sha256:}"

upload_response="$(
    curl --fail-with-body --silent --show-error \
        -X POST "${api_url}/v1/payloads" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/x-tar" \
        -H "Content-Length: ${size}" \
        -H "X-Content-SHA256: ${digest}" \
        -H "X-Frontier-Purpose: train" \
        -H "Idempotency-Key: ${upload_key}" \
        --data-binary "@${archive}"
)"
payload_id="$(printf '%s' "${upload_response}" | jq -er '.payload_id')"
verification="$(
    curl --fail-with-body --silent --show-error \
        -X POST "${api_url}/v1/verifier-train" \
        -H "Authorization: Bearer ${token}" \
        -H "Content-Type: application/json" \
        -H "Idempotency-Key: speed-add-check-${payload_id}" \
        --data "{\"payload_id\":\"${payload_id}\"}"
)"
verification_id="$(printf '%s' "${verification}" | jq -er '.verification_id')"

remaining=190
while [ "${remaining}" -gt 0 ]; do
    verification="$(
        curl --fail-with-body --silent --show-error \
            -H "Authorization: Bearer ${token}" \
            "${api_url}/v1/verifications/${verification_id}"
    )"
    state="$(printf '%s' "${verification}" | jq -er '.state')"
    case "${state}" in
        passed)
            printf '%s\n' "${verification}"
            exit 0
            ;;
        failed)
            printf '%s\n' "${verification}" >&2
            exit 1
            ;;
        queued|running)
            ;;
        *)
            echo "unknown verification state: ${state}" >&2
            exit 1
            ;;
    esac
    remaining=$((remaining - 1))
    sleep 1
done

echo "verifier-train did not finish before the local polling deadline" >&2
exit 1
