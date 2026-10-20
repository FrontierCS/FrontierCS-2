# Native publication on the local Kubernetes profile

`native-publish` adapts a frozen native publication to Preview's local object
store without importing it into the public task catalog. It preserves the
TaskVersion ID, full TaskConfig, definition digest, image identities, source and
build facts. Its local receipt uses `local-sha256:…` content versions; the bundle
and final envelope retain the original S3 VersionIds and exact GET readbacks.
It does not create evaluations or runs.

The input is the original native create-task-version JSON with `driver_config`
and `author_receipt`. Both train and eval must have content-addressed native
dataset keys, real S3 VersionIds, positive sizes and matching SHA-256 hashes.
Both published key shapes are supported:
`datasets/frontier-cs-2.0/<task-id>/<train|eval>/sha256/<hash>.<json|tar>`
and the legacy `datasets/frontier-cs-2.0/<task-id>/sha256/<hash>.<json|tar>`.
Task IDs may contain hyphens or underscores. Each key must bind the exact task
and content hash; when a mode segment exists it must match the config/receipt
mode. Legacy keys take their mode from that exact pair, and train/eval may not
alias the same mounted destination. Original URIs are preserved verbatim.
Registration also requires the definition JSON to exist at its original S3
`definition_ref`. A reference used only as a label is insufficient: the command
fails closed if it cannot fetch and verify those bytes.

All examples use a shell array to keep the config argument before the stage:

```bash
native=(./python scripts/frontier_deploy.py native-publish --config config/site.json)
"${native[@]}" plan --document frozen.json
"${native[@]}" prepare --document frozen.json --output native-bundle
"${native[@]}" render-seed --document frozen.json --bundle native-bundle --output native-seed.json
"${native[@]}" render-inspector --document frozen.json --bundle native-bundle --output native-inspector.json
```

`plan` and both render commands are offline. `prepare` uses the existing AWS CLI
credential chain for exact-version `s3api get-object` requests; it does not write
to S3. The output directory must not already exist. Its standard OCI layout has
the reference name `datasets`, together with the original document and custody
provenance. Every subsequent stage verifies all manifest, config, layer and
payload hashes and the complete original receipt and S3 readbacks.

Publish the transport layout through the site's approved registry publication
procedure, retaining the manifest digest. The exact destination is
`<network.registry_cluster_ip>:5000/frontier/tasks/<task-id-with-hyphens>/dataset`;
the seed manifest pins its digest. With an authorized registry connection, a
standard OCI publisher can use:

```bash
skopeo copy --preserve-digests oci:native-bundle:datasets \
  docker://REGISTRY/frontier/tasks/TASK-SLUG/dataset:TRANSPORT-TAG
```

Use a unique transport tag and the site's existing registry transport settings.
Do not retag or rebuild the native agent or verifier images. Their original
references must already be reachable by the Driver.

Creating the seed and inspector workloads and registering a TaskVersion are
operator actions under the existing [live site change procedure](operations.md#live-site-changes):
use an approved window, backup and site checks. The commands below are an
operational procedure, not actions performed by rendering the files:

```bash
kubectl --kubeconfig .state/SITE/kubeconfig apply -f native-seed.json
# Inspect the completed seed Job before proceeding.
kubectl --kubeconfig .state/SITE/kubeconfig -n frontier-system get jobs
kubectl --kubeconfig .state/SITE/kubeconfig create -f native-inspector.json -o json > inspector-created.json
inspector=$(./python -c 'import json; print(json.load(open("inspector-created.json"))["metadata"]["name"])')
kubectl --kubeconfig .state/SITE/kubeconfig -n frontier-system get pod "$inspector"
"${native[@]}" finalize --document frozen.json --bundle native-bundle \
  --pvc-pod "$inspector" --output native-finalized.json
"${native[@]}" register --check --document frozen.json --bundle native-bundle \
  --pvc-pod "$inspector"
"${native[@]}" register --apply --document frozen.json --bundle native-bundle \
  --pvc-pod "$inspector" --output native-registration.json
./python - <<'PYDELETE' > inspector-delete.json
import json
created = json.load(open("inspector-created.json"))
print(json.dumps({"apiVersion": "v1", "kind": "DeleteOptions",
                  "preconditions": {"uid": created["metadata"]["uid"]}}))
PYDELETE
kubectl --kubeconfig .state/SITE/kubeconfig delete \
  --raw "/api/v1/namespaces/frontier-system/pods/$inspector" -f inspector-delete.json
```

The inspector name binds the task and frozen-document hash. Use `create`, which
rejects an existing name, and retain its returned UID; cleanup uses that UID as
an atomic deletion precondition so it cannot delete another operator's Pod.
The inspector uses the site's pinned storage-controller runtime, no service
account token, a one-hour deadline and a read-only `frontier-object-seed` PVC.
The CLI checks the actual pod's image, mount and UID, executes the byte checker
inside that mount, then checks pod identity again. Every path component is opened
relative to a directory descriptor with `O_NOFOLLOW`; leaf files also use
`O_NONBLOCK` so FIFOs cannot hang the checker. Only stable regular files with the
expected size and hash pass. Pod logs, seed success and supplied receipts cannot
replace this read. If running on a host that already has the actual dedicated
read-only PVC mount, `--mounted-root /actual/object/root` is an alternative.

Both registration modes repeat bundle and mounted-byte verification.
`--check` returns `MATCH` or `MISSING` and does not POST. `--apply` uses the
existing official version helper, never overwrites a conflicting version, and
requires an output receipt. Management's public API omits the full TaskConfig;
the CLI fetches the actual S3 definition reference returned by Management,
verifies its canonical digest against that API record, and compares the entire
configuration, including exact images, dataset URIs/hashes and allowed domains.
It also compares TaskVersion/task identities, definition reference and publication
metadata. Missing artifacts or mismatches are errors. Management independently
verifies the mounted dataset bytes when accepting the local author receipt.
