use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use tempfile::TempDir;

const DATASET_SCHEMA: &str = "frontier.speed-add-dataset/v1";
const MAX_DATASET_BYTES: u64 = 1024 * 1024;
const MAX_PAYLOAD_FILES: usize = 64;
const MAX_PAYLOAD_BYTES: u64 = 8 * 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 4096;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    schema_version: String,
    mode: Mode,
    correctness: Vec<Case>,
    benchmarks: Vec<Benchmark>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Train,
    Eval,
}

impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "train" => Ok(Self::Train),
            "eval" => Ok(Self::Eval),
            _ => bail!("verification mode must be train or eval"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    a: u64,
    b: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Benchmark {
    a: u64,
    b: u64,
    iterations: u32,
    timeout_ms: u64,
    reference_runtime_ns: u64,
}

#[derive(Debug, Serialize)]
struct TaskResult {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_score: Option<f64>,
    summary: String,
    metrics: BTreeMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure: Option<Failure>,
}

#[derive(Debug, Serialize)]
struct Failure {
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Debug, Serialize)]
struct Evidence {
    schema_version: &'static str,
    mode: Mode,
    dataset_sha256: String,
    payload_sha256: String,
    correctness_cases: usize,
    benchmark_cases: usize,
    benchmark_iterations: u64,
    measured_median_ns: Vec<u64>,
    reference_runtime_ns: Vec<u64>,
    raw_score: Option<f64>,
    outcome: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure: Option<String>,
}

#[derive(Debug)]
struct Invocation {
    elapsed: Duration,
    stdout: String,
    stderr: String,
    success: bool,
    timed_out: bool,
}

fn main() {
    if let Err(error) = run_from_environment() {
        eprintln!("speed-add verifier error: {error:#}");
        if let Ok(result_path) = required_absolute_path("FRONTIER_RESULT_PATH") {
            let result = failed_result("verifier_internal_error", format!("{error:#}"));
            if write_json(&result_path, &result).is_ok() {
                return;
            }
        }
        std::process::exit(1);
    }
}

fn run_from_environment() -> Result<()> {
    let payload_path = required_absolute_path("FRONTIER_PAYLOAD_PATH")?;
    let dataset_path = required_absolute_path("FRONTIER_DATASET_PATH")?;
    let result_path = required_absolute_path("FRONTIER_RESULT_PATH")?;
    let evidence_path = required_absolute_path("FRONTIER_EVIDENCE_PATH")?;
    let mode = Mode::parse(
        &env::var("FRONTIER_VERIFICATION_MODE")
            .context("FRONTIER_VERIFICATION_MODE is required")?,
    )?;

    let dataset_bytes = read_limited(&dataset_path, MAX_DATASET_BYTES)?;
    let dataset: Dataset =
        serde_json::from_slice(&dataset_bytes).context("parse speed-add dataset")?;
    validate_dataset(&dataset, mode)?;
    let payload_digest = hash_file(&payload_path)?;
    let dataset_digest = format!("sha256:{:x}", Sha256::digest(&dataset_bytes));

    let workspace_root = mounted_workspace_root(&payload_path)?;
    let workspace = tempfile::tempdir_in(workspace_root)
        .context("create verifier workspace on the mounted work volume")?;
    extract_payload(&payload_path, workspace.path())?;
    let entrypoint = workspace.path().join("payload/run.sh");
    validate_entrypoint(&entrypoint)?;

    let outcome = evaluate(&entrypoint, workspace.path(), &dataset);
    let (result, evidence) = match outcome {
        Ok((raw_score, medians, iterations)) => {
            let metrics = BTreeMap::from([
                ("benchmark_cases".into(), json!(dataset.benchmarks.len())),
                ("benchmark_iterations".into(), json!(iterations)),
                ("correctness_cases".into(), json!(dataset.correctness.len())),
                ("measured_median_ns".into(), json!(medians)),
            ]);
            (
                TaskResult {
                    status: "passed",
                    raw_score: Some(raw_score),
                    summary: format!(
                        "all {} correctness cases passed; raw performance score {:.6}",
                        dataset.correctness.len(),
                        raw_score
                    ),
                    metrics,
                    failure: None,
                },
                Evidence {
                    schema_version: "frontier.speed-add-evidence/v1",
                    mode,
                    dataset_sha256: dataset_digest,
                    payload_sha256: payload_digest,
                    correctness_cases: dataset.correctness.len(),
                    benchmark_cases: dataset.benchmarks.len(),
                    benchmark_iterations: iterations,
                    measured_median_ns: medians,
                    reference_runtime_ns: dataset
                        .benchmarks
                        .iter()
                        .map(|case| case.reference_runtime_ns)
                        .collect(),
                    raw_score: Some(raw_score),
                    outcome: "passed",
                    failure: None,
                },
            )
        }
        Err(error) => (
            failed_result("candidate_failed", error.to_string()),
            Evidence {
                schema_version: "frontier.speed-add-evidence/v1",
                mode,
                dataset_sha256: dataset_digest,
                payload_sha256: payload_digest,
                correctness_cases: dataset.correctness.len(),
                benchmark_cases: dataset.benchmarks.len(),
                benchmark_iterations: 0,
                measured_median_ns: Vec::new(),
                reference_runtime_ns: dataset
                    .benchmarks
                    .iter()
                    .map(|case| case.reference_runtime_ns)
                    .collect(),
                raw_score: None,
                outcome: "failed",
                failure: Some(error.to_string()),
            },
        ),
    };

    write_evidence(&evidence_path, &evidence)?;
    write_json(&result_path, &result)?;
    Ok(())
}

fn mounted_workspace_root(payload_path: &Path) -> Result<&Path> {
    payload_path
        .parent()
        .and_then(Path::parent)
        .context("payload path must be below a mounted verifier workspace")
}

fn validate_dataset(dataset: &Dataset, expected_mode: Mode) -> Result<()> {
    ensure!(
        dataset.schema_version == DATASET_SCHEMA,
        "unsupported dataset schema"
    );
    ensure!(
        dataset.mode == expected_mode,
        "dataset mode does not match verification mode"
    );
    ensure!(
        !dataset.correctness.is_empty() && dataset.correctness.len() <= 128,
        "correctness case count must be in 1..=128"
    );
    ensure!(
        !dataset.benchmarks.is_empty() && dataset.benchmarks.len() <= 16,
        "benchmark case count must be in 1..=16"
    );
    for case in &dataset.correctness {
        case.a
            .checked_add(case.b)
            .context("correctness case sum overflows u64")?;
    }
    for benchmark in &dataset.benchmarks {
        benchmark
            .a
            .checked_add(benchmark.b)
            .context("benchmark sum overflows u64")?;
        ensure!(
            (1..=20).contains(&benchmark.iterations),
            "benchmark iterations must be in 1..=20"
        );
        ensure!(
            (1..=30_000).contains(&benchmark.timeout_ms),
            "benchmark timeout_ms must be in 1..=30000"
        );
        ensure!(
            benchmark.reference_runtime_ns > 0,
            "reference runtime must be positive"
        );
    }
    Ok(())
}

fn extract_payload(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = File::open(archive_path).context("open payload tar")?;
    let mut archive = tar::Archive::new(file);
    let mut count = 0_usize;
    let mut bytes = 0_u64;
    for entry in archive.entries().context("read payload tar")? {
        let mut entry = entry.context("read payload entry")?;
        let path = entry.path().context("read payload path")?.into_owned();
        validate_payload_path(&path)?;
        let entry_type = entry.header().entry_type();
        ensure!(
            entry_type.is_file() || entry_type.is_dir(),
            "payload may contain only regular files and directories"
        );
        count = count
            .checked_add(1)
            .context("payload file count overflow")?;
        ensure!(
            count <= MAX_PAYLOAD_FILES,
            "payload contains too many entries"
        );
        if entry_type.is_file() {
            bytes = bytes
                .checked_add(entry.size())
                .context("payload size overflow")?;
            ensure!(
                bytes <= MAX_PAYLOAD_BYTES,
                "payload uncompressed size exceeds limit"
            );
        }
        ensure!(
            entry
                .unpack_in(destination)
                .context("extract payload entry")?,
            "payload path escapes extraction root"
        );
    }
    ensure!(count > 0, "payload tar is empty");
    Ok(())
}

fn validate_payload_path(path: &Path) -> Result<()> {
    ensure!(!path.is_absolute(), "payload path must be relative");
    let mut components = path.components();
    ensure!(
        components.next() == Some(Component::Normal("payload".as_ref())),
        "every archive entry must be below payload/"
    );
    ensure!(
        components.all(|part| matches!(part, Component::Normal(_))),
        "payload path contains a forbidden component"
    );
    Ok(())
}

fn validate_entrypoint(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path).context("payload/run.sh is required")?;
    ensure!(metadata.is_file(), "payload/run.sh must be a regular file");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        ensure!(
            metadata.permissions().mode() & 0o111 != 0,
            "payload/run.sh must be executable"
        );
    }
    Ok(())
}

fn evaluate(
    entrypoint: &Path,
    workspace: &Path,
    dataset: &Dataset,
) -> Result<(f64, Vec<u64>, u64)> {
    for case in &dataset.correctness {
        let invocation = invoke(entrypoint, workspace, case, Duration::from_millis(2_000))?;
        validate_invocation(&invocation, case)?;
    }

    let mut medians = Vec::with_capacity(dataset.benchmarks.len());
    let mut total_measured_ns = 0_u128;
    let mut total_reference_ns = 0_u128;
    let mut total_iterations = 0_u64;
    for benchmark in &dataset.benchmarks {
        let case = Case {
            a: benchmark.a,
            b: benchmark.b,
        };
        let mut timings = Vec::with_capacity(benchmark.iterations as usize);
        for _ in 0..benchmark.iterations {
            let invocation = invoke(
                entrypoint,
                workspace,
                &case,
                Duration::from_millis(benchmark.timeout_ms),
            )?;
            validate_invocation(&invocation, &case)?;
            let elapsed_ns = u64::try_from(invocation.elapsed.as_nanos())
                .unwrap_or(u64::MAX)
                .max(1);
            timings.push(elapsed_ns);
        }
        timings.sort_unstable();
        let median = timings[timings.len() / 2];
        medians.push(median);
        total_measured_ns = total_measured_ns
            .checked_add(u128::from(median))
            .context("measured runtime overflow")?;
        total_reference_ns = total_reference_ns
            .checked_add(u128::from(benchmark.reference_runtime_ns))
            .context("reference runtime overflow")?;
        total_iterations += u64::from(benchmark.iterations);
    }

    ensure!(total_measured_ns > 0, "measured runtime must be positive");
    let score = total_reference_ns as f64 / total_measured_ns as f64;
    ensure!(
        score.is_finite() && score >= 0.0,
        "computed raw score is invalid"
    );
    Ok((score, medians, total_iterations))
}

fn invoke(
    entrypoint: &Path,
    workspace: &Path,
    case: &Case,
    timeout: Duration,
) -> Result<Invocation> {
    let output_dir = TempDir::new_in(workspace).context("create invocation output directory")?;
    let stdout_path = output_dir.path().join("stdout");
    let stderr_path = output_dir.path().join("stderr");
    let stdout = File::create(&stdout_path).context("create candidate stdout")?;
    let stderr = File::create(&stderr_path).context("create candidate stderr")?;
    let started = Instant::now();
    let mut child = Command::new(entrypoint)
        .arg(case.a.to_string())
        .arg(case.b.to_string())
        .current_dir(workspace.join("payload"))
        .env_clear()
        .env("HOME", output_dir.path())
        .env("LANG", "C")
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .context("spawn payload/run.sh")?;
    let deadline = Instant::now() + timeout;
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().context("poll candidate")? {
            break (status, false);
        }
        if Instant::now() >= deadline {
            child.kill().context("kill timed-out candidate")?;
            break (child.wait().context("reap timed-out candidate")?, true);
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    let elapsed = started.elapsed();
    let stdout = read_output(&stdout_path)?;
    let stderr = read_output(&stderr_path)?;
    Ok(Invocation {
        elapsed,
        stdout,
        stderr,
        success: status.success(),
        timed_out,
    })
}

fn validate_invocation(invocation: &Invocation, case: &Case) -> Result<()> {
    if invocation.timed_out {
        bail!("candidate timed out");
    }
    if !invocation.success {
        bail!(
            "candidate exited unsuccessfully; stderr={:?}",
            invocation.stderr
        );
    }
    let value = invocation
        .stdout
        .trim()
        .parse::<u64>()
        .context("candidate stdout must contain one u64")?;
    let expected = case
        .a
        .checked_add(case.b)
        .context("expected sum overflow")?;
    ensure!(
        value == expected,
        "wrong answer: expected {expected}, got {value}"
    );
    Ok(())
}

fn write_evidence(path: &Path, evidence: &Evidence) -> Result<()> {
    ensure_parent(path)?;
    let bytes = serde_json::to_vec_pretty(evidence).context("serialize evidence")?;
    let file = File::create(path).context("create evidence file")?;
    let encoder = zstd::Encoder::new(file, 3).context("create zstd encoder")?;
    let mut archive = tar::Builder::new(encoder);
    let mut header = tar::Header::new_ustar();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o444);
    header.set_uid(0);
    header.set_gid(0);
    header.set_mtime(0);
    header.set_cksum();
    archive
        .append_data(&mut header, "report.json", bytes.as_slice())
        .context("append evidence report")?;
    let encoder = archive.into_inner().context("finish evidence tar")?;
    encoder.finish().context("finish evidence compression")?;
    Ok(())
}

fn failed_result(code: impl Into<String>, message: impl Into<String>) -> TaskResult {
    let message = message.into();
    TaskResult {
        status: "failed",
        raw_score: None,
        summary: message.clone(),
        metrics: BTreeMap::new(),
        failure: Some(Failure {
            code: code.into(),
            message,
            retryable: false,
        }),
    }
}

fn read_limited(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    ensure!(
        metadata.len() <= max_bytes,
        "{} exceeds {} bytes",
        path.display(),
        max_bytes
    );
    fs::read(path).with_context(|| format!("read {}", path.display()))
}

fn read_output(path: &Path) -> Result<String> {
    let bytes = read_limited(path, MAX_OUTPUT_BYTES)?;
    String::from_utf8(bytes).context("candidate output must be UTF-8")
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn required_absolute_path(name: &str) -> Result<PathBuf> {
    let value = env::var_os(name).ok_or_else(|| anyhow!("{name} is required"))?;
    let path = PathBuf::from(value);
    ensure!(path.is_absolute(), "{name} must be absolute");
    Ok(path)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    ensure_parent(path)?;
    let bytes = serde_json::to_vec_pretty(value).context("serialize result")?;
    let mut file = File::create(path).context("create result file")?;
    file.write_all(&bytes).context("write result file")?;
    file.sync_all().context("sync result file")?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<()> {
    let parent = path.parent().context("output path must have a parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_rejects_mode_mismatch_and_overflow() {
        let dataset = Dataset {
            schema_version: DATASET_SCHEMA.into(),
            mode: Mode::Train,
            correctness: vec![Case { a: u64::MAX, b: 1 }],
            benchmarks: vec![Benchmark {
                a: 1,
                b: 1,
                iterations: 1,
                timeout_ms: 100,
                reference_runtime_ns: 1,
            }],
        };
        assert!(validate_dataset(&dataset, Mode::Eval).is_err());
        assert!(validate_dataset(&dataset, Mode::Train).is_err());
    }

    #[test]
    fn payload_path_must_stay_below_payload() {
        assert!(validate_payload_path(Path::new("payload/run.sh")).is_ok());
        assert!(validate_payload_path(Path::new("other/run.sh")).is_err());
        assert!(validate_payload_path(Path::new("../payload/run.sh")).is_err());
        assert!(validate_payload_path(Path::new("/payload/run.sh")).is_err());
    }

    #[test]
    fn verifier_workspace_stays_on_the_mounted_work_volume() {
        assert_eq!(
            mounted_workspace_root(Path::new("/frontier/input/payload.tar")).unwrap(),
            Path::new("/frontier")
        );
        assert!(mounted_workspace_root(Path::new("/payload.tar")).is_err());
    }

    #[test]
    fn score_is_finite_and_rewards_lower_runtime() {
        let fast = 100_f64 / 10_f64;
        let slow = 100_f64 / 50_f64;
        assert!(fast.is_finite() && fast >= 0.0);
        assert!(fast > slow);
    }

    #[test]
    fn writes_zstd_evidence_tar() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("evidence.tar.zst");
        let evidence = Evidence {
            schema_version: "frontier.speed-add-evidence/v1",
            mode: Mode::Train,
            dataset_sha256: format!("sha256:{}", "0".repeat(64)),
            payload_sha256: format!("sha256:{}", "1".repeat(64)),
            correctness_cases: 1,
            benchmark_cases: 1,
            benchmark_iterations: 1,
            measured_median_ns: vec![1],
            reference_runtime_ns: vec![1],
            raw_score: Some(1.0),
            outcome: "passed",
            failure: None,
        };
        write_evidence(&path, &evidence).unwrap();
        assert!(fs::metadata(&path).unwrap().len() > 0);
        let file = File::open(&path).unwrap();
        let decoder = zstd::Decoder::new(file).unwrap();
        let mut archive = tar::Archive::new(decoder);
        let mut entries = archive.entries().unwrap();
        let mut report = String::new();
        let mut entry = entries.next().unwrap().unwrap();
        assert_eq!(entry.path().unwrap(), Path::new("report.json"));
        entry.read_to_string(&mut report).unwrap();
        let decoded: Value = serde_json::from_str(&report).unwrap();
        assert_eq!(decoded["schema_version"], "frontier.speed-add-evidence/v1");
        assert!(entries.next().is_none());
    }
}
