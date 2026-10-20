from __future__ import annotations

import deployment_test_guard  # noqa: F401

import importlib.util
import json
import os
import re
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Crypto/quarter_pir"
CODE = TASK / "judge/harbor/app/quarter_pir/code"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuarterPIRFullComputeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load("quarter_pir_gate_metrics_test", TASK / "judge/table2_metrics.py")
        submission = types.ModuleType("submission_patch")
        submission.apply_submission_patch = None
        with patch.dict(sys.modules, {"table2_metrics": cls.gate, "submission_patch": submission}):
            cls.evaluator = load("quarter_pir_evaluator_gate_test", TASK / "judge/evaluator.py")
        cls.spec = json.loads((TASK / "judge/beat_spec.json").read_text())

    def result(self, *, size=65536, offline=1_000_000_000, online=400_000_000, storage=2_000_000,
               hint="a" * 64, up=100, down=128, rss=700_000_000, probe=492, answered=None,
               probe_failure="", probe_digest="d" * 64, audit=512, mismatched=(), audit_failure="",
               audit_digest="f" * 64):
        if probe_failure:
            audit, audit_digest = 0, ""
        import math
        return {"valid": True, "failure": "", "db_size": size, "actual_queries": 1000,
                "planned_queries": int(math.sqrt(size) * math.log(size)), "threads": 4,
                "offline_elapsed_ns": offline, "online_elapsed_ns": online,
                "client_storage_bytes": storage, "client_peak_rss_bytes": rss,
                "hint_export_bytes": 1040384,
                "hint_export_sha256": hint, "transcript_sha256": "b" * 64,
                "upload_bytes": [up] * 1000, "download_bytes": [down] * 1000,
                "offline_communication_bytes": size * 16,
                "probe_queries": probe, "probe_answered": probe if answered is None else answered,
                "probe_failure": probe_failure, "probe_sha256": probe_digest,
                "audit_hints": audit, "audit_mismatched": list(mismatched), "audit_failure": audit_failure,
                "audit_sha256": audit_digest}

    def test_reference_messages_are_counted_at_the_tight_encoding(self):
        # Packed offsets: 32 offsets of 9 bits and 128 offsets of 14 bits.
        self.assertEqual(self.gate.reference_message_bytes(65536), (100, 128))
        self.assertEqual(self.gate.reference_message_bytes(67108864), (736, 1024))
        source = (CODE / "refwire/refwire.go").read_text()
        self.assertIn("(int(p.SBSize)*int(p.OffsetBits())+7)/8", source)

    def test_every_comparison_is_strict_without_allowance(self):
        reference = self.result()
        equal = self.gate.compare_experiment(self.result(), reference)
        self.assertEqual(equal, {"preprocessing_client_time_minutes": False,
                                 "end_to_end_online_time_ms": False,
                                 "client_storage_bytes": True})
        better = self.gate.compare_experiment(
            self.result(offline=reference["offline_elapsed_ns"] - 1,
                        online=reference["online_elapsed_ns"] - 1), reference)
        self.assertEqual(set(better.values()), {True})
        worse = self.gate.compare_experiment(
            self.result(offline=reference["offline_elapsed_ns"] + 1), reference)
        self.assertFalse(worse["preprocessing_client_time_minutes"])
        self.assertEqual(self.gate.compare_rpc(10, 11, 227, 228),
                         {"online_communication_bytes": True, "server_compute_time_ns": True})
        self.assertFalse(self.gate.compare_rpc(10, 11, 228, 228)["online_communication_bytes"])
        self.assertFalse(self.gate.compare_rpc(11, 11, 227, 228)["server_compute_time_ns"])
        self.assertFalse(self.gate.compare_rpc(0, 11, 0, 228)["online_communication_bytes"])
        for name in ("preprocessing_allowance", "compare_table2", "parse_table2"):
            self.assertFalse(hasattr(self.gate, name))

    def test_client_storage_is_gated_as_non_inferiority_with_the_measured_jitter_only(self):
        reference = self.result()
        jitter = self.gate.STORAGE_JITTER_BYTES
        # Next power of two above the largest measured reference spread (19,344 bytes).
        self.assertEqual(jitter, 32768)
        key = "client_storage_bytes"
        self.assertTrue(self.gate.compare_experiment(self.result(storage=2_000_000 + jitter), reference)[key])
        self.assertFalse(self.gate.compare_experiment(self.result(storage=2_000_001 + jitter), reference)[key])
        self.assertTrue(self.gate.compare_experiment(self.result(storage=1), reference)[key])
        # One dropped superblock key costs far more than the jitter at either size.
        for size in (65536, 67108864):
            p = self.gate.parameters(size)
            self.assertGreater(16 * (p["primary_hints"] + p["backup_hints"]), 10 * jitter)
        self.assertFalse(hasattr(self.gate, "storage_cap"))

    def test_paired_validity_requires_the_hint_export_and_the_process_memory_guard(self):
        reference = self.result()
        self.assertIsNone(self.gate.validity_failure(self.result(), reference))
        self.assertEqual(self.gate.validity_failure(self.result(hint="c" * 64), reference),
                         "hint_export_mismatch")
        limit = self.gate.PEAK_RSS_LIMIT_BYTES
        self.assertEqual(limit, 4 * 1024 ** 3)
        self.assertIsNone(self.gate.validity_failure(self.result(rss=limit), reference))
        self.assertEqual(self.gate.validity_failure(self.result(rss=limit + 1), reference),
                         "process_memory_limit")
        # Storage far above the reference is a failed comparison, not a rejection.
        self.assertIsNone(self.gate.validity_failure(self.result(storage=10 ** 9), reference))

    def test_parameters_match_the_go_derivation(self):
        small, large = self.gate.parameters(65536), self.gate.parameters(67108864)
        self.assertEqual((small["chunk_size"], small["set_size"], small["superblocks"],
                          small["primary_hints"], small["backup_hints"]), (512, 128, 4, 18944, 8704))
        self.assertEqual((large["chunk_size"], large["set_size"], large["superblocks"],
                          large["primary_hints"], large["backup_hints"]), (16384, 4096, 32, 671744, 442368))

    def test_score_is_led_by_communication_and_uses_only_candidate_measurements(self):
        def rows(small_bytes=228, large_bytes=1760, speed=1.0):
            return [{"db_size": 65536, "queries": 1000, "bytes": [small_bytes] * 1000,
                     "server_ns": [11000 / speed] * 1000, "offline_ns": 0.16e9 / speed,
                     "online_ns": 0.32e9 / speed},
                    {"db_size": 67108864, "queries": 1000, "bytes": [large_bytes] * 1000,
                     "server_ns": [480000 / speed] * 1000, "offline_ns": 120e9 / speed,
                     "online_ns": 4.1e9 / speed}]
        base = self.gate.continuous_score(rows())
        self.assertTrue(20 < base < 40)
        self.assertGreater(self.gate.continuous_score(rows(small_bytes=196, large_bytes=1696)), base)
        self.assertLess(self.gate.continuous_score(rows(small_bytes=320, large_bytes=2048)), base)
        self.assertGreater(self.gate.continuous_score(rows(speed=2.0)), base)
        self.assertLess(self.gate.continuous_score(rows(speed=1e9, small_bytes=1, large_bytes=1)), 100.0)
        weights = self.gate.SCORE_WEIGHTS
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(max(weights, key=weights.get), "communication")
        self.assertEqual(self.gate.score_scales(65536)["communication"], 228.0)
        self.assertEqual(self.gate.score_scales(67108864)["communication"], 1760.0)
        import inspect
        self.assertNotIn("reference", inspect.signature(self.gate.continuous_score).parameters)
        with self.assertRaises(ValueError):
            self.gate.continuous_score(rows(small_bytes=0))

    def test_malformed_experiment_records_are_rejected(self):
        self.gate.check_result(self.result(), size=65536)
        for change in ({"valid": False}, {"failure": "wrong_value"}, {"actual_queries": 999},
                       {"planned_queries": 1000}, {"offline_elapsed_ns": 0},
                       {"online_elapsed_ns": 1.5}, {"upload_bytes": [100] * 999},
                       {"download_bytes": [0] * 1000}, {"hint_export_sha256": ""},
                       {"probe_sha256": ""}, {"probe_queries": 0}, {"probe_queries": 1840, "probe_answered": 1840},
                       {"probe_answered": 491}, {"probe_failure": "wrong_value"},
                       {"probe_answered": 10, "probe_failure": "skipped"}, {"probe_answered": -1},
                       {"audit_hints": 0}, {"audit_sha256": ""}, {"audit_mismatched": [512]},
                       {"audit_mismatched": [3, 3]}, {"audit_failure": "other"},
                       {"audit_failure": "client_crashed_or_protocol", "audit_mismatched": [1]}):
            record = self.result()
            record.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.gate.check_result(record, size=65536)
        with self.assertRaises(ValueError):
            self.gate.check_result(self.result(), size=67108864)

    def test_amortized_offline_time_is_reported_but_not_gated(self):
        values = self.gate.timing_values(self.result())
        self.assertIn("amortized_offline_time_ms", values)
        self.assertNotIn("amortized_offline_time_ms", self.gate.GATED_EXPERIMENT_METRICS)
        gated = [row["name"] for row in self.spec["metrics"]]
        self.assertEqual(gated, ["online_communication_bytes", "server_compute_time_ns",
                                 "preprocessing_client_time_minutes", "end_to_end_online_time_ms",
                                 "client_storage_bytes"])
        self.assertEqual(self.spec["beats_reference_aggregation"], {"kind": "conjunction"})

    def test_failure_rows_cover_every_gated_metric_at_its_declared_granularity(self):
        for final, role in ((False, "train"), (True, "final")):
            rows = self.evaluator.failure_metrics(final)["beat_spec_workload_passes"]
            for group, record in self.spec["workload_groups"].items():
                selected = [row for row in rows if row["group"] == group]
                self.assertEqual(len(selected), record["counts"][role])
                self.assertTrue(all(list(row["metrics"]) == record["metrics"] for row in selected))
                self.assertFalse(any(value for row in selected for value in row["metrics"].values()))

    def test_visible_feedback_is_declared_consistently(self):
        visible = [row["name"] for row in self.spec["solver_feedback"]]
        policy = json.loads((TASK / "policy.json").read_text())
        self.assertEqual(self.evaluator.VISIBLE, visible)
        self.assertEqual(policy["metadata"]["evaluation"]["solver_visible_metrics"], visible)
        self.assertNotIn("preprocessing_calibration", policy["metadata"])
        statement = (TASK / "judge/readme").read_text()
        for phrase in ("strictly fewer bytes on the wire", "deterministic public function",
                       "The response is free", "client storage must not exceed",
                       "trusted and locked", "Goroutines must settle before an entry point returns",
                       "in the client, in the server and in `ReduceRequest` alike",
                       "32,768 bytes",
                       "scored about 28 on the authors' development host; the value depends on the hardware"):
            self.assertIn(phrase, statement)
        for stale in ("#346", "issue ", "predecessor", "calibrated", "64 MiB", "@", "between 26 and 28"):
            self.assertNotIn(stale, statement)
        for name in visible:
            self.assertIn(name, statement)
        for category, code in self.evaluator.VALIDITY_CODES.items():
            self.assertRegex(statement, rf"\b{code} ")
        self.assertEqual(sorted(self.evaluator.VALIDITY_CODES.values()), list(range(13)))
        self.assertEqual(self.evaluator.VALIDITY_CODES["hint_audit"], 12)
        self.assertIn("12 primary-hint audit failed", statement)
        self.assertEqual(self.evaluator.VALIDITY_CODES["coverage_probe"], 11)
        self.assertIn("11 failed an additional untimed query", statement)
        self.assertIn("11 failed an additional untimed query", json.dumps(self.spec))

    def test_coverage_probe_is_a_paired_validity_condition_outside_every_metric(self):
        reference = self.result()
        self.gate.check_result(self.result(answered=300, probe_failure="wrong_value"), size=65536)
        self.assertIsNone(self.gate.probe_failure(self.result(), reference))
        # A failed additional query that the reference answered invalidates the run.
        for check in sorted(self.gate.PROBE_FAILURES):
            failed = self.gate.check_result(self.result(answered=300, probe_failure=check), size=65536)
            self.assertEqual(self.gate.probe_failure(failed, reference), check)
        self.assertEqual(self.gate.probe_failure(self.result(answered=0, probe_failure="wrong_value"), reference),
                         "wrong_value")
        # A query the reference itself failed, and what follows it, does not count.
        broken = self.result(answered=300, probe_failure="wrong_value")
        self.assertIsNone(self.gate.probe_failure(self.result(answered=300, probe_failure="wrong_value"), broken))
        self.assertIsNone(self.gate.probe_failure(self.result(answered=400, probe_failure="wrong_value"), broken))
        self.assertIsNone(self.gate.probe_failure(self.result(), broken))
        self.assertEqual(self.gate.probe_failure(self.result(answered=299, probe_failure="wrong_value"), broken),
                         "wrong_value")
        # Both runs must have been asked the identical sequence.
        for other in (self.result(probe_digest="e" * 64), self.result(probe=491)):
            with self.assertRaises(ValueError):
                self.gate.probe_failure(other, reference)
        # The probe changes no gate decision and no score input.
        decisions = self.gate.compare_experiment(self.result(answered=0, probe_failure="wrong_value"), reference)
        self.assertEqual(decisions, self.gate.compare_experiment(self.result(), reference))
        conductor = (CODE / "conductor/conductor.go").read_text()
        probe = conductor.index("planProbe(generator, issued, settings.ProbeSeed")
        for measured in ("outcome.OnlineQueryNs +=", "outcome.UploadBytes = append", "transcript = append",
                         "outcome.OnlineNs =", "outcome.TranscriptSHA256 ="):
            self.assertEqual(conductor.count(measured), 1)
            self.assertLess(conductor.index(measured), probe)
        self.assertIn('failure = "message_longer_than_measured"', conductor)
        evaluator = (TASK / "judge/evaluator.py").read_text()
        self.assertIn("raise Rejected('additional untimed query failed', 'coverage_probe'", evaluator)
        self.assertIn("raise TaskConfigurationError('paired coverage probes differ')", evaluator)

    def test_primary_hint_audit_is_paired_and_asks_only_for_unconsumed_primary_hints(self):
        reference = self.result()
        self.assertIsNone(self.gate.audit_failure(self.result(), reference))
        self.assertEqual(self.gate.audit_failure(self.result(mismatched=[7]), reference), "record_not_canonical")
        self.assertEqual(self.gate.audit_failure(self.result(audit_failure="client_crashed_or_protocol"), reference),
                         "client_crashed_or_protocol")
        # A hint the reference client itself did not reproduce does not count.
        self.assertIsNone(self.gate.audit_failure(self.result(mismatched=[7]), self.result(mismatched=[7, 9])))
        self.assertEqual(self.gate.audit_failure(self.result(mismatched=[7, 8]), self.result(mismatched=[7])),
                         "record_not_canonical")
        self.assertIsNone(self.gate.audit_failure(self.result(mismatched=[7]),
                                                  self.result(audit_failure="client_crashed_or_protocol")))
        # No audit after a failed probe; the probe verdict decides.
        failed = self.gate.check_result(self.result(answered=3, probe_failure="wrong_value"), size=65536)
        self.assertIsNone(self.gate.audit_failure(failed, reference))
        with self.assertRaises(ValueError):
            self.gate.audit_failure(self.result(audit_digest="e" * 64), reference)
        probe = (CODE / "conductor/probe.go").read_text()
        self.assertIn("g.Labels[hint] != refwire.NoLabel || uint64(g.Tags[hint]) != hint", probe)
        driver = (CODE / "ec24/ec24_client.go").read_text()
        audit = driver[driver.index("if tag == 'U'"):driver.index("if tag != 'Q'")]
        self.assertLess(audit.index("client.PrimaryHint(hint)"), audit.index("settled(goroutines)"))
        self.assertNotIn("online +=", audit)
        self.assertIn("export[hint*uint64(len(record)):", audit)
        statement = (TASK / "judge/readme").read_text()
        self.assertIn("func (c *Client) PrimaryHint(j uint64) (uint32, util.DBEntry)", statement)
        self.assertIn("Nothing is asked about backup hints or about hints a query has consumed", statement)

    def test_submission_code_is_given_the_planned_budget_not_the_measured_query_count(self):
        conductor = (CODE / "conductor/conductor.go").read_text()
        spawn = conductor[conductor.index("func spawn("):conductor.index("func peakResident(")]
        self.assertNotIn("Queries", spawn)
        self.assertNotIn("queries", spawn.replace("number of queries this run", ""))
        for name in ("ec24/ec24_client.go", "reducer/reducer.go", "server/server.go", "server/replay.go"):
            source = (CODE / name).read_text()
            self.assertIn("refwire.NewPlannedParams(", source)
            self.assertNotIn("refwire.NewParams(", source)
            self.assertNotIn('"queries"', source)
        refwire = (CODE / "refwire/refwire.go").read_text()
        self.assertIn("p.OnlineQueries = p.PlannedQueries()", refwire)
        # The generator's history is not sized by the query count it is given.
        self.assertIn("g.HistoryHit = make([]uint64, 0, historyReserve)", refwire)
        self.assertNotIn("make([]uint64, p.OnlineQueries)", refwire)
        self.assertNotIn("NewPlannedParams", self.policy_allowlist()["easypir/refwire"])
        statement = (TASK / "judge/readme").read_text()
        self.assertIn("`OnlineQueries` is the planned query budget of one preprocessing", statement)
        self.assertIn("chosen by the harness, possibly adversarially, within what the provided implementation answers",
                      statement)
        self.assertNotIn("`OnlineQueries` (1,000)", statement)
        self.assertNotIn("For each of 1,000 queries", statement)
        for fixture in sorted((ROOT / "tests/maintenance/quarter_pir/fixtures").rglob("*.patch")) + [TASK / "judge/reference.patch"]:
            self.assertNotIn("OnlineQueries", fixture.read_text(), fixture.name)

    def test_seeds_are_fresh_private_and_never_constant(self):
        first, second = self.evaluator.fresh_seeds(), self.evaluator.fresh_seeds()
        self.assertEqual(set(first), {"db_seed", "coin_seed", "query_seed", "probe_seed"})
        self.assertNotEqual(first, second)
        source = (TASK / "judge/evaluator.py").read_text()
        self.assertNotIn("133191", source)
        self.assertNotIn("PUBLIC_SEED", source)
        driver = (CODE / "ec24/ec24_client.go").read_text()
        self.assertNotIn("DBSeed", driver)
        self.assertNotIn("config.txt", driver)
        self.assertNotIn('"net"', driver)
        self.assertNotIn("grpc", driver)

    def policy_allowlist(self):
        policy = (CODE / "policy/policy.go").read_text()
        block = policy[policy.index("var allowedImports"):policy.index("func names(")]
        allowed = {}
        for path, value in re.findall(r'^\s*"([^"]+)":\s*(nil|names\((?:"[^"]*"\s*\+?\s*)+\)),', block, flags=re.M):
            allowed[path] = None if value == "nil" else set(" ".join(re.findall(r'"([^"]*)"', value)).split())
        return allowed

    def test_python_and_go_import_allowlists_agree(self):
        allowed = self.policy_allowlist()
        self.assertEqual(set(allowed), set(self.evaluator.ALLOWED_IMPORTS))
        for forbidden in ("os", "net", "syscall", "unsafe", "reflect", "time", "runtime", "log", "fmt",
                          "embed", "plugin", "C", "os/exec", "runtime/debug", "io/ioutil",
                          "crypto/aes", "crypto/cipher", "crypto/subtle", "errors"):
            self.assertNotIn(forbidden, allowed)
        for required in ("sync", "sync/atomic", "math/rand"):
            self.assertIn(required, allowed)
        self.assertEqual(allowed["math/rand"], {"New", "NewSource", "Rand", "Source", "Source64"})
        # Every trusted in-repo package is restricted to an explicit symbol list.
        for path in ("easypir/safeaes", "easypir/refwire", "example.com/util"):
            self.assertIsNotNone(allowed[path])
        self.assertEqual(allowed["easypir/safeaes"], {"RoundKeys", "ExpandKey", "MMO", "Encrypt", "Xor16"})
        statement = (TASK / "judge/readme").read_text()
        safe_api = statement[statement.index("**Safe API.**"):statement.index("Seeds are fresh and private")]
        for path in ("easypir/safeaes", "easypir/refwire", "example.com/util"):
            for name in allowed[path]:
                self.assertRegex(safe_api, rf"\b{name}\b")

    def test_no_slice_or_raw_pointer_can_reach_assembly_from_the_editable_package(self):
        allowed = self.policy_allowlist()
        # The editable package is Go source only; assembly lives in trusted packages.
        for tree in (CODE / "pir", TASK / "agent-base/frontier/task/repo/quarter_pir/code/pir"):
            self.assertEqual(sorted(path.name for path in tree.iterdir()), ["client.go", "reduce.go", "server.go"])
            for path in tree.iterdir():
                self.assertFalse(re.search(r"^func [^{]*\)\s*$", path.read_text(), flags=re.M))
        assembly = sorted(path.relative_to(CODE).as_posix() for path in CODE.rglob("*.s"))
        self.assertEqual(assembly, ["safeaes/aes_amd64.s", "safeaes/aes_arm64.s",
                                    "util/aes_amd64.s", "util/aes_arm64.s"])
        # safeaes: every exported function takes only pointers to fixed-size arrays.
        safeaes = (CODE / "safeaes/safeaes.go").read_text()
        exported = re.findall(r"^func ([A-Z]\w*)\(([^)]*)\)", safeaes, flags=re.M)
        self.assertEqual({name for name, _ in exported} | {"RoundKeys"}, allowed["easypir/safeaes"])
        for name, parameters in exported:
            parts = [parameter.split() for parameter in parameters.split(",")]
            self.assertEqual(len(parts[-1]), 2, name)  # the last parameter carries a type
            for part in parts:
                if len(part) == 2:  # "a, b T" groups: names without a type share the next one
                    self.assertIn(part[1], ("*[16]byte", "*RoundKeys"), name)
        self.assertIn("type RoundKeys [44]uint32", safeaes)
        self.assertNotIn("unsafe", safeaes)
        # Raw entry points are unexported, so no other package can name them.
        for raw in re.findall(r"^func ([a-z]\w*)\(", safeaes, flags=re.M):
            self.assertNotIn(raw, allowed["easypir/safeaes"])
        # refwire: no allowed function accepts an expanded key as a slice.
        refwire = (CODE / "refwire/refwire.go").read_text()
        self.assertIn("type LongKey [44]uint32", refwire)
        for name in ("Eval", "SubKey", "SetOffset"):
            signature = re.search(rf"^func {name}\(([^)]*)\)", refwire, flags=re.M).group(1)
            self.assertTrue(signature.startswith("longKey *LongKey"), name)
        self.assertIn("func NewLongKey(key *[16]byte) *LongKey", refwire)
        self.assertNotRegex(refwire, r"^func [A-Z]\w*\([^)]*\[\]uint32[^)]*\) (uint64|\[16\]byte)", )
        self.assertIn("LongKey        *LongKey", refwire)
        self.assertNotIn("unsafe", refwire)
        # util: nothing that forwards a caller's slice to assembly is importable.
        util = (CODE / "util/util.go").read_text()
        unsafe_helpers = {"NewCipher", "AesPrf", "PRFEvalWithLongKey", "PRFEvalWithLongKeyAndTag",
                          "PRFEvalWithLongKeyAndTag128Output", "MembTestWithLongKeyAndTag", "PRFEval1",
                          "PRFEval2", "PRFEval3", "PRFEval4", "PRFEval", "GetLongKey", "GenDBEntry",
                          "DefaultHash", "GenParams", "PRSet", "PRSetWithShortTag"}
        self.assertFalse(unsafe_helpers & allowed["example.com/util"])
        for name in allowed["example.com/util"]:
            self.assertRegex(util, rf"(func|type) {name}\b|\b{name}\s+=")
        # The upstream under-allocation in NewCipher is fixed and Encrypt checks its blocks.
        self.assertIn("c := AesPrf{make([]uint32, 11*4)}", util)
        self.assertIn("len(c.enc) < 11*4 || len(dst) < 16 || len(src) < 16", util)
        # Compiled-package check: no assembly, cgo or ignored files in the editable package.
        evaluator = (TASK / "judge/evaluator.py").read_text()
        self.assertIn("'IgnoredOtherFiles', 'SFiles'", evaluator)
        self.assertNotIn("aes_locked", evaluator)

    def test_trusted_processes_keep_submission_code_away_from_secrets(self):
        conductor = (CODE / "conductor/conductor.go").read_text()
        reducer = (CODE / "reducer/reducer.go").read_text()
        server = (CODE / "server/server.go").read_text()
        self.assertNotIn('"easypir/pir"', conductor)
        self.assertIn("rawrpc.Call(ctx, connection, reduced)", conductor)
        self.assertIn("bytes.Equal(request, reduced)", conductor)
        self.assertIn("util.GenDBEntry(settings.DBSeed, x)", conductor)
        self.assertIn("pir.ReduceRequest(params,", reducer)
        for secret in ("DBSeed", "CoinSeed", "QuerySeed", "seed"):
            self.assertNotIn(secret, reducer)
        clock = server[server.index("func (s *QueryServiceServer) TimedQuery"):]
        clock = clock[:clock.index("return response")]
        self.assertLess(clock.index("time.Now()"), clock.index("pir.HandleQuery(s.DB, Params, request)"))
        self.assertLess(clock.index("pir.HandleQuery"), clock.index("time.Since(start)"))
        self.assertNotIn("PlaintextQuery", server)

    def test_every_trusted_call_of_submission_code_has_a_completion_boundary(self):
        import re
        calls = {}
        for path in sorted(CODE.rglob("*.go")):
            if path.parent.name == "pir":
                continue
            found = re.findall(r"\bpir\.([A-Z]\w*)\(", path.read_text())
            if found:
                calls[path.relative_to(CODE).as_posix()] = found
        # The only trusted callers: driver (constructor), reducer and server clock.
        self.assertEqual(calls, {"ec24/ec24_client.go": ["NewClient"],
                                 "reducer/reducer.go": ["ReduceRequest"],
                                 "server/server.go": ["HandleQuery"]})
        server = (CODE / "server/server.go").read_text()
        clock = server[server.index("func (s *QueryServiceServer) TimedQuery"):]
        clock = clock[:clock.index("return response")]
        order = [clock.index(part) for part in (
            "runtime.NumGoroutine()", "time.Now()", "pir.HandleQuery(", "settle.Wait(goroutines)",
            "time.Since(start)")]
        self.assertEqual(order, sorted(order))
        self.assertIn("response, _ := s.TimedQuery(request)", server)  # live path uses the same clock
        replay = (CODE / "server/replay.go").read_text()
        self.assertNotIn("pir.", replay)
        self.assertGreaterEqual(replay.count("server.TimedQuery("), 3)
        reducer = (CODE / "reducer/reducer.go").read_text()
        body = reducer[reducer.index("func reduce("):reducer.index("func main()")]
        order = [body.index(part) for part in (
            "runtime.NumGoroutine()", "pir.ReduceRequest(", "settle.Wait(goroutines)", "return append(")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(reducer.count("reduce(params, reference)"), 2)
        self.assertEqual(reducer.count("pir.ReduceRequest("), 1)
        driver = (CODE / "ec24/ec24_client.go").read_text()
        for call in ("pir.NewClient(params, coinSeed)", "client.Preprocess(", "client.ExportHints(",
                     "client.Query(", "client.Reconstruct(", "client.PrimaryHint("):
            after = driver[driver.index(call):]
            self.assertIn("settled(goroutines)", after)
        self.assertEqual(driver.count("settled(goroutines)"), 6)
        # Preprocess's boundary precedes closing the stream; export's precedes the clock marker.
        self.assertLess(driver.index("client.ExportHints("), driver.index("frame.Write(out, 'p', nil)"))
        self.assertLess(driver.rindex("settled(goroutines)", 0, driver.index("frame.Write(out, 'p', nil)")),
                        driver.index("frame.Write(out, 'p', nil)"))
        helper = (CODE / "settle/settle.go").read_text()
        self.assertIn("const Limit = 10 * time.Second", helper)

    def test_protected_files_must_be_byte_identical(self):
        harbor = TASK / "judge/harbor/app"
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            tree = Path(directory) / "tree"
            shutil.copytree(harbor, tree)
            with self.subTest("protected tree"):
                self.evaluator.check_protected(tree)
                (tree / "quarter_pir/code/pir/extra.go").write_text("package pir\n")
                self.evaluator.check_protected(tree)
                driver = tree / "quarter_pir/code/ec24/ec24_client.go"
                driver.write_text(driver.read_text() + "\n")
                with self.assertRaises(self.evaluator.Rejected):
                    self.evaluator.check_protected(tree)
                shutil.copy(harbor / "quarter_pir/code/ec24/ec24_client.go", driver)
                (tree / "quarter_pir/code/util/extra.go").write_text("package util\n")
                with self.assertRaises(self.evaluator.Rejected):
                    self.evaluator.check_protected(tree)
                (tree / "quarter_pir/code/util/extra.go").unlink()
                (tree / "quarter_pir/code/pir/link.go").symlink_to("/etc/passwd")
                with self.assertRaises(self.evaluator.Rejected):
                    self.evaluator.check_protected(tree)


if __name__ == "__main__":
    unittest.main()
