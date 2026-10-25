"""Integrity and failure checks for frozen native reference captures."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from capture_reference_stump import parse_result
from capture_reference_core import parse_result as parse_core, inputs as core_inputs, capture, parse_pruning, parse_fixed_path
from reference_cases import cases, all_cases

ROOT = Path(__file__).resolve().parents[1]


class ReferenceCaptureTests(unittest.TestCase):
    def test_manual_matches_recorded_provenance(self):
        manual = json.loads((ROOT / "docs/reference-upstream.json").read_text())["manual"]
        local_copy = ROOT / manual["local_copy"]
        if not local_copy.exists():
            self.skipTest(f"local manual {manual['local_copy']} is not present (ignored by git)")
        self.assertEqual(hashlib.sha256(local_copy.read_bytes()).hexdigest(),
                         manual["sha256"])

    def test_all_frozen_files_match_recorded_hashes_and_runtime(self):
        manifests = sorted((ROOT / "tests/fixtures/reference").glob("guide-*/*/manifest.json"))
        catalog = all_cases()
        self.assertTrue(manifests)
        found = {}
        for path in manifests:
            with self.subTest(fixture=str(path)):
                manifest = json.loads(path.read_text())
                self.assertTrue(manifest["completed"])
                self.assertEqual(manifest["returncode"], 0)
                files = {}
                for name, digest in manifest["file_sha256"].items():
                    raw = (path.parent / name).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), digest, name)
                    files[name] = raw.decode()
                version = manifest["requested_version"]
                name = path.parent.name
                found.setdefault(version, set()).add(name)
                self.assertEqual(manifest["result"]["version"], version)
                metadata = json.loads((ROOT / "docs/reference-upstream.json").read_text())
                reference = next(r for r in metadata["references"] if r["version"] == version)
                self.assertEqual(manifest["binary_sha256"], reference["binary_sha256"])
                if manifest["fixture_format"] == 1:
                    self.assertEqual(parse_result(files["stdout.txt"], files, version), manifest["result"])
                else:
                    self.assertEqual(manifest["fixture_format"], 2)
                    self.assertEqual(manifest["settings"], catalog[name]["options"])
                    for filename, expected in core_inputs(catalog[name], version).items():
                        self.assertEqual(files[filename], expected)
                    self.assertEqual(parse_core(files["stdout.txt"], files, version, catalog[name]),
                                     manifest["result"])
        self.assertEqual(set(found), {"46.2", "46.3"})
        for names in found.values():
            self.assertEqual(names, set(catalog))

    def fixture(self):
        path = ROOT / "tests/fixtures/reference/guide-46.3/ls-stump-10"
        return {name: (path / name).read_text()
                for name in ["stdout.txt", "predictions.txt", "splits.txt", "regression.out"]}

    def test_runtime_version_mismatch_is_rejected(self):
        files = self.fixture()
        with self.assertRaisesRegex(ValueError, "runtime version"):
            parse_result(files["stdout.txt"], files, "44.1")

    def test_incomplete_or_ambiguous_output_is_rejected(self):
        files = self.fixture()
        for name in ["predictions.txt", "splits.txt", "regression.out"]:
            with self.subTest(file=name):
                broken = copy.deepcopy(files)
                broken[name] = "incomplete\n"
                with self.assertRaises((ValueError, IndexError)):
                    parse_result(files["stdout.txt"], broken, "46.3")

    def core_fixture(self, name):
        path = ROOT / "tests/fixtures/reference/guide-46.3" / name
        manifest = json.loads((path / "manifest.json").read_text())
        return {n: (path / n).read_text() for n in manifest["file_sha256"]}

    def test_core_rejects_malformed_predictions_and_wrong_version(self):
        files = self.core_fixture("class-balanced")
        value = cases()["class-balanced"]
        with self.assertRaisesRegex(ValueError, "runtime version"):
            parse_core(files["stdout.txt"], files, "46.2", value)
        for corrupt in ["train node observed predicted\n", files["predictions.txt"].replace("0.99383E+00", "NaN", 1),
                        files["predictions.txt"].replace('"a"  "a"', '"z"  "a"', 1)]:
            with self.subTest(predictions=corrupt[:80]):
                broken = {**files, "predictions.txt": corrupt}
                with self.assertRaises(ValueError):
                    parse_core(files["stdout.txt"], broken, "46.3", value)

    def test_core_rejects_conflicting_or_disconnected_trees(self):
        files = self.core_fixture("class-categorical")
        value = cases()["class-categorical"]
        splits = files["splits.txt"]
        for corrupt in ["\n".join(splits.splitlines()[:-1]), splits + splits.splitlines()[0] + "\n",
                        splits.replace('"c" "d"', '"a" "b"', 1), splits.replace("3 t", "9 t")]:
            with self.subTest(splits=corrupt):
                with self.assertRaises(ValueError):
                    parse_core(files["stdout.txt"], {**files, "splits.txt": corrupt}, "46.3", value)

    def test_single_class_is_only_accepted_with_explicit_rejection(self):
        files = self.core_fixture("class-constant-response")
        value = cases()["class-constant-response"]
        self.assertEqual(parse_core(files["stdout.txt"], files, "46.3", value)["outcome"], "rejected")
        for changed in [{**files, "report.out": "incomplete"}, {**files, "predictions.txt": "unexpected"}]:
            with self.assertRaises(ValueError):
                parse_core(files["stdout.txt"], changed, "46.3", value)

    def test_failed_processes_are_diagnostics_and_existing_outputs_are_preserved(self):
        value = cases()["class-balanced"]
        with tempfile.TemporaryDirectory(prefix="guide-capture-test-") as temporary:
            directory = Path(temporary)
            binary = directory / "binary"
            binary.write_bytes(b"test double; never executed")
            failures = [subprocess.CompletedProcess([], -9, "interrupted", "failure"),
                        subprocess.CompletedProcess([], 0, "wrong version", ""),
                        subprocess.TimeoutExpired([], 30, output=b"partial output", stderr=b"partial error")]
            for i, failure in enumerate(failures):
                output = directory / str(i)
                with patch("capture_reference_core.subprocess.run") as run:
                    if isinstance(failure, Exception):
                        run.side_effect = failure
                    else:
                        run.return_value = failure
                    manifest = capture(binary, "46.3", "https://example.invalid/reference", value, output)
                self.assertFalse(manifest["completed"])
                self.assertIn("error", manifest)
                self.assertNotIn("result", manifest)
                if isinstance(failure, subprocess.TimeoutExpired):
                    self.assertEqual((output / "stdout.txt").read_text(), "partial output")
                before = (output / "manifest.json").read_bytes()
                with patch("capture_reference_core.subprocess.run") as run:
                    with self.assertRaises(FileExistsError):
                        capture(binary, "46.3", "unused", value, output)
                    run.assert_not_called()
                self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_effective_settings_drift_is_rejected(self):
        files = self.core_fixture("class-balanced")
        value = cases()["class-balanced"]
        for original, changed in [("Maximum number of split levels: 1", "Maximum number of split levels: 15"),
                                  ("Minimum node sample size: 2", "Minimum node sample size: 3")]:
            report = files["report.out"].replace(original, changed)
            self.assertNotEqual(report, files["report.out"])
            with self.assertRaisesRegex(ValueError, "effective"):
                parse_core(files["stdout.txt"], {**files, "report.out": report}, "46.3", value)

    def test_cv_table_missing_rows_bad_values_and_selection_are_rejected(self):
        files = self.core_fixture("ls-loocv-default")
        report = files["report.out"]
        score_line = next(line for line in report.splitlines() if "0**" in line)
        last_line = next(line for line in report.splitlines() if "9.972E+00" in line)
        for corrupt in [report.replace(last_line, ""), report.replace("1.800E+00", "NaN"),
                        report.replace(score_line, score_line.replace("0**", "3**"))]:
            with self.assertRaises(ValueError):
                parse_pruning(corrupt)
        for corrupt in [report.replace("0**", "0"), report.replace("with v = 20", "with v = 10")]:
            with self.assertRaises(ValueError):
                parse_core(files["stdout.txt"], {**files, "report.out": corrupt}, "46.3", all_cases()["ls-loocv-default"])

    def test_fixed_path_and_native_counterexample_provenance(self):
        report = self.core_fixture("ls-fixed-ties")["report.out"]
        rows = parse_fixed_path(report)
        self.assertEqual([r["leaves"] for r in rows], [4, 3, 2, 1])
        line = next(line for line in report.splitlines() if "16.000" in line)
        with self.assertRaises(ValueError):
            parse_fixed_path(report.replace(line, ""))
        trace = json.loads((ROOT / "tests/fixtures/reference/diagnostics/cv-node-order-46.3.json").read_text())
        metadata = json.loads((ROOT / "docs/reference-upstream.json").read_text())
        native = next(r for r in metadata["references"] if r["version"] == "46.3")
        self.assertEqual(trace["binary_sha256"], native["binary_sha256"])
        base = ROOT / "tests/fixtures/reference/guide-46.3" / trace["case"]
        for filename, digest in trace["input_sha256"].items():
            self.assertEqual(hashlib.sha256((base / filename).read_bytes()).hexdigest(), digest)
        costs = [a for a, terminal in zip(trace["node_alphas"], trace["terminal_flags"]) if not terminal]
        self.assertGreater(trace["selected_candidate_alpha"], min(costs))


if __name__ == "__main__":
    unittest.main()
