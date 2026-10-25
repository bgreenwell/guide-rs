"""Immutable, pinned multiple-linear tree captures and their failure checks."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from capture_reference_linear_tree import capture, inputs, parse_result
from reference_linear_tree_cases import cases
from capture_multilevel_cv_oracle import capture as capture_oracle

ROOT = Path(__file__).resolve().parents[1]


class LinearTreeCaptureTests(unittest.TestCase):
    def fixture(self, name):
        path = ROOT / "tests/fixtures/linear-tree/guide-46.3" / name
        manifest = json.loads((path / "manifest.json").read_text())
        return {name: (path / name).read_text() for name in manifest["file_sha256"]}

    def test_complete_pinned_inputs_hashes_and_matching_native_versions(self):
        metadata = json.loads((ROOT / "docs/reference-upstream.json").read_text())
        found, results = {}, {}
        for path in sorted((ROOT / "tests/fixtures/linear-tree").glob("guide-*/*/manifest.json")):
            with self.subTest(fixture=str(path)):
                m = json.loads(path.read_text())
                version, value = m["requested_version"], cases()[m["case"]]
                found.setdefault(version, set()).add(m["case"])
                self.assertTrue(m["completed"])
                self.assertEqual(m["returncode"], 0)
                native = next(r for r in metadata["references"] if r["version"] == version)
                self.assertEqual(m["binary_sha256"], native["binary_sha256"])
                self.assertEqual(m["source_url"], native["source_url"])
                files = {}
                for name, digest in m["file_sha256"].items():
                    raw = (path.parent / name).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                    files[name] = raw.decode()
                for name, expected in inputs(value, version).items():
                    self.assertEqual(files[name], expected)
                self.assertEqual(m["settings"], value["options"])
                self.assertEqual(m["result"], parse_result(files["stdout.txt"], files, version, value))
                comparable = {k: v for k, v in m["result"].items() if k not in ("version", "build_date")}
                if m["case"] in results:
                    self.assertEqual(results[m["case"]], comparable)
                results[m["case"]] = comparable
        self.assertEqual(set(found), {"46.2", "46.3"})
        for names in found.values():
            self.assertEqual(names, set(cases()))

    def test_wrong_settings_versions_and_role_encodings_fail(self):
        name = "linear-tree-loocv-default"
        files, value = self.fixture(name), cases()[name]
        with self.assertRaisesRegex(ValueError, "version"):
            parse_result(files["stdout.txt"], files, "46.2", value)
        for before, after in [("Minimum node sample size: 10", "Minimum node sample size: 2"),
                              ("Maximum number of split levels: 1", "Maximum number of split levels: 2"),
                              ("with v = 40", "with v = 4"),
                              ("Number of SE's for pruned tree: 0.2500", "Number of SE's for pruned tree: 100.00"),
                              ("Split values for N and S variables based on exhaustive search", "Quantile cutpoint search"),
                              ("No truncation of predicted values", "Node range truncation")]:
            changed = files["report.out"].replace(before, after)
            self.assertNotEqual(changed, files["report.out"])
            with self.subTest(setting=before), self.assertRaises(ValueError):
                parse_result(files["stdout.txt"], {**files, "report.out": changed}, "46.3", value)
        name = "linear-tree-split-and-fit"
        files = self.fixture(name)
        changed = files["splits.txt"].replace("1 n x", "1 s x")
        self.assertNotEqual(changed, files["splits.txt"])
        with self.assertRaisesRegex(ValueError, "role"):
            parse_result(files["stdout.txt"], {**files, "splits.txt": changed}, "46.3", cases()[name])

    def test_incomplete_and_conflicting_coefficients_fail(self):
        name = "linear-tree-split-only"
        files, value = self.fixture(name), cases()[name]
        table = files["coefficients.txt"]
        for changed in ["\n".join(table.splitlines()[:-1]), table+table.splitlines()[1]+"\n",
                        table.replace("1.9900", "9.9900"), table.replace(" z\n", " unknown\n")]:
            self.assertNotEqual(changed, table)
            with self.assertRaises(ValueError):
                parse_result(files["stdout.txt"], {**files, "coefficients.txt": changed}, "46.3", value)
        with self.assertRaises(ValueError):
            parse_result(files["stdout.txt"], {**files, "predict.R": "partial"}, "46.3", value)

    def test_failure_keeps_diagnostics_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"test double; never executed")
            for i, failure in enumerate([subprocess.CompletedProcess([], -9, "partial", "failed"),
                                        subprocess.TimeoutExpired([], 30, output=b"partial")]):
                output = root / str(i)
                with patch("capture_reference_core.subprocess.run") as run:
                    if isinstance(failure, Exception): run.side_effect = failure
                    else: run.return_value = failure
                    result = capture(binary, "46.3", "https://example.invalid", cases()["linear-tree-split-only"], output)
                self.assertFalse(result["completed"])
                self.assertNotIn("result", result)
                before = (output / "manifest.json").read_bytes()
                with patch("capture_reference_core.subprocess.run") as run:
                    with self.assertRaises(FileExistsError):
                        capture(binary, "46.3", "unused", cases()["linear-tree-split-only"], output)
                    run.assert_not_called()
                self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_missing_fitting_values_require_consistent_flags_and_coefficients(self):
        name = "linear-multilevel-weighted-fit-missing"
        files, value = self.fixture(name), cases()[name]
        self.assertIn("z.NA", files["coefficients.txt"])
        changed = files["coefficients.txt"].replace(" z.NA", " unknown.NA")
        with self.assertRaisesRegex(ValueError, "names"):
            parse_result(files["stdout.txt"], {**files, "coefficients.txt": changed}, "46.3", value)
        rows = files["coefficients.txt"].splitlines()
        fields = rows[1].split()
        fields[-1] = "1000.0"
        rows[1] = " ".join(fields)
        with self.assertRaisesRegex(ValueError, "disagree"):
            parse_result(files["stdout.txt"], {**files, "coefficients.txt": "\n".join(rows)}, "46.3", value)

    def test_independent_oracle_provenance_and_heldout_error_arithmetic(self):
        path = ROOT / "tests/fixtures/linear-tree/diagnostics/multilevel-cv-oracle.json"
        record = json.loads(path.read_text())
        self.assertTrue(record["completed"])
        self.assertEqual(record["script_sha256"], hashlib.sha256((ROOT / "tools/multilevel_linear_oracle.R").read_bytes()).hexdigest())
        for name, evidence in record["cases"].items():
            with self.subTest(case=name):
                data = cases()[name]["dataset"]
                n = len(data["rows"])
                weights = data["weights"] or [1.0]*n
                for version in ("46.3", "46.2"):
                    directory = ROOT / "tests/fixtures/linear-tree" / f"guide-{version}" / name
                    for file, digest in evidence["input_sha256"].items():
                        self.assertEqual(hashlib.sha256((directory / file).read_bytes()).hexdigest(), digest)
                    native = json.loads((directory / "manifest.json").read_text())["result"]
                    if "pruning_path" in native:
                        for actual, expected in zip(evidence["scores"], native["pruning_path"], strict=True):
                            self.assertEqual(actual["leaves"], expected["leaves"])
                            self.assertEqual(actual["pruned_node"], expected["pruned_node"])
                            self.assertLessEqual(abs(actual["alpha"]-expected["alpha"]),expected["alpha_tolerance"])
                    else:
                        self.assertEqual(evidence["correct_selected_leaves"], 4)
                        self.assertEqual(evidence["native_scan_selected_leaves"], 3)
                        self.assertEqual(native["leaf_count"], 3)
                        for actual, expected in zip(evidence["scores"], native["pruning_scores"], strict=True):
                            for field, simulated in [("mean_loss","native_scan_mean"),("standard_error","native_scan_se")]:
                                self.assertLessEqual(abs(actual[simulated]-expected[field]),expected[field+"_tolerance"])
                for matrix, mean_key, se_key in [("heldout_predictions","mean_loss","standard_error"),
                                                 ("native_scan_heldout_predictions","native_scan_mean","native_scan_se")]:
                    for j, score in enumerate(evidence["scores"],1):
                        errors = [n*w/sum(weights)*(y-row[f"V{j}"])**2
                                  for y,w,row in zip(data["targets"],weights,evidence[matrix],strict=True)]
                        mean = sum(errors)/n
                        se = math.sqrt(sum((v-mean)**2 for v in errors))/n
                        self.assertAlmostEqual(mean,score[mean_key],places=9)
                        self.assertAlmostEqual(se,score[se_key],places=9)

    def test_independent_oracle_cannot_overwrite_evidence(self):
        path = ROOT / "tests/fixtures/linear-tree/diagnostics/multilevel-cv-oracle.json"
        before = path.read_bytes()
        with patch("capture_multilevel_cv_oracle.compute") as run:
            with self.assertRaises(FileExistsError):
                capture_oracle(path)
            run.assert_not_called()
        self.assertEqual(path.read_bytes(),before)


if __name__ == "__main__":
    unittest.main()
