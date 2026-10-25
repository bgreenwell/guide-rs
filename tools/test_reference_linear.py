"""Integrity and strict completion checks for native linear leaf captures."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from capture_reference_linear import capture, inputs, parse_result
from reference_linear_cases import cases

ROOT = Path(__file__).resolve().parents[1]


class LinearCaptureTests(unittest.TestCase):
    def fixture(self, name="ls-linear-complete"):
        directory = ROOT / "tests/fixtures/linear/guide-46.3" / name
        m = json.loads((directory / "manifest.json").read_text())
        return {n: (directory / n).read_text() for n in m["file_sha256"]}

    def test_pinned_binaries_reproducible_inputs_and_parsed_results(self):
        metadata = json.loads((ROOT / "docs/reference-upstream.json").read_text())
        found, results = {}, {}
        for path in sorted((ROOT / "tests/fixtures/linear").glob("guide-*/*/manifest.json")):
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
                    self.assertEqual(comparable, results[m["case"]])
                results[m["case"]] = comparable
        self.assertEqual(set(found), {"46.2", "46.3"})
        for names in found.values():
            self.assertEqual(names, set(cases()))

    def test_bad_settings_versions_and_completion_are_rejected(self):
        files = self.fixture("ls-stepwise-extreme-tails")
        value = cases()["ls-stepwise-extreme-tails"]
        with self.assertRaisesRegex(ValueError, "version"):
            parse_result(files["stdout.txt"], files, "46.2", value)
        for original, changed in [("Maximum number of split levels: 1", "Maximum number of split levels: 2"),
                                  ("Maximum number of variables in subset: 1", "Maximum number of variables in subset: 2"),
                                  ("No truncation of predicted values", "Node range truncation"),
                                  ("F-alpha for stepwise variable selection: .050", "F-alpha for stepwise variable selection: .500")]:
            with self.subTest(setting=original):
                report = files["report.out"].replace(original, changed)
                self.assertNotEqual(report, files["report.out"])
                with self.assertRaises(ValueError):
                    parse_result(files["stdout.txt"], {**files, "report.out": report}, "46.3", value)
        with self.assertRaises(ValueError):
            parse_result("incomplete", files, "46.3", value)

    def test_corrupt_prediction_and_coefficient_tables_are_rejected(self):
        files = self.fixture()
        value = cases()["ls-linear-complete"]
        for key, changed in [("predictions.txt", "\n".join(files["predictions.txt"].splitlines()[:-1])),
                             ("coefficients.txt", files["coefficients.txt"].replace("2.9746", "200.00")),
                             ("predict.R", files["predict.R"].replace("2.00150590151", "3.00150590151")),
                             ("predict.R", files["predict.R"].replace("2.97460317460", "NaN")),
                             ("splits.txt", "1 n x 1.0 n\n")]:
            self.assertNotEqual(changed, files[key])
            with self.subTest(file=key), self.assertRaises(ValueError):
                parse_result(files["stdout.txt"], {**files, key: changed}, "46.3", value)

    def test_failure_is_a_diagnostic_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / "binary"
            binary.write_bytes(b"test double; never executed")
            for i, failure in enumerate([subprocess.CompletedProcess([], -9, "partial", "failure"),
                                        subprocess.TimeoutExpired([], 30, output=b"partial")]):
                output = directory / str(i)
                with patch("capture_reference_linear.subprocess.run") as run:
                    if isinstance(failure, Exception):
                        run.side_effect = failure
                    else:
                        run.return_value = failure
                    m = capture(binary, "46.3", "https://example.invalid", cases()["ls-linear-complete"], output)
                self.assertFalse(m["completed"])
                self.assertNotIn("result", m)
                before = (output / "manifest.json").read_bytes()
                with patch("capture_reference_linear.subprocess.run") as run:
                    with self.assertRaises(FileExistsError):
                        capture(binary, "46.3", "unused", cases()["ls-linear-complete"], output)
                    run.assert_not_called()
                self.assertEqual((output / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
