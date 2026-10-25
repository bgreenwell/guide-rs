"""Integrity and strict parsing for frozen native importance reports."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from capture_reference_importance import capture, inputs, parse_result
from reference_importance_cases import cases

ROOT = Path(__file__).resolve().parents[1]


class ImportanceCaptureTests(unittest.TestCase):
    def fixture(self, name="ls-importance-duplicates"):
        path = ROOT / "tests/fixtures/importance/guide-46.3" / name
        m = json.loads((path / "manifest.json").read_text())
        return {n: (path / n).read_text() for n in m["file_sha256"]}

    def test_all_captures_have_pinned_binaries_reproducible_inputs_and_valid_reports(self):
        metadata = json.loads((ROOT / "docs/reference-upstream.json").read_text())
        found = {}
        for path in sorted((ROOT / "tests/fixtures/importance").glob("guide-*/*/manifest.json")):
            with self.subTest(fixture=str(path)):
                m = json.loads(path.read_text())
                version = m["requested_version"]
                value = cases()[m["case"]]
                found.setdefault(version, set()).add(m["case"])
                self.assertTrue(m["completed"])
                self.assertEqual(m["returncode"], 0)
                native = next(r for r in metadata["references"] if r["version"] == version)
                self.assertEqual(m["binary_sha256"], native["binary_sha256"])
                files = {}
                for name, digest in m["file_sha256"].items():
                    raw = (path.parent / name).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                    files[name] = raw.decode()
                for name, expected in inputs(value, version).items():
                    self.assertEqual(files[name], expected)
                self.assertEqual(m["settings"], value["options"])
                self.assertEqual(m["importance_options"], value["importance_options"])
                self.assertEqual(m["independent_raw_statistics"], value["expected_raw_statistics"])
                self.assertEqual(m["result"], parse_result(files["stdout.txt"], files, version, value))
        self.assertEqual(set(found), {"46.2", "46.3"})
        for names in found.values():
            self.assertEqual(names, set(cases()))

    def test_wrong_versions_settings_and_bad_tables_are_rejected(self):
        files = self.fixture()
        value = cases()["ls-importance-duplicates"]
        with self.assertRaisesRegex(ValueError, "version"):
            parse_result(files["stdout.txt"], files, "46.2", value)
        for original, changed in [("Maximum number of split levels: 1", "Maximum number of split levels: 4"),
                                  ("Using 300 permutations", "Using 30 permutations")]:
            report = files["report.out"].replace(original, changed)
            self.assertNotEqual(report, files["report.out"])
            with self.assertRaisesRegex(ValueError, "settings"):
                parse_result(files["stdout.txt"], {**files, "report.out": report}, "46.3", value)
        lines = files["scores.txt"].splitlines()
        for text in ["broken", "\n".join(lines[:-1]), files["scores.txt"] + lines[1] + "\n",
                     files["scores.txt"].replace("signal", "unknown"),
                     files["scores.txt"].replace("1.258E+01", "NaN")]:
            with self.subTest(score_text=text):
                with self.assertRaises(ValueError):
                    parse_result(files["stdout.txt"], {**files, "scores.txt": text}, "46.3", value)

    def test_thresholds_and_completion_evidence_are_required(self):
        files = self.fixture()
        value = cases()["ls-importance-duplicates"]
        line = next(s for s in files["report.out"].splitlines() if s.strip().startswith("95% threshold"))
        for report in [files["report.out"].replace(line, ""), files["report.out"] + line + "\n",
                       files["report.out"].replace(line, " 95% threshold is NaN")]:
            with self.assertRaises(ValueError):
                parse_result(files["stdout.txt"], {**files, "report.out": report}, "46.3", value)
        with self.assertRaises(ValueError):
            parse_result(files["stdout.txt"].replace("Importance scores are stored in scores.txt", "incomplete"), files, "46.3", value)

    def test_failed_processes_are_diagnostics_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / "binary"
            binary.write_bytes(b"test double; never executed")
            for i, failure in enumerate([subprocess.CompletedProcess([], -9, "partial", "failure"),
                                        subprocess.TimeoutExpired([], 30, output=b"partial")]):
                output = directory / str(i)
                with patch("capture_reference_importance.subprocess.run") as run:
                    if isinstance(failure, Exception): run.side_effect = failure
                    else: run.return_value = failure
                    m = capture(binary, "46.3", "https://example.invalid", cases()["ls-importance-duplicates"], output)
                self.assertFalse(m["completed"])
                self.assertNotIn("result", m)
                before = (output / "manifest.json").read_bytes()
                with patch("capture_reference_importance.subprocess.run") as run:
                    with self.assertRaises(FileExistsError):
                        capture(binary, "46.3", "unused", cases()["ls-importance-duplicates"], output)
                    run.assert_not_called()
                self.assertEqual((output / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
