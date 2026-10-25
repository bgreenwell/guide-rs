"""Missing-value flag evidence integrity, parser rejection and the native defect."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from capture_reference_missing_flag import capture, inputs, parse_result
from reference_missing_flag_cases import cases

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tests/fixtures/missing-flag"


class MissingFlagCaptureTests(unittest.TestCase):
    def files(self, name):
        p = BASE / "guide-46.3" / ("missing-flag-" + name)
        m = json.loads((p / "manifest.json").read_text())
        return {n: (p / n).read_text() for n in m["file_sha256"]}

    def test_pinned_hashes_inputs_versions_and_results(self):
        metadata = json.loads((ROOT / "docs/reference-upstream.json").read_text())
        found, results = {}, {}
        for path in sorted(BASE.glob("guide-*/*/manifest.json")):
            with self.subTest(path=str(path)):
                m = json.loads(path.read_text())
                version = m["requested_version"]
                v = cases()[m["case"]]
                found.setdefault(version, set()).add(m["case"])
                ref = next(r for r in metadata["references"] if r["version"] == version)
                self.assertTrue(m["completed"])
                self.assertEqual(m["returncode"], 0)
                self.assertEqual(m["binary_sha256"], ref["binary_sha256"])
                self.assertEqual(m["source_url"], ref["source_url"])
                files = {}
                for name, digest in m["file_sha256"].items():
                    raw = (path.parent / name).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                    files[name] = raw.decode()
                for name, expected in inputs(v, version).items():
                    self.assertEqual(files[name], expected)
                self.assertIn("x_ m", files["data.dsc"])
                self.assertEqual(m["settings"], v["options"])
                self.assertEqual(parse_result(files["stdout.txt"], files, version, v), m["result"])
                # Both native versions agree on every parsed result.
                result = {k: x for k, x in m["result"].items() if k not in ("version", "build_date")}
                if m["case"] in results:
                    self.assertEqual(results[m["case"]], result)
                results[m["case"]] = result
        self.assertEqual(set(found), {"46.2", "46.3"})
        for names in found.values():
            self.assertEqual(names, set(cases()))

    def test_native_only_tries_code_subsets_containing_the_first_code(self):
        # Independent of Rust: brute-force every depth-1 split on x. The best
        # split over all code subsets beats native, and the best split over
        # subsets containing the first code reproduces native's loss exactly.
        def sse(values):
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            return sum((v - mean) ** 2 for v in values)

        anchored = [n for n, v in cases().items() if v["native_difference"]]
        self.assertEqual(len(anchored), 5)
        for name in anchored:
            with self.subTest(case=name):
                v = cases()[name]
                rows = [(r[0], r[1], y) for r, y in zip(v["dataset"]["rows"], v["dataset"]["targets"])
                        if y is not None]
                codes = sorted({c for x, c, _ in rows if x is None})
                observed = sorted({x for x, _, _ in rows if x is not None})
                cuts = [(a + b) / 2 for a, b in zip(observed, observed[1:])]
                subsets = [set(s) for k in range(len(codes) + 1)
                           for s in itertools.combinations(codes, k)]
                splits = []  # (loss, code subset)
                for cut in cuts:
                    for left in subsets:
                        go = [x <= cut if x is not None else c in left for x, c, _ in rows]
                        splits.append((sse([y for g, (_, _, y) in zip(go, rows) if g])
                                       + sse([y for g, (_, _, y) in zip(go, rows) if not g]), left))
                for left in subsets[1:]:
                    go = [x is None and c in left for x, c, _ in rows]
                    splits.append((sse([y for g, (_, _, y) in zip(go, rows) if g])
                                   + sse([y for g, (_, _, y) in zip(go, rows) if not g]), left))
                m = json.loads((BASE / "guide-46.3" / name / "manifest.json").read_text())
                native = sum((y - p) ** 2 for (_, _, y), p in
                             zip(rows, [p for p, t in zip(m["result"]["predictions"], m["result"]["training_rows"]) if t]))
                best = min(loss for loss, _ in splits)
                best_anchored = min(loss for loss, left in splits if codes[0] in left)
                self.assertLess(best, native - 1e-6)
                self.assertAlmostEqual(best_anchored, native, delta=1e-6 * (1 + native))

    def test_versions_flag_declaration_and_incomplete_predictions_are_rejected(self):
        f = self.files("numeric-left-A")
        v = cases()["missing-flag-numeric-left-A"]
        with self.assertRaisesRegex(ValueError, "version"):
            parse_result(f["stdout.txt"], f, "46.2", v)
        report = f["report.out"].replace("x_         m", "x_         c")
        with self.assertRaisesRegex(ValueError, "M"):
            parse_result(f["stdout.txt"], {**f, "report.out": report}, "46.3", v)
        truncated = "\n".join(f["predictions.txt"].splitlines()[:-1])
        with self.assertRaises(ValueError):
            parse_result(f["stdout.txt"], {**f, "predictions.txt": truncated}, "46.3", v)

    def test_failed_capture_is_diagnostic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"test double")
            output = root / "capture"
            v = cases()["missing-flag-numeric-left-A"]
            with patch("capture_reference_core.subprocess.run",
                       return_value=subprocess.CompletedProcess([], -9, "partial", "failed")):
                result = capture(binary, "46.3", "https://example.invalid", v, output)
            self.assertFalse(result["completed"])
            self.assertNotIn("result", result)
            before = (output / "manifest.json").read_bytes()
            with patch("capture_reference_core.subprocess.run") as run:
                with self.assertRaises(FileExistsError):
                    capture(binary, "46.3", "unused", v, output)
                run.assert_not_called()
            self.assertEqual((output / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
