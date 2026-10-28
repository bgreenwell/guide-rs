"""Single-node LDA evidence integrity and parser rejection."""
import hashlib
import json
import re
from pathlib import Path
import unittest
from capture_reference_discriminant import inputs, parse_result
from reference_discriminant_cases import cases

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tests/fixtures/discriminant"


class DiscriminantCaptureTests(unittest.TestCase):
    def files(self, version, name):
        p = BASE / f"guide-{version}" / name
        m = json.loads((p / "manifest.json").read_text())
        return {n: (p / n).read_text() for n in m["file_sha256"]}

    def test_pinned_hashes_inputs_versions_and_results(self):
        metadata = json.loads((ROOT / "docs/reference-upstream.json").read_text())
        found = {}
        for path in sorted(BASE.glob("guide-*/*/manifest.json")):
            with self.subTest(path=str(path)):
                m = json.loads(path.read_text())
                version = m["requested_version"]
                value = cases()[m["case"]]
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
                for name, expected in inputs(value, version).items():
                    self.assertEqual(files[name], expected)
                self.assertEqual(parse_result(files["stdout.txt"], files, version, value), m["result"])
        self.assertEqual(set(found), {"46.2", "46.3"})
        for names in found.values():
            self.assertEqual(names, set(cases()))

    def test_versions_agree_on_selection_and_predictions(self):
        for name in cases():
            with self.subTest(case=name):
                a, b = (json.loads((BASE / f"guide-{v}" / name / "manifest.json").read_text())["result"]
                        for v in ("46.2", "46.3"))
                if "depth" in cases()[name]:
                    self.assertEqual(a, b)
                    continue
                self.assertEqual([c["variable"] for c in a["coefficients"]],
                                 [c["variable"] for c in b["coefficients"]])
                self.assertEqual([f["predicted"] for f in a["fitted"]], [f["predicted"] for f in b["fitted"]])

    def test_wrong_version_split_roots_and_missing_rows_are_rejected(self):
        f = self.files("46.3", "lda-leaf-three")
        value = cases()["lda-leaf-three"]
        with self.assertRaisesRegex(ValueError, "version"):
            parse_result(f["stdout.txt"], f, "46.2", value)
        split = re.sub(r"(Number of terminal nodes of final tree:\s+)1", r"\g<1>2", f["report.out"])
        self.assertNotEqual(split, f["report.out"])
        with self.assertRaisesRegex(ValueError, "single node"):
            parse_result(f["stdout.txt"], {**f, "report.out": split}, "46.3", value)
        short = "\n".join(f["fitted.txt"].splitlines()[:-1]) + "\n"
        with self.assertRaisesRegex(ValueError, "every row"):
            parse_result(f["stdout.txt"], {**f, "fitted.txt": short}, "46.3", value)

    def test_single_node_cases_cannot_split(self):
        for name, value in cases().items():
            if "depth" in value:
                # Tree cases keep every node within GUIDE's exhaustive cut range.
                self.assertLessEqual(len(value["rows"]), 2 * value["min_node"] + 101)
                continue
            with self.subTest(case=name):
                self.assertLess(len(value["rows"]), 2.5 * value["min_node"])
                self.assertLessEqual(value["min_node"], len(value["rows"]) // 2)


if __name__ == "__main__":
    unittest.main()
