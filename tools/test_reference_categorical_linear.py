"""Integrity and parser rejection checks for native categorical linear evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from capture_reference_categorical_linear import capture, inputs, parse_result
from reference_categorical_linear_cases import cases

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"tests/fixtures/categorical-linear"


class CategoricalLinearCaptureTests(unittest.TestCase):
    def files(self,name):
        path=BASE/"guide-46.3"/("categorical-linear-"+name)
        m=json.loads((path/"manifest.json").read_text())
        return {n:(path/n).read_text() for n in m["file_sha256"]}

    def test_hashes_inputs_pinned_versions_and_equal_results(self):
        metadata=json.loads((ROOT/"docs/reference-upstream.json").read_text())
        found={};results={}
        for path in sorted(BASE.glob("guide-*/*/manifest.json")):
            with self.subTest(path=str(path)):
                m=json.loads(path.read_text());version=m["requested_version"];v=cases()[m["case"]]
                found.setdefault(version,set()).add(m["case"])
                ref=next(r for r in metadata["references"] if r["version"]==version)
                self.assertTrue(m["completed"]);self.assertEqual(m["returncode"],0)
                self.assertEqual(m["binary_sha256"],ref["binary_sha256"]);self.assertEqual(m["source_url"],ref["source_url"])
                files={}
                for name,digest in m["file_sha256"].items():
                    raw=(path.parent/name).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),digest);files[name]=raw.decode()
                for name,expected in inputs(v,version).items():self.assertEqual(files[name],expected)
                self.assertNotIn("predict.R",files);self.assertEqual(m["settings"],v["options"])
                self.assertEqual(parse_result(files["stdout.txt"],files,version,v),m["result"])
                result={k:v for k,v in m["result"].items() if k not in ("version","build_date")}
                if m["case"] in results:self.assertEqual(results[m["case"]],result)
                results[m["case"]]=result
        self.assertEqual(set(found),{"46.2","46.3"})
        for names in found.values():self.assertEqual(names,set(cases()))

    def test_bad_versions_truncated_predictions_and_incomplete_category_splits_fail(self):
        f=self.files("split-and-fit-skewed");v=cases()["categorical-linear-split-and-fit-skewed"]
        with self.assertRaisesRegex(ValueError,"version"):parse_result(f["stdout.txt"],f,"46.2",v)
        with self.assertRaises(ValueError):parse_result(f["stdout.txt"],{**f,"predictions.txt":"\n".join(f["predictions.txt"].splitlines()[:-1])},"46.3",v)
        lines=f["splits.txt"].splitlines()
        with self.assertRaisesRegex(ValueError,"categorical"):parse_result(f["stdout.txt"],{**f,"splits.txt":"\n".join(lines[1:])},"46.3",v)
        # The shared parser also accepts an upstream encoding that retains role B.
        b={**f,"splits.txt":f["splits.txt"].replace(" c group "," b group ")}
        parsed=parse_result(b["stdout.txt"],b,"46.3",v)
        self.assertEqual(parsed["splits"][0]["native_fields"][0],"b")

    def test_failed_capture_is_diagnostic_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);binary=root/"binary";binary.write_bytes(b"test double");out=root/"capture"
            v=cases()["categorical-linear-reference"]
            with patch("capture_reference_core.subprocess.run",return_value=subprocess.CompletedProcess([],-9,"partial","failed")):
                m=capture(binary,"46.3","https://example.invalid",v,out)
            self.assertFalse(m["completed"]);self.assertNotIn("result",m)
            before=(out/"manifest.json").read_bytes()
            with patch("capture_reference_core.subprocess.run") as run:
                with self.assertRaises(FileExistsError):capture(binary,"46.3","unused",v,out)
                run.assert_not_called()
            self.assertEqual((out/"manifest.json").read_bytes(),before)

    def test_balanced_sign_stop_and_strong_signal_control_are_explicit(self):
        for version in ["46.3","46.2"]:
            path=BASE/("guide-"+version)/"categorical-linear-split-and-fit"
            m=json.loads((path/"manifest.json").read_text());d=json.loads((path/"dataset.json").read_text())
            self.assertEqual(m["result"]["leaf_count"],1)
            self.assertAlmostEqual(sum((y-p)**2 for y,p in zip(d["targets"],m["result"]["predictions"])),28750)
            # Independent intercepts/slope and balanced signs, not Rust output.
            expected=[28.75+6*(i//24)-(57.5 if i//24>=2 else 0)-0.5*(i%24) for i in range(96)]
            self.assertEqual(m["result"]["predictions"],expected)
            for g in range(4):self.assertEqual(sum(d["targets"][i]>expected[i] for i in range(g*24,(g+1)*24)),12)
            control=json.loads((BASE/("guide-"+version)/"categorical-linear-split-and-fit-skewed/manifest.json").read_text())
            self.assertEqual(control["result"]["leaf_count"],2)


if __name__=="__main__":unittest.main()
