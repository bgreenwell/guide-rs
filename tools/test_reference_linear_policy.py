"""Pinned native fallback/truncation captures, including held-out rows."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from capture_reference_linear_policy import capture,inputs,parse_result
from capture_reference_core import printed_tolerance
from reference_linear_policy_cases import cases
from capture_linear_qr_difference import gradient,capture as capture_qr_evidence

ROOT=Path(__file__).resolve().parents[1]


class LinearPolicyCaptureTests(unittest.TestCase):
    def fixture(self,name):
        path=ROOT/"tests/fixtures/linear-policy/guide-46.3"/name
        manifest=json.loads((path/"manifest.json").read_text())
        return {n:(path/n).read_text() for n in manifest["file_sha256"]}

    def test_pinned_files_reproducible_inputs_and_versions(self):
        metadata=json.loads((ROOT/"docs/reference-upstream.json").read_text())
        found={};results={}
        for path in sorted((ROOT/"tests/fixtures/linear-policy").glob("guide-*/*/manifest.json")):
            with self.subTest(fixture=str(path)):
                m=json.loads(path.read_text());version=m["requested_version"];value=cases()[m["case"]]
                found.setdefault(version,set()).add(m["case"])
                reference=next(r for r in metadata["references"] if r["version"]==version)
                self.assertTrue(m["completed"]);self.assertEqual(m["returncode"],0)
                self.assertEqual(m["binary_sha256"],reference["binary_sha256"])
                self.assertEqual(m["source_url"],reference["source_url"])
                files={}
                for name,digest in m["file_sha256"].items():
                    raw=(path.parent/name).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),digest);files[name]=raw.decode()
                for name,expected in inputs(value,version).items():self.assertEqual(files[name],expected)
                self.assertNotIn("predict.R",files)
                self.assertEqual(m["settings"],value["options"])
                self.assertEqual(parse_result(files["stdout.txt"],files,version,value),m["result"])
                result={k:v for k,v in m["result"].items() if k not in ["version","build_date"]}
                if m["case"] in results:self.assertEqual(results[m["case"]],result)
                results[m["case"]]=result
        self.assertEqual(set(found),{"46.2","46.3"})
        for names in found.values():self.assertEqual(names,set(cases()))

    def test_wrong_versions_settings_and_incomplete_tables_fail(self):
        name="linear-limit-tree-global_range";files=self.fixture(name);value=cases()[name]
        with self.assertRaisesRegex(ValueError,"version"):parse_result(files["stdout.txt"],files,"46.2",value)
        report=files["report.out"].replace("global min. and max.","node min. and max.")
        self.assertNotEqual(report,files["report.out"])
        with self.assertRaisesRegex(ValueError,"truncation"):parse_result(files["stdout.txt"],{**files,"report.out":report},"46.3",value)
        rows=files["predictions.txt"].splitlines()
        for text in ["\n".join(rows[:-1]),files["predictions.txt"].replace("NA","0",1),
                     files["predictions.txt"].replace("     n ","     y ",1)]:
            self.assertNotEqual(text,files["predictions.txt"])
            with self.assertRaises(ValueError):parse_result(files["stdout.txt"],{**files,"predictions.txt":text},"46.3",value)

    def test_fortran_whole_decimal_precision_is_supported(self):
        self.assertEqual(printed_tolerance("42025."),0.5)
        self.assertEqual(printed_tolerance("12.E+2"),50.0)
        with self.assertRaises(ValueError):printed_tolerance("NaN")

    def test_failed_processes_are_diagnostics_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);binary=root/"binary";binary.write_bytes(b"test double")
            for i,failure in enumerate([subprocess.CompletedProcess([],-9,"partial","failed"),subprocess.TimeoutExpired([],30,output=b"partial")]):
                output=root/str(i)
                with patch("capture_reference_core.subprocess.run") as run:
                    if isinstance(failure,Exception):run.side_effect=failure
                    else:run.return_value=failure
                    result=capture(binary,"46.3","https://example.invalid",cases()["linear-policy-complete-p1-n7"],output)
                self.assertFalse(result["completed"]);self.assertNotIn("result",result)
                before=(output/"manifest.json").read_bytes()
                with patch("capture_reference_core.subprocess.run") as run:
                    with self.assertRaises(FileExistsError):capture(binary,"46.3","unused",cases()["linear-policy-complete-p1-n7"],output)
                    run.assert_not_called()
                self.assertEqual((output/"manifest.json").read_bytes(),before)

    def test_native_qr_discrepancy_has_pinned_independent_evidence(self):
        path=ROOT/"tests/fixtures/linear-policy/diagnostics/native-qr-difference.json"
        record=json.loads(path.read_text());name=record["case"]
        self.assertTrue(record["completed"])
        self.assertEqual(record["script_sha256"],hashlib.sha256((ROOT/"tools/linear_policy_oracle.R").read_bytes()).hexdigest())
        for version,reference in record["references"].items():
            fixture=ROOT/"tests/fixtures/linear-policy"/f"guide-{version}"/name
            self.assertEqual(reference["manifest_sha256"],hashlib.sha256((fixture/"manifest.json").read_bytes()).hexdigest())
            manifest=json.loads((fixture/"manifest.json").read_text())
            self.assertEqual(reference["binary_sha256"],manifest["binary_sha256"])
            self.assertEqual(record["native_predictions"],manifest["result"]["predictions"])
            for file,digest in record["input_sha256"].items():
                self.assertEqual(digest,hashlib.sha256((fixture/file).read_bytes()).hexdigest())
        data=cases()[name]["dataset"]
        for kind in ["native","correct"]:
            predictions=record[kind+"_predictions"]
            sse=sum((y-p)**2 for y,p in zip(data["targets"],predictions,strict=True))
            self.assertAlmostEqual(sse,record[kind+"_sse"],places=10)
            for term,value in gradient(data,predictions).items():
                self.assertAlmostEqual(value,record[kind+"_normal_equations"][term],places=10)
        self.assertAlmostEqual(record["native_normal_equations"]["x1.NA"],-0.5)
        self.assertLess(max(abs(v) for v in record["correct_normal_equations"].values()),1e-8)
        self.assertGreater(record["native_sse"]-record["correct_sse"],0.1)
        before=path.read_bytes()
        with self.assertRaises(FileExistsError):capture_qr_evidence(path)
        self.assertEqual(path.read_bytes(),before)


if __name__=="__main__":unittest.main()
