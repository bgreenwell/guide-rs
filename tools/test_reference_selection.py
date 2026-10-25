"""Selection evidence integrity, exact zero/weak contrasts and parser rejection."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from capture_reference_selection import capture,inputs,parse_result
from reference_selection_cases import cases

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"tests/fixtures/selection"


class SelectionCaptureTests(unittest.TestCase):
    def files(self,name):
        p=BASE/"guide-46.3"/("selection-"+name)
        m=json.loads((p/"manifest.json").read_text())
        return {n:(p/n).read_text() for n in m["file_sha256"]}

    def test_pinned_hashes_inputs_versions_and_results(self):
        metadata=json.loads((ROOT/"docs/reference-upstream.json").read_text());found={};results={}
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
        legacy=BASE/"diagnostics"
        record=json.loads((legacy/"legacy-reference8-provenance.json").read_text())
        self.assertEqual(hashlib.sha256((legacy/"legacy-reference8.json").read_bytes()).hexdigest(),record["model_sha256"])
        source=ROOT/"tests/fixtures/categorical-linear/guide-46.3"/record["source_case"]
        for name,digest in record["input_sha256"].items():self.assertEqual(hashlib.sha256((source/name).read_bytes()).hexdigest(),digest)

    def test_versions_interaction_settings_and_incomplete_predictions_are_rejected(self):
        f=self.files("no-interactions-k11");v=cases()["selection-no-interactions-k11"]
        with self.assertRaisesRegex(ValueError,"version"):parse_result(f["stdout.txt"],f,"46.2",v)
        report=f["report.out"].replace("No nodewise interaction tests","Nodewise interaction tests on all variables")
        self.assertNotEqual(report,f["report.out"])
        with self.assertRaisesRegex(ValueError,"interaction"):parse_result(f["stdout.txt"],{**f,"report.out":report},"46.3",v)
        with self.assertRaises(ValueError):parse_result(f["stdout.txt"],{**f,"predictions.txt":"\n".join(f["predictions.txt"].splitlines()[:-1])},"46.3",v)

    def test_zero_is_distinct_from_a_positive_score_with_p_above_099(self):
        for name,value in cases().items():
            if value["expected_root_association"]=="interaction":continue
            data=value["dataset"];counts={}
            # These deliberately orthogonal designs have a zero root fit;
            # positive/nonpositive target signs give its residual-sign table.
            for row,y in zip(data["rows"],data["targets"],strict=True):
                count=counts.setdefault(row[0],[0,0]);count[y>0]+=1
            total=sum(map(sum,counts.values()));positive=sum(c[1] for c in counts.values())
            chi=0
            for c in counts.values():
                for label in [0,1]:
                    expected=sum(c)*(positive if label else total-positive)/total
                    chi+=(c[label]-expected)**2/expected
            self.assertEqual(chi==0,value["expected_root_association"]=="zero")
            if "near-one" in name:
                self.assertGreater(chi,0)
                # Exact chi-square(3) upper tail, independent of Rust/R.
                probability=math.erfc(math.sqrt(chi/2))+math.sqrt(2*chi/math.pi)*math.exp(-chi/2)
                self.assertGreater(probability,0.99);self.assertLess(probability,1)
            for version in ["46.3","46.2"]:
                m=json.loads((BASE/("guide-"+version)/name/"manifest.json").read_text())
                self.assertEqual(m["result"]["leaf_count"],value["expected_root_leaves"])

    def test_failed_capture_is_diagnostic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);binary=root/"binary";binary.write_bytes(b"test double");output=root/"capture"
            v=cases()["selection-categorical-both-g2-k12"]
            with patch("capture_reference_core.subprocess.run",return_value=subprocess.CompletedProcess([],-9,"partial","failed")):
                result=capture(binary,"46.3","https://example.invalid",v,output)
            self.assertFalse(result["completed"]);self.assertNotIn("result",result)
            before=(output/"manifest.json").read_bytes()
            with patch("capture_reference_core.subprocess.run") as run:
                with self.assertRaises(FileExistsError):capture(binary,"46.3","unused",v,output)
                run.assert_not_called()
            self.assertEqual((output/"manifest.json").read_bytes(),before)


if __name__=="__main__":unittest.main()
