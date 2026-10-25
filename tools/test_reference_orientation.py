"""Pinned orientation evidence, held-out routes, and categorical CV discrepancy."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from capture_reference_orientation import capture,inputs,parse_result
from capture_orientation_cv_difference import capture_evidence
from reference_orientation_cases import cases

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"tests/fixtures/orientation"
def sha(raw):return hashlib.sha256(raw).hexdigest()


class OrientationCaptureTests(unittest.TestCase):
    def files(self,name):
        path=BASE/"guide-46.3"/name;m=json.loads((path/"manifest.json").read_text())
        return {n:(path/n).read_text() for n in m["file_sha256"]}

    def test_pinned_inputs_hashes_and_versions_agree(self):
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
                    raw=(path.parent/name).read_bytes();self.assertEqual(sha(raw),digest);files[name]=raw.decode()
                for name,expected in inputs(v,version).items():self.assertEqual(files[name],expected)
                self.assertNotIn("predict.R",files);self.assertEqual(m["settings"],v["options"])
                self.assertEqual(parse_result(files["stdout.txt"],files,version,v),m["result"])
                result={k:v for k,v in m["result"].items() if k not in ("version","build_date")}
                if m["case"] in results:self.assertEqual(results[m["case"]],result)
                results[m["case"]]=result
        self.assertEqual(set(found),{"46.2","46.3"})
        for names in found.values():self.assertEqual(names,set(cases()))
        legacy=BASE/"diagnostics";r=json.loads((legacy/"legacy-reference9-provenance.json").read_text())
        self.assertEqual(sha((legacy/"legacy-reference9.json").read_bytes()),r["model_sha256"])
        source=ROOT/"tests/fixtures/selection/guide-46.3"/r["source_case"]
        for name,digest in r["input_sha256"].items():self.assertEqual(sha((source/name).read_bytes()),digest)

    def test_bad_versions_missing_responses_and_incomplete_predictions_fail(self):
        name="orientation-missing-lower-p1-both";f=self.files(name);v=cases()[name]
        with self.assertRaisesRegex(ValueError,"version"):parse_result(f["stdout.txt"],f,"46.2",v)
        with self.assertRaises(ValueError):parse_result(f["stdout.txt"],{**f,"predictions.txt":"\n".join(f["predictions.txt"].splitlines()[:-1])},"46.3",v)
        text=f["predictions.txt"].replace("               NA", "                0", 1)
        self.assertNotEqual(text,f["predictions.txt"])
        with self.assertRaises(ValueError):parse_result(f["stdout.txt"],{**f,"predictions.txt":text},"46.3",v)

    def test_first_active_level_anchors_each_local_split(self):
        def descendant(terminal,node):
            while terminal>node:terminal//=2
            return terminal==node
        for name,value in cases().items():
            d=value["dataset"]
            m=json.loads((BASE/"guide-46.3"/name/"manifest.json").read_text());r=m["result"]
            for split in r["splits"]:
                if split["kind"]!="categorical":continue
                j=next(j for j,f in enumerate(d["schema"]) if f["name"]==split["feature"])
                ids=[i for i,active in enumerate(r["training_rows"]) if active and descendant(r["terminal_node_ids"][i],split["node_id"])]
                levels={d["rows"][i][j] for i in ids}
                first=min(levels,key=lambda s:"NA" if s is None else s)
                self.assertTrue(split["missing_left"] if first is None else first in split["left_categories"])
                self.assertFalse(split["unknown_left"])
                self.assertNotIn("!future",levels)

    def test_failure_is_diagnostic_and_capture_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);binary=root/"binary";binary.write_bytes(b"test double");out=root/"capture"
            value=cases()["orientation-complete-p3-both"]
            with patch("capture_reference_core.subprocess.run",return_value=subprocess.CompletedProcess([],-9,"partial","failed")):
                m=capture(binary,"46.3","https://example.invalid",value,out)
            self.assertFalse(m["completed"]);self.assertNotIn("result",m)
            before=(out/"manifest.json").read_bytes()
            with patch("capture_reference_core.subprocess.run") as run:
                with self.assertRaises(FileExistsError):capture(binary,"46.3","unused",value,out)
                run.assert_not_called()
            self.assertEqual((out/"manifest.json").read_bytes(),before)

    def test_cv_discrepancy_has_independent_and_direct_native_evidence(self):
        path=BASE/"diagnostics/categorical-cv-difference.json";r=json.loads(path.read_text());value=cases()[r["case"]]
        self.assertTrue(r["completed"]);self.assertEqual(r["script_sha256"],sha((ROOT/"tools/categorical_linear_oracle.R").read_bytes()))
        for column in [0,1]:
            losses=[(y-p[column])**2 for y,p in zip(value["dataset"]["targets"],r["correct_heldout_predictions"],strict=True)]
            mean=sum(losses)/len(losses);se=(sum((e-mean)**2 for e in losses)/len(losses)**2)**.5
            self.assertAlmostEqual(mean,r["correct_scores"][column],places=10);self.assertAlmostEqual(se,r["correct_scores"][column+2],places=10)
        predictions=[]
        for version,info in r["references"].items():
            manifest=BASE/("guide-"+version)/r["case"]/'manifest.json';m=json.loads(manifest.read_text())
            self.assertEqual(info["manifest_sha256"],sha(manifest.read_bytes()));self.assertEqual(info["binary_sha256"],m["binary_sha256"])
            self.assertEqual(info["native_cv_scores"],m["result"]["pruning_scores"])
            self.assertEqual(len(info["direct_refits"]),96)
            current=[]
            for i,fold in enumerate(info["direct_refits"]):
                self.assertEqual(fold["held"],i);self.assertEqual(fold["leaf_count"],2)
                self.assertEqual(float(fold["prediction_token"]),fold["prediction"])
                self.assertLessEqual(abs(fold["prediction"]-r["correct_heldout_predictions"][i][0]),fold["prediction_tolerance"])
                v=copy.deepcopy(value);v["name"]=f"orientation-held-{i}";v["options"]["pruning"]={"kind":"none"};v["dataset"]["targets"][i]=None
                self.assertEqual(fold["input_sha256"],{n:sha(s.encode()) for n,s in inputs(v,version).items()})
                current.append(fold["prediction"])
            predictions.append(current)
            self.assertGreater(info["native_cv_scores"][0]["mean_loss"]-r["correct_scores"][0],50)
        self.assertEqual(*predictions)
        before=path.read_bytes()
        with self.assertRaises(FileExistsError):capture_evidence(path)
        self.assertEqual(path.read_bytes(),before)


if __name__=="__main__":unittest.main()
