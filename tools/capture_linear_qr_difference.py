#!/usr/bin/env python3
"""Freeze the independently verified four-predictor native LS discrepancy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"tools/linear_policy_oracle.R"
NAME="linear-policy-two-missing-p4-n9"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient(data,predictions):
    names=[f["name"] for f in data["schema"]]
    residual=[y-p for y,p in zip(data["targets"],predictions,strict=True)]
    result={"intercept":sum(residual)}
    for j,name in enumerate(names):
        seen=[row[j] for row in data["rows"] if row[j] is not None];mean=sum(seen)/len(seen)
        result[name]=sum((mean if row[j] is None else row[j])*r for row,r in zip(data["rows"],residual))
        if any(row[j] is None for row in data["rows"]):
            result[name+".NA"]=sum(r for row,r in zip(data["rows"],residual) if row[j] is None)
    return result


def capture(output):
    if output.exists():raise FileExistsError(f"output already exists: {output}")
    fixture=ROOT/"tests/fixtures/linear-policy/guide-46.3"/NAME
    data=json.loads((fixture/"dataset.json").read_text());manifest=json.loads((fixture/"manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="guide-qr-difference-") as directory:
        work=Path(directory);shutil.copyfile(fixture/"data.csv",work/"data.csv")
        (work/"leaves.txt").write_text("\n".join(map(str,manifest["result"]["terminal_node_ids"])))
        (work/"training.txt").write_text("\n".join("1" for _ in data["rows"]))
        subprocess.run(["Rscript","--vanilla",str(SCRIPT),"none",",".join(f["name"] for f in data["schema"]),"linear","no_cv"],
                       cwd=work,text=True,capture_output=True,timeout=30,check=True)
        predictions=list(map(float,(work/"oracle-predictions.txt").read_text().split()))
        version=(work/"oracle-version.txt").read_text().strip()
    native=manifest["result"]["predictions"]
    sse=lambda p:sum((y-v)**2 for y,v in zip(data["targets"],p,strict=True))
    native_gradient=gradient(data,native);correct_gradient=gradient(data,predictions)
    if sse(native)-sse(predictions)<0.1 or max(abs(v) for v in correct_gradient.values())>1e-8:
        raise ValueError("least-squares discrepancy not independently established")
    references={}
    for version_id in ["46.3","46.2"]:
        path=ROOT/"tests/fixtures/linear-policy"/f"guide-{version_id}"/NAME
        m=json.loads((path/"manifest.json").read_text())
        if m["result"]["predictions"]!=native:raise ValueError("native versions disagree")
        references[version_id]={"binary_sha256":m["binary_sha256"],"manifest_sha256":sha(path/"manifest.json")}
    record={"completed":True,"case":NAME,"r_version":version,"script_sha256":sha(SCRIPT),
            "input_sha256":{f:sha(fixture/f) for f in ["data.csv","dataset.json","options.json"]},
            "references":references,"native_predictions":native,"correct_predictions":predictions,
            "native_sse":sse(native),"correct_sse":sse(predictions),
            "native_normal_equations":native_gradient,"correct_normal_equations":correct_gradient}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:json.dump(record,stream,indent=2);stream.write("\n")
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();capture(args.output);print(f"Captured independent QR evidence: {args.output}")
