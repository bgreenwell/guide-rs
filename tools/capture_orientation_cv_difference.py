#!/usr/bin/env python3
"""Freeze independent categorical LOOCV and auxiliary native singleton refits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from capture_reference_orientation import capture, inputs
from reference_orientation_cases import cases

ROOT=Path(__file__).resolve().parents[1]
NAME="orientation-loocv"


def sha(raw):return hashlib.sha256(raw).hexdigest()


def capture_evidence(output):
    if output.exists():raise FileExistsError(output)
    value=cases()[NAME];script=ROOT/"tools/categorical_linear_oracle.R"
    record={"completed":False,"case":NAME,"script_sha256":sha(script.read_bytes()),"references":{}}
    with tempfile.TemporaryDirectory(prefix="guide-orientation-oracle-") as directory:
        work=Path(directory)
        source=ROOT/"tests/fixtures/orientation/guide-46.3"/NAME
        m=json.loads((source/"manifest.json").read_text())
        (work/"data.csv").write_bytes((source/"data.csv").read_bytes())
        for name,values in [("leaves.txt",m["result"]["terminal_node_ids"]),("training.txt",[int(v) for v in m["result"]["training_rows"]])]:
            (work/name).write_text("\n".join(map(str,values))+"\n")
        process=subprocess.run(["Rscript","--vanilla",str(script),"none","group","x","cv"],cwd=work,capture_output=True,text=True,check=True)
        record["r_stderr"]=process.stderr
        record["correct_scores"]=list(map(float,(work/"oracle-cv.txt").read_text().split()))
        record["correct_heldout_predictions"]=[list(map(float,line.split())) for line in (work/"oracle-heldout.txt").read_text().splitlines()]
        metadata=json.loads((ROOT/"docs/reference-upstream.json").read_text())
        for version in ["46.3","46.2"]:
            reference=next(r for r in metadata["references"] if r["version"]==version)
            binary=ROOT/".reference/upstream"/("macos-arm64" if version=="46.3" else "macos-arm64-46.2")/"guide"
            if sha(binary.read_bytes())!=reference["binary_sha256"]:raise ValueError("unpinned binary")
            source=ROOT/"tests/fixtures/orientation"/("guide-"+version)/NAME
            manifest=json.loads((source/"manifest.json").read_text())
            info={"binary_sha256":reference["binary_sha256"],"manifest_sha256":sha((source/"manifest.json").read_bytes()),
                  "native_cv_scores":manifest["result"]["pruning_scores"],"direct_refits":[]}
            # These are auxiliary diagnostic executions, not full acceptance
            # fixtures. Retain input hashes, the parsed structure and raw held
            # prediction tokens; temporary full capture files are discarded.
            for held,y in enumerate(value["dataset"]["targets"]):
                fold=copy.deepcopy(value);fold["name"]=f"orientation-held-{held}";fold["options"]["pruning"]={"kind":"none"}
                fold["dataset"]["targets"][held]=None
                destination=work/f"{version}-held-{held}"
                captured=capture(binary,version,reference["source_url"],fold,destination)
                if not captured["completed"]:raise ValueError(captured.get("error"))
                tokens=(destination/"predictions.txt").read_text().splitlines()[held+1].split()
                result=captured["result"]
                info["direct_refits"].append({"held":held,"prediction_token":tokens[-1],
                    "prediction":result["predictions"][held],"prediction_tolerance":result["prediction_tolerances"][held],
                    "leaf_count":result["leaf_count"],"splits":result["splits"],
                    "input_sha256":{n:sha(s.encode()) for n,s in inputs(fold,version).items()}})
            record["references"][version]=info
    record["completed"]=True
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(record,indent=2)+"\n")
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();record=capture_evidence(args.output)
    print("Captured independent categorical CV and direct native refits:",args.output)
