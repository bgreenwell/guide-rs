#!/usr/bin/env python3
"""Capture fallback/truncation behavior with native predictions, without R export."""
import argparse
import copy
import json
from pathlib import Path
import re
import shlex

from capture_reference_core import capture as core_capture, parse_result as parse_core, printed_tolerance
from capture_reference_linear import number
from reference_linear_policy_cases import cases,data_files


def inputs(value,version):
    files=data_files(value);options=value["options"];pruning=options["pruning"]
    mode=options.get("prediction_truncation","none");code={"none":0,"node_range":1,"node_range_expanded":2,"global_range":3}[mode]
    lines=["GUIDE (numeric multiple-linear policy acceptance)",version,"1",'"report.out"',"1","2","1","1","1","1",str(code),"1",
           str({"none":2,"cross_validation":1,"fixed_leaves":0}[pruning["kind"]])]
    if pruning["kind"]=="fixed_leaves":lines += [str(pruning["max_leaves"])]
    lines += ['"data.dsc"']
    if pruning["kind"]=="cross_validation":
        lines += [str(pruning["folds"]),"1",str(pruning["standard_errors"])]
        if value["dataset"]["weights"]:lines += ["2"]
    lines += ["2","2",str(options["max_depth"]),"2",str(options["min_samples_leaf"]),"0","3",'"splits.txt"',"2",'"coefficients.txt"',"2",'"predictions.txt"',"1 (skip R)","1"]
    files["input.txt"]="\n".join(lines)+"\n"
    files["dataset.json"]=json.dumps(value["dataset"],indent=2)+"\n"
    files["options.json"]=json.dumps(options,indent=2)+"\n"
    return files


def parse_result(stdout,files,version,value):
    data=value["dataset"];report=files["report.out"]
    if "Piecewise linear model" not in report:raise ValueError("not a multiple-linear fit")
    mode=value["options"].get("prediction_truncation","none")
    # Printed settings are checked independently of the supplied input stream.
    labels={"none":"No truncation of predicted values",
            "node_range":"Predictions truncated at min. and max. of D sample values in each node",
            "node_range_expanded":"Predictions truncated at 10% below min. and 10% above max. of D sample values in each node",
            "global_range":"Predictions truncated at global min. and max. of D sample values"}
    if labels[mode] not in report.split("DSC file:",1)[0]:raise ValueError("truncation setting mismatch")
    pruning=value["options"]["pruning"]
    if pruning["kind"]=="none" and "No pruning" not in report:raise ValueError("pruning settings mismatch")
    if pruning["kind"]=="fixed_leaves":
        maximum=re.findall(r"Largest tree with at most (\d+) terminal nodes",report)
        if maximum != [str(pruning["max_leaves"])]:raise ValueError("fixed-size settings mismatch")
    if pruning["kind"]=="cross_validation":
        se=re.findall(r"Number of SE's for pruned tree:\s*([^\s]+)",report)
        if len(se)!=1 or number(se[0])!=pruning["standard_errors"] or "Selected tree is based on mean of CV estimates" not in report:
            raise ValueError("CV settings mismatch")
    rows=[shlex.split(s) for s in files["predictions.txt"].splitlines() if s.strip()]
    if not rows or rows[0]!=["train","node","observed","predicted"] or len(rows)!=len(data["rows"])+1:
        raise ValueError("incomplete prediction table")
    normalized=["train node observed predicted"];flags=[];tolerances=[]
    sanitized=copy.deepcopy(value)
    for i,(row,target) in enumerate(zip(rows[1:],data["targets"],strict=True)):
        active=target is not None and (not data["weights"] or data["weights"][i]>0)
        if len(row)!=4 or row[0]!=("y" if active else "n"):raise ValueError("training membership mismatch")
        if target is None:
            if row[2]!="NA":raise ValueError("missing-response mismatch")
            sanitized["dataset"]["targets"][i]=0
        elif abs(number(row[2])-target)>printed_tolerance(row[2])+1e-12:raise ValueError("observed response mismatch")
        number(row[3]);flags.append(active);tolerances.append(printed_tolerance(row[3])+1e-9)
        normalized.append(f'y {row[1]} {sanitized["dataset"]["targets"][i]} {row[3]}')
    parsed=parse_core(stdout,{**files,"predictions.txt":"\n".join(normalized)+"\n"},version,sanitized)
    parsed["observed"]=data["targets"];parsed["training_rows"]=flags;parsed["prediction_tolerances"]=tolerances
    active_count=sum(flags)
    counts=re.findall(r"Number of cases used for training:\s*(\d+)",report)
    if counts!=[str(active_count)]:raise ValueError("active training count mismatch")
    parsed["active_count"]=active_count
    parsed["expected_fit"]=value["expected_fit"]
    if "nominal_minimum" in value:parsed["nominal_minimum"]=value["nominal_minimum"]
    if value["expected_fit"]=="constant":
        total=sum((data["weights"][i] if data["weights"] else 1) for i,yes in enumerate(flags) if yes)
        mean=sum(data["targets"][i]*(data["weights"][i] if data["weights"] else 1)/total for i,yes in enumerate(flags) if yes)
        if parsed["leaf_count"]!=1 or any(abs(v-mean)>t for v,t in zip(parsed["predictions"],tolerances)):
            raise ValueError("expected constant fallback not observed")
    if "pruning_scores" in parsed:
        # The exact affine examples have floating-point residuals near zero.
        for score in parsed["pruning_scores"]:
            for field in ["mean_loss","standard_error"]:
                score[field+"_tolerance"]=max(score[field+"_tolerance"],1e-18)
    return parsed


def capture(binary,version,source_url,value,output):
    return core_capture(binary,version,source_url,value,output,input_builder=inputs,result_parser=parse_result,
                        extra_output_files=("coefficients.txt",))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--version",choices=["46.2","46.3"],required=True)
    parser.add_argument("--source-url",required=True);parser.add_argument("--case",choices=sorted(cases()),required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error("output exists; choose a new directory")
    m=capture(args.binary.resolve(),args.version,args.source_url,cases()[args.case],args.output)
    if not m["completed"]:parser.exit(1,f'Incomplete diagnostic record: {m.get("error")}\n')
    print(f"Captured GUIDE {args.version}: {args.output}")
