#!/usr/bin/env python3
"""Capture multiple-linear residual-sign selection boundaries with pinned GUIDE."""
import argparse
from pathlib import Path
from capture_reference_categorical_linear import inputs as linear_inputs
from capture_reference_linear_policy import parse_result as parse_linear
from capture_reference_core import capture as core_capture
from reference_selection_cases import cases


def inputs(value,version):
    files=linear_inputs(value,version)
    lines=files["input.txt"].splitlines()
    lines[0]="GUIDE (residual-sign selection acceptance)"
    lines[11]="1" if value["options"]["interactions"] else "2"
    files["input.txt"]="\n".join(lines)+"\n"
    return files


def parse_result(stdout,files,version,value):
    report=files["report.out"]
    expected="Nodewise interaction tests on all variables" if value["options"]["interactions"] else "No nodewise interaction tests"
    if expected not in report:raise ValueError("interaction setting mismatch")
    result=parse_linear(stdout,files,version,value)
    if result["leaf_count"]!=value["expected_root_leaves"]:raise ValueError("unexpected selection boundary outcome")
    result["expected_root_association"]=value["expected_root_association"]
    return result


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
