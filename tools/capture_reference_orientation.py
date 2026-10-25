#!/usr/bin/env python3
"""Capture categorical-linear orientation, missingness and held-out routes."""
import argparse
from pathlib import Path
from capture_reference_categorical_linear import inputs as linear_inputs
from capture_reference_linear_policy import parse_result
from capture_reference_core import capture as core_capture
from reference_orientation_cases import cases


def inputs(value,version):
    files=linear_inputs(value,version)
    files["input.txt"]=files["input.txt"].replace("categorical multiple-linear acceptance","categorical orientation acceptance",1)
    return files


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
