#!/usr/bin/env python3
"""Capture multiple-linear tree splits, fitted leaves and LOOCV tables."""
import argparse
import json
from pathlib import Path
import re
import shlex

from capture_reference_core import capture as core_capture, parse_result as parse_core, printed_tolerance
from capture_reference_linear import number
from reference_linear_tree_cases import cases, data_files


def inputs(value, version):
    files = data_files(value)
    options = value["options"]
    pruning = options["pruning"]
    cv = pruning["kind"] == "cross_validation"
    prune_code = {"none": "2 (no pruning)", "cross_validation": "1 (CV)",
                  "fixed_leaves": "0 (fixed tree size)"}[pruning["kind"]]
    lines = ["GUIDE (multiple-linear tree acceptance)", version,
             "1 (model fitting)", '"report.out"', "1 (single tree)", "2 (regression)",
             "1 (linear)", "1 (least squares)", "1 (multiple linear)", "1 (intercept)",
             "0 (no truncation)", "1 (interaction tests)", prune_code]
    if pruning["kind"] == "fixed_leaves":
        lines += [str(pruning["max_leaves"])]
    lines += ['"data.dsc"']
    if cv:
        lines += [str(pruning["folds"]), "1 (mean-based CV)", str(pruning["standard_errors"])]
        if value["dataset"]["weights"]:
            lines += ["2 (weighted pruning errors)"]
    lines += ["2 (exhaustive cutpoints)", "2 (explicit depth)", str(options["max_depth"]),
              "2 (explicit minimum node size)", str(options["min_samples_leaf"]),
              "0 (no LaTeX)", "3 (store splits)", '"splits.txt"',
              "2 (store coefficients)", '"coefficients.txt"', "2 (store predictions)",
              '"predictions.txt"', "2 (R for training data)", '"predict.R"', "1 (root variable rank)"]
    files["input.txt"] = "\n".join(lines) + "\n"
    files["dataset.json"] = json.dumps(value["dataset"], indent=2) + "\n"
    files["options.json"] = json.dumps(options, indent=2) + "\n"
    return files


def parse_result(stdout, files, version, value):
    report = files["report.out"]
    if "Piecewise linear model" not in report or "No truncation of predicted values" not in report:
        raise ValueError("multiple-linear model or truncation settings mismatch")
    if "Split values for N and S variables based on exhaustive search" not in report:
        raise ValueError("cutpoint search settings mismatch")
    if value["options"]["pruning"]["kind"] == "none" and "No pruning" not in report:
        raise ValueError("pruning settings mismatch")
    if value["options"]["pruning"]["kind"] == "cross_validation":
        se = re.findall(r"Number of SE's for pruned tree:\s*([^\s]+)", report)
        if (len(se) != 1 or number(se[0]) != value["options"]["pruning"]["standard_errors"]
                or "Selected tree is based on mean of CV estimates" not in report):
            raise ValueError("CV standard-error settings mismatch")
    # N and S denote the same numerical split, with different fitting roles.
    # The common parser expects S; retain the original encoding as evidence.
    original = files["splits.txt"]
    normalized = re.sub(r"^(\s*\d+) n ", r"\1 s ", original, flags=re.MULTILINE)
    result = parse_core(stdout, {**files, "splits.txt": normalized}, version, value)
    for split in result["splits"]:
        source = [shlex.split(s) for s in original.splitlines() if s.strip()]
        raw = next(r for r in source if int(r[0]) == split["node_id"])
        schema = next((f for f in value["dataset"]["schema"] if f["name"] == split["feature"]), None)
        if schema is None or raw[1] != {"split": "s", "both": "n"}.get(schema["role"]):
            raise ValueError("native split encoding does not match predictor role")
        split["native_fields"] = raw[1:]
    table = [shlex.split(s) for s in files["coefficients.txt"].splitlines() if s.strip()]
    if not table or table[0][:2] != ["Node", "Constant"]:
        raise ValueError("invalid coefficient header")
    data = value["dataset"]
    allowed = {f["name"] for f in data["schema"] if f["role"] in ("fit", "both")}
    allowed.update(f'{f["name"]}.NA' for j, f in enumerate(data["schema"])
                   if f["role"] in ("fit", "both") and any(r[j] is None for r in data["rows"]))
    if set(table[0][2:]) != allowed or len(set(table[0])) != len(table[0]):
        raise ValueError("coefficient names do not match fitting roles")
    nodes = {}
    for row in table[1:]:
        if len(row) != len(table[0]) or int(row[0]) in nodes:
            raise ValueError("invalid or duplicate coefficient row")
        nodes[int(row[0])] = {name: {"value": number(token),
                                    "tolerance": printed_tolerance(token) + 1e-9}
                             for name, token in zip(table[0][1:], row[1:])}
    if set(nodes) != set(result["terminal_node_ids"]):
        raise ValueError("coefficients do not cover terminal nodes")
    # Imputations are weighted observed means within the final leaf. Reconstruct
    # predictions from printed original-unit coefficients and missing flags.
    means = {}
    for leaf in nodes:
        ids = [i for i, n in enumerate(result["terminal_node_ids"]) if n == leaf]
        for j, feature in enumerate(data["schema"]):
            if feature["name"] in nodes[leaf]:
                seen = [(data["rows"][i][j], data["weights"][i] if data["weights"] else 1.0)
                        for i in ids if data["rows"][i][j] is not None]
                if not seen:
                    raise ValueError("all-missing fitting column requires a separate fallback case")
                means[leaf, j] = sum(x*w for x,w in seen) / sum(w for _,w in seen)
    for row, leaf, prediction in zip(data["rows"], result["terminal_node_ids"], result["predictions"]):
        fit = nodes[leaf]
        estimate = fit["Constant"]["value"]
        tolerance = fit["Constant"]["tolerance"] + 5e-5 + 5e-6 * abs(prediction)
        for j, (feature, x) in enumerate(zip(data["schema"], row)):
            if feature["name"] in fit:
                if x is None:
                    x = means[leaf, j]
                estimate += fit[feature["name"]]["value"] * x
                tolerance += fit[feature["name"]]["tolerance"] * abs(x)
            flag = fit.get(feature["name"] + ".NA")
            if flag and row[j] is None:
                estimate += flag["value"]
                tolerance += flag["tolerance"]
        if abs(estimate - prediction) > tolerance:
            raise ValueError("leaf coefficients and predictions disagree")
    if (files["predict.R"].count("guide_predict <- function(){") != 1
            or files["predict.R"].count("## end of function") != 1):
        raise ValueError("incomplete native R export")
    result["leaf_coefficients"] = {str(n): v for n, v in sorted(nodes.items())}
    return result


def capture(binary, version, source_url, value, output):
    return core_capture(binary, version, source_url, value, output, input_builder=inputs,
                        result_parser=parse_result, extra_output_files=("coefficients.txt", "predict.R"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--version", choices=["46.2", "46.3"], required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--case", choices=sorted(cases()), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new fixture directory")
    m = capture(args.binary.resolve(), args.version, args.source_url, cases()[args.case], args.output)
    if not m["completed"]:
        parser.exit(1, f'Incomplete diagnostic record: {m.get("error")}\n')
    print(f"Captured GUIDE {args.version}: {args.output}")


if __name__ == "__main__":
    main()
