#!/usr/bin/env python3
"""Capture constant-node GUIDE trees with missing-value flag (M) variables.

GUIDE's split file does not encode flag codes (categorical flag splits are
written inconsistently), so structure is taken from per-row terminal nodes and
predictions, and the report's tree text is retained for review.
"""
import argparse
import math
from pathlib import Path
import re
import shlex

from capture_reference_core import capture as core_capture, inputs as core_inputs
from reference_missing_flag_cases import cases


def inputs(value, version):
    files = core_inputs(value, version)
    flags = {f["name"] for f in value["dataset"]["schema"] if f["role"] == "flag"}
    lines = []
    for line in files["data.dsc"].splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in flags:
            parts[2] = "m"
        lines.append(" ".join(parts))
    files["data.dsc"] = "\n".join(lines) + "\n"
    files["input.txt"] = files["input.txt"].replace(
        "GUIDE (46.x constant-node reference)", "GUIDE (missing-value flag acceptance)", 1)
    return files


def parse_result(stdout, files, version, value):
    banner = re.search(r"Version ([\d.]+) \(Build date: ([^)]+)\)", stdout)
    if banner is None or banner[1] != version:
        raise ValueError("runtime version does not match the requested reference version")
    report = files["report.out"]
    # Numeric parents print a levels note; categorical parents print a count.
    if ("Levels of M variables are for missing values in associated variables" not in report
            and "Number of M variables associated with C variables: 1" not in report):
        raise ValueError("flag variable not recognized")
    if not re.search(r"^\s+2\s+x_\s+m\b", report, re.MULTILINE):
        raise ValueError("flag column not declared as M")
    leaves = re.findall(r"Number of terminal nodes of final tree:\s*(\d+)", report)
    if len(leaves) != 1:
        raise ValueError("missing or ambiguous final tree size")
    tree = re.search(r"(?:Regression|Classification) tree:\n(.*?)\n \*{10}", report, re.DOTALL)
    if tree is None:
        raise ValueError("missing tree text")
    data = value["dataset"]
    classes = sorted({t for t in data["targets"] if t is not None}) if value["classification"] else []
    lines = [shlex.split(s) for s in files["predictions.txt"].splitlines() if s.strip()]
    if (not lines or lines[0][:4] != ["train", "node", "observed", "predicted"]
            or lines[0][4:] != [f"P({c})" for c in classes]
            or len(lines) != len(data["rows"]) + 1):
        raise ValueError("unexpected prediction file")
    nodes, predictions, probabilities, training = [], [], [], []
    for row, target in zip(lines[1:], data["targets"]):
        active = target is not None
        if len(row) != len(lines[0]) or row[0] != ("y" if active else "n"):
            raise ValueError("training membership mismatch")
        if active:
            observed = row[2] if classes else float(row[2])
            if observed != target and not (not classes and math.isclose(observed, target, abs_tol=5e-5)):
                raise ValueError("observed response does not match the input row")
        nodes.append(int(row[1]))
        training.append(active)
        if classes:
            if row[3] not in classes:
                raise ValueError("unknown predicted class")
            predictions.append(row[3])
            probabilities.append([float(p) for p in row[4:]])
        else:
            predictions.append(float(row[3]))
            if not math.isfinite(predictions[-1]):
                raise ValueError("nonfinite prediction")
    if len(set(n for n, t in zip(nodes, training) if t)) != int(leaves[0]):
        raise ValueError("terminal nodes disagree with the final tree size")
    return {"version": banner[1], "build_date": banner[2], "classes": classes,
            "leaf_count": int(leaves[0]), "tree_text": tree[1].strip("\n"),
            "nodes": nodes, "predictions": predictions, "probabilities": probabilities,
            "training_rows": training, "native_difference": value["native_difference"]}


def capture(binary, version, source_url, value, output):
    return core_capture(binary, version, source_url, value, output,
                        input_builder=inputs, result_parser=parse_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--version", choices=["46.2", "46.3"], required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--case", choices=sorted(cases()), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new directory")
    m = capture(args.binary.resolve(), args.version, args.source_url, cases()[args.case], args.output)
    if not m["completed"]:
        parser.exit(1, f'Incomplete diagnostic record: {m.get("error")}\n')
    print(f"Captured GUIDE {args.version}: {args.output}")
