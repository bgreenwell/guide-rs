#!/usr/bin/env python3
"""Freeze root-only linear/stepwise GUIDE fits; existing captures are immutable.

Templates follow the 46.3 interactive writer with explicit no truncation. Generated
R is captured as native output, never executed or used by the Rust implementation.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from capture_reference_core import printed_tolerance, sha
from reference_linear_cases import cases, data_files


def inputs(value, version):
    files = data_files(value)
    stepwise = value["options"]["node_model"]["kind"] == "stepwise"
    lines = ["GUIDE (root-only numerical least-squares acceptance)", version,
             "1 (model fitting)", '"report.out"', "1 (single tree)",
             "2 (regression)", "1 (linear)", "1 (least squares)"]
    lines += (["0 (stepwise)", "1 (univariate splits)", "2 (forward and backward)",
               f'{value["options"]["node_model"]["max_variables"]} (variable limit)', "0.05 (entry alpha)"] if stepwise else
              ["1 (multiple linear)", "1 (include intercept)"])
    lines += ["0 (no truncation)", "1 (interaction tests)", "2 (no pruning)", '"data.dsc"']
    if not stepwise:
        lines += ["2 (exhaustive cutpoints)"]
    lines += ["2 (explicit depth)", "1", "2 (explicit minimum node size)", "2",
              "0 (no LaTeX)", "3 (store splits)", '"splits.txt"',
              "2 (store coefficients or selected variables)", '"coefficients.txt"',
              "2 (store predictions)", '"predictions.txt"', "2 (R for training data)", '"predict.R"']
    if not stepwise:
        lines += ["1 (root variable rank)"]
    files["input.txt"] = "\n".join(lines) + "\n"
    files["dataset.json"] = json.dumps(value["dataset"], indent=2) + "\n"
    files["options.json"] = json.dumps(value["options"], indent=2) + "\n"
    return files


def number(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("nonfinite native coefficient or prediction")
    return value


def parse_result(stdout, files, version, value):
    banner = re.search(r"Version ([\d.]+) \(Build date: ([^)]+)\)", stdout)
    if banner is None or banner[1] != version:
        raise ValueError("runtime version mismatch")
    report = files["report.out"]
    for pattern in [r"Number of terminal nodes of final tree:\s*1\s*$",
                    r"Total number of nodes of final tree:\s*1\s*$",
                    r"Maximum number of split levels:\s*1\s*$",
                    r"Minimum node sample size:\s*2\s*$"]:
        if len(re.findall(pattern, report, re.MULTILINE)) != 1:
            raise ValueError("root-only settings or final tree mismatch")
    if ("No truncation of predicted values" not in report or "No pruning" not in report
            or "Results are stored in report.out" not in stdout):
        raise ValueError("missing settings or completion evidence")
    model = value["options"]["node_model"]["kind"]
    if model == "stepwise":
        limit = value["options"]["node_model"]["max_variables"]
        expected_limit = (f"Maximum number of variables in subset: {limit}" if limit else
                          "Using as many variables as needed")
        if ("Piecewise forward and backward stepwise regression" not in report
                or "F-alpha for stepwise variable selection: .050" not in report
                or expected_limit not in report):
            raise ValueError("stepwise settings mismatch")
    elif "Piecewise linear model" not in report:
        raise ValueError("multiple linear model missing")
    if shlex.split(files["splits.txt"]) != ["1", "t", "NONE", "NONE", "t"]:
        raise ValueError("unexpected split in root-only capture")
    # Independently reconstruct the native exported function at the input rows;
    # its high-precision coefficients must reproduce the separate prediction file.
    if files["predict.R"].count("## end of function") != 1:
        raise ValueError("incomplete coefficient export")
    code = files["predict.R"].split("## end of function")[0]
    if code.count(" return(c(nodeid,predict))") != 1 or code.count("guide_predict <- function(){") != 1:
        raise ValueError("incomplete root function export")
    intercepts = re.findall(r"^ predict <- ([^\s]+)\s*$", code, re.MULTILINE)
    if len(intercepts) != 1 or code.count(" nodeid <- 1\n") != 1:
        raise ValueError("ambiguous root coefficient export")
    intercept = number(intercepts[0])
    means = dict((name, number(token)) for name, token in re.findall(
        r"^ x <- ifelse\(is.na\((\w+)\), ([^,]+), \1\)$", code, re.MULTILINE))
    names = [s["name"] for s in value["dataset"]["schema"]]
    coefficients = {}
    current = None
    for line in code.splitlines():
        if line.startswith(" x <- ifelse"):
            current = re.search(r"is.na\((\w+)\)", line)[1]
        elif line.startswith(" predict <- predict +"):
            match = re.fullmatch(r" predict <- predict \+\s+([^ ]+) \* (\w+(?:\.NA)?)", line)
            if match is None:
                raise ValueError("unrecognized coefficient export")
            feature = current if match[2] == "x" else match[2]
            if feature is None or feature in coefficients or feature.removesuffix(".NA") not in names:
                raise ValueError("unknown or duplicate coefficient")
            coefficients[feature] = number(match[1])
    if set(means) - set(names):
        raise ValueError("unknown imputation variable")
    stored = [shlex.split(s) for s in files["coefficients.txt"].splitlines() if s.strip()]
    if len(stored) != 2 or stored[1][0] != "1":
        raise ValueError("incomplete coefficient/variable table")
    if model == "stepwise":
        if (stored[0][:3] != ["node", "lower", "upper"]
                # GUIDE prints rounded DBL_MAX sentinels that exceed f64::MAX.
                or len(stored[1]) < 3 or stored[1][1:3] != ["-0.17977E+309", "0.17977E+309"]
                or sorted(stored[1][3:]) != sorted(coefficients)):
            raise ValueError("selected variables or truncation disagree with export")
    else:
        if stored[0][:2] != ["Node", "Constant"] or len(stored[0]) != len(stored[1]):
            raise ValueError("invalid coefficient table")
        for name, token in zip(stored[0][1:], stored[1][1:]):
            reference = intercept if name == "Constant" else coefficients.get(name, 0.0)
            if name != "Constant" and name.removesuffix(".NA") not in names:
                raise ValueError("unknown coefficient table variable")
            if abs(number(token) - reference) > printed_tolerance(token) + 1e-9:
                raise ValueError("stored coefficients disagree with export")
    table = [shlex.split(s) for s in files["predictions.txt"].splitlines() if s.strip()]
    data = value["dataset"]
    if (not table or table[0] != ["train", "node", "observed", "predicted"]
            or len(table) != len(data["rows"]) + 1):
        raise ValueError("incomplete prediction table")
    predictions, tolerances = [], []
    for record, row, y in zip(table[1:], data["rows"], data["targets"]):
        if len(record) != 4 or record[:2] != ["y", "1"]:
            raise ValueError("invalid prediction row")
        if abs(number(record[2]) - y) > printed_tolerance(record[2]) + 1e-12:
            raise ValueError("observed response mismatch")
        prediction = number(record[3])
        tolerance = printed_tolerance(record[3]) + 1e-9
        values = dict(zip(names, row))
        expected = intercept
        for name, beta in coefficients.items():
            if name.endswith(".NA"):
                term = float(values[name[:-3]] is None)
            else:
                term = means[name] if values[name] is None else values[name]
            expected += beta * term
        if abs(expected - prediction) > tolerance:
            raise ValueError("exported coefficients disagree with native predictions")
        predictions.append(prediction)
        tolerances.append(tolerance)
    return {"version": banner[1], "build_date": banner[2], "leaves": 1,
            "intercept": intercept, "coefficients": coefficients, "imputations": means,
            "predictions": predictions, "prediction_tolerances": tolerances,
            "native_difference": value["native_difference"]}


def capture(binary, version, source_url, value, output):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    files = inputs(value, version)
    manifest = {"fixture_format": 1, "case": value["name"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_url": source_url, "binary_sha256": sha(binary.read_bytes()),
                "requested_version": version, "template_generated_by": "46.3",
                "completed": False, "settings": value["options"]}
    with tempfile.TemporaryDirectory(prefix="guide-linear-") as temporary:
        work = Path(temporary)
        for name, content in files.items():
            (work / name).write_text(content)
        try:
            process = subprocess.run([str(binary)], input=files["input.txt"], text=True,
                                     capture_output=True, cwd=work, timeout=30,
                                     env={**os.environ, "TMPDIR": temporary})
            manifest["returncode"] = process.returncode
            files.update({"stdout.txt": process.stdout, "stderr.txt": process.stderr})
            for name in ("report.out", "splits.txt", "predictions.txt", "coefficients.txt", "predict.R"):
                if (work / name).exists():
                    files[name] = (work / name).read_text()
            if process.returncode != 0:
                raise ValueError("reference process did not complete successfully")
            manifest["result"] = parse_result(process.stdout, files, version, value)
            manifest["completed"] = True
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as error:
            manifest["error"] = str(error)
            if isinstance(error, subprocess.TimeoutExpired):
                for name, content in (("stdout.txt", error.stdout), ("stderr.txt", error.stderr)):
                    files[name] = content.decode(errors="replace") if isinstance(content, bytes) else (content or "")
    manifest["file_sha256"] = {name: sha(s.encode()) for name, s in files.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (output / name).write_text(content)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


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
