#!/usr/bin/env python3
"""Freeze independent base-R CV predictions and the documented native-scan model."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/multilevel_linear_oracle.R"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows or any(not math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError("incomplete or nonfinite independent oracle output")
    return rows


def compute(name):
    fixture = ROOT / "tests/fixtures/linear-tree/guide-46.3" / name
    options = json.loads((fixture / "options.json").read_text())
    with tempfile.TemporaryDirectory(prefix="guide-linear-oracle-") as directory:
        work = Path(directory)
        shutil.copyfile(fixture / "data.csv", work / "data.csv")
        process = subprocess.run(["Rscript", "--vanilla", str(SCRIPT), str(options["max_depth"]),
                                  str(options["min_samples_leaf"]), "0"],
                                 cwd=work, text=True, capture_output=True, timeout=60, check=True)
        scores = table(work / "oracle-scores.csv")
        if scores[0]["leaves"] != 2**options["max_depth"] or scores[-1]["leaves"] != 1:
            raise ValueError("unexpected independent pruning path")
        se = options["pruning"].get("standard_errors", 0.25)
        selected = {}
        for prefix, mean, error in [("correct", "mean_loss", "standard_error"),
                                     ("native_scan", "native_scan_mean", "native_scan_se")]:
            best = min(scores, key=lambda r: r[mean])
            cutoff = best[mean] + se * best[error]
            selected[prefix + "_selected_leaves"] = next(int(s["leaves"]) for s in reversed(scores)
                                                         if s[mean] <= cutoff)
        result = {"input_sha256": {name: sha(fixture / name) for name in ("data.csv", "dataset.json", "options.json")},
                  "scores": scores, **selected,
                  "trees": {str(int(s["leaves"])): table(work / f"oracle-tree-{i}.csv")
                            for i, s in enumerate(scores, 1)},
                  "heldout_predictions": table(work / "oracle-held-predictions.csv"),
                  "native_scan_heldout_predictions": table(work / "oracle-native-scan-predictions.csv")}
        return result, (work / "oracle-version.txt").read_text().strip(), process.stderr


def capture(output):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    record = {"completed": False, "method": "independent base-R weighted QR and exhaustive cut search",
              "script_sha256": sha(SCRIPT), "cases": {}}
    for name in ("linear-multilevel-loocv", "linear-multilevel-loocv-weighted", "linear-multilevel-fixed-3"):
        value, version, stderr = compute(name)
        record["cases"][name] = value
        record["r_version"] = version
        if stderr.strip():
            raise ValueError(f"unexpected independent oracle stderr: {stderr}")
        for native_version in ("46.3", "46.2"):
            path = ROOT / "tests/fixtures/linear-tree" / f"guide-{native_version}" / name
            native = json.loads((path / "manifest.json").read_text())
            if "pruning_path" in native["result"]:
                for actual, expected in zip(value["scores"], native["result"]["pruning_path"], strict=True):
                    if (actual["leaves"] != expected["leaves"] or actual["pruned_node"] != expected["pruned_node"]
                            or abs(actual["alpha"]-expected["alpha"]) > expected["alpha_tolerance"]):
                        raise ValueError("independent path does not match the native fixed-size path")
                continue
            for actual, expected in zip(value["scores"], native["result"]["pruning_scores"], strict=True):
                for field, oracle_field in [("mean_loss", "native_scan_mean"), ("standard_error", "native_scan_se")]:
                    if abs(actual[oracle_field] - expected[field]) > expected[field + "_tolerance"]:
                        raise ValueError("native table not explained by the documented scan model")
            if value["native_scan_selected_leaves"] != native["result"]["leaf_count"]:
                raise ValueError("simulated native selection does not explain the captured final tree")
    record["completed"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.output)
    print(f"Captured independent R oracle: {args.output}")
