#!/usr/bin/env python3
"""Capture GUIDE 46.x importance reports, retaining normalized scores and groups.

The 46.3 interactive writer verified the least-squares template. Classification
uses the importance-mode interaction/prior/cost inputs, not the fitting menu.
No overwrite, version substitution beyond the header, or binary patching occurs.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from capture_reference_core import sha, printed_tolerance
from reference_cases import data_files
from reference_importance_cases import cases


def inputs(value, version):
    files = data_files(value)
    lines = ["GUIDE (46.x importance reference)", version, "2 (importance scoring)", '"report.out"']
    if value["classification"]:
        lines += ["1 (classification)", "1 (interaction tests)", '"data.dsc"',
                  "1 (estimated priors)", "1 (unit costs)"]
    else:
        lines += ["2 (regression)", "1 (linear)", "1 (least squares)",
                  "1 (interaction tests)", '"data.dsc"']
    lines += ["2 (exhaustive numerical cuts)", "2 (explicit depth)",
              str(value["importance_options"]["max_depth"]), "2 (explicit minimum size)",
              str(value["options"]["min_samples_leaf"]), "0 (no LaTeX)",
              "1 (no split storage)", "1 (no selected-variable DSC)",
              "1 (store importance scores)", '"scores.txt"']
    files.update({"input.txt": "\n".join(lines) + "\n",
                  "dataset.json": json.dumps(value["dataset"], indent=2) + "\n",
                  "options.json": json.dumps(value["options"], indent=2) + "\n",
                  "importance.json": json.dumps(value["importance_options"], indent=2) + "\n"})
    return files


def parse_result(stdout, files, version, value):
    banner = re.search(r"Version ([\d.]+) \(Build date: ([^)]+)\)", stdout)
    if banner is None or banner[1] != version:
        raise ValueError("runtime version does not match requested version")
    report = files["report.out"]
    if "Importance scores are stored in scores.txt" not in stdout or "Error:" in report:
        raise ValueError("importance calculation did not complete")
    settings = [(r"Maximum number of split levels:\s*(\d+)", value["importance_options"]["max_depth"]),
                (r"Minimum node sample size:\s*(\d+)", value["options"]["min_samples_leaf"]),
                (r"Using (\d+) permutations to standardize", value["importance_options"]["permutations"])]
    for pattern, expected in settings:
        if re.findall(pattern, report) != [str(expected)]:
            raise ValueError("effective importance settings do not match requested settings")
    lines = [line.split() for line in files["scores.txt"].splitlines() if line.strip()]
    if not lines or lines[0] != ["Rank", "Group", "Score", "Missing", "Variable"]:
        raise ValueError("unexpected importance score header")
    names = {f["name"] for f in value["dataset"]["schema"]}
    scores = []
    for row in lines[1:]:
        if len(row) != 5 or row[1] not in set("ABCDEF") or row[4] not in names:
            raise ValueError("malformed importance row")
        rank, score, missing = float(row[0]), float(row[2]), float(row[3])
        if any(not math.isfinite(n) for n in [rank, score, missing]) or rank < 1 or score < 0 or missing < 0:
            raise ValueError("invalid importance score")
        scores.append({"feature": row[4], "rank": rank, "group": row[1],
                       "normalized_score": score, "score_tolerance": printed_tolerance(row[2]),
                       "missing_native": missing})
    reported = {s["feature"] for s in scores}
    if len(reported) != len(scores) or any(a["rank"] > b["rank"] or a["normalized_score"] < b["normalized_score"]
                                          for a, b in zip(scores, scores[1:])):
        raise ValueError("importance rows are duplicated or not ordered")
    block = re.findall(r"Scaled importance scores of predictor variables\n(.*?)(?=\s*99% threshold)", report, re.DOTALL)
    if len(block) != 1:
        raise ValueError("missing normalized score report")
    table = [line.split() for line in block[0].splitlines() if line.strip()][1:]
    if len(table) != len(scores):
        raise ValueError("importance file/report row counts differ")
    for row, expected in zip(table, scores):
        if len(row) != 3 or row[2] != expected["feature"] or float(row[0]) != expected["normalized_score"] or float(row[1]) != expected["rank"]:
            raise ValueError("importance file and report disagree")
    thresholds = {}
    for confidence, token in re.findall(r"^\s*(99|95|90|80|50)% threshold is\s*(\S+)\s*$", report, re.MULTILINE):
        score = float(token)
        if confidence in thresholds or not math.isfinite(score) or score < 0:
            raise ValueError("invalid normalized threshold")
        thresholds[confidence] = score
    if set(thresholds) != {"99", "95", "90", "80", "50"} or thresholds["95"] != 1.0:
        raise ValueError("incomplete or unnormalized importance thresholds")
    if not all(thresholds[a] >= thresholds[b] for a, b in zip(["99", "95", "90", "80"], ["95", "90", "80", "50"])):
        raise ValueError("importance thresholds are not ordered")
    return {"version": banner[1], "build_date": banner[2], "score_kind": "normalized",
            "scores": scores, "thresholds": thresholds,
            "omitted_features": sorted(names - reported)}


def capture(binary, version, source_url, value, output):
    if output.exists():
        raise FileExistsError(output)
    files = inputs(value, version)
    manifest = {"fixture_format": 1, "case": value["name"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "binary_sha256": sha(binary.read_bytes()), "source_url": source_url,
                "requested_version": version, "completed": False,
                "settings": value["options"], "importance_options": value["importance_options"],
                "independent_raw_statistics": value["expected_raw_statistics"],
                "expected_important": value["expected_important"], "expected_zero": value["expected_zero"],
                "comparison_scope": "Reported important sets/rank relations; raw scores checked independently. Exact Monte Carlo-adjusted magnitudes and thresholds are not claimed equivalent."}
    with tempfile.TemporaryDirectory(prefix="guide-importance-") as temporary:
        work = Path(temporary)
        for name, content in files.items():
            (work / name).write_text(content)
        try:
            run = subprocess.run([str(binary)], input=files["input.txt"], text=True, capture_output=True,
                                 cwd=work, timeout=30, env={**os.environ, "TMPDIR": temporary})
            manifest["returncode"] = run.returncode
            files.update({"stdout.txt": run.stdout, "stderr.txt": run.stderr})
            for name in ["report.out", "scores.txt"]:
                if (work / name).exists():
                    files[name] = (work / name).read_text()
            if run.returncode != 0:
                raise ValueError("reference process did not complete successfully")
            manifest["result"] = parse_result(run.stdout, files, version, value)
            manifest["completed"] = True
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as error:
            manifest["error"] = str(error)
            if isinstance(error, subprocess.TimeoutExpired):
                for name, content in [("stdout.txt", error.stdout), ("stderr.txt", error.stderr)]:
                    files[name] = content.decode(errors="replace") if isinstance(content, bytes) else (content or "")
    manifest["file_sha256"] = {n: sha(s.encode()) for n, s in files.items()}
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
        parser.error("output exists; choose a new directory")
    m = capture(args.binary.resolve(), args.version, args.source_url, cases()[args.case], args.output)
    if not m["completed"]:
        parser.exit(1, f"Incomplete diagnostic record: {m.get('error')}\n")
    print(f"Captured GUIDE {args.version}: {args.output}")


if __name__ == "__main__":
    main()
