#!/usr/bin/env python3
"""Capture a small, version-checked GUIDE 46.2/46.3 least-squares fixture.

The batch template was generated interactively by GUIDE 46.3. Both versions
have been exercised; only the version header is adapted. This script does not
change binary permissions or security attributes. It never overwrites fixtures.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inputs(version, left_size):
    batch = f'''GUIDE       (generated from the GUIDE 46.3 interactive input writer)
  {version}      (version header; tested with 46.2 and 46.3)
 1          (model fitting)
"regression.out"
 1          (single tree)
 2          (regression)
 1          (linear)
 1          (least squares)
 3          (constant node models)
 1          (interaction tests)
 2          (no pruning)
"data.dsc"
 2          (exhaustive numerical split search)
 2          (specify maximum depth)
 1          (one split level)
 2          (specify minimum node size)
 2          (minimum node size)
 0          (no LaTeX)
 3          (store split variables and cutpoints)
"splits.txt"
 2          (save fitted values and node IDs)
"predictions.txt"
 1          (skip R export)
 1          (rank of root split variable)
'''
    return {
        "input.txt": batch,
        "data.dsc": "data.txt\nNA\n2\n1 x s\n2 y d\n",
        "data.txt": "x y\n" + "".join(f"{i} {2 if i < left_size else 8}\n" for i in range(20)),
        "data.csv": "x,y\n" + "".join(f"{i},{2 if i < left_size else 8}\n" for i in range(20)),
    }


def parse_result(stdout, files, expected_version):
    banner = re.search(r"Version ([\d.]+) \(Build date: ([^)]+)\)", stdout)
    if banner is None or banner[1] != expected_version:
        raise ValueError("runtime version does not match the requested reference version")
    lines = [line.split() for line in files["predictions.txt"].splitlines() if line.strip()]
    if lines[0] != ["train", "node", "observed", "predicted"] or len(lines) != 21:
        raise ValueError("unexpected prediction table")
    splits = [line.split() for line in files["splits.txt"].splitlines() if line.strip()]
    if len(splits) != 3 or splits[0][1] != "s" or any(row[1] != "t" for row in splits[1:]):
        raise ValueError("expected one split and two terminal nodes")
    conditions = [s.strip() for s in files["regression.out"].splitlines()
                  if s.startswith(" Node 1:") and "or NA" in s]
    if len(conditions) != 1:
        raise ValueError("missing-value direction is not explicit in output")
    return {
        "version": banner[1], "build_date": banner[2],
        "root_feature": splits[0][2], "root_threshold": float(splits[0][5]),
        "leaf_count": 2, "missing_left": "<=" in conditions[0],
        "missing_route_evidence": conditions[0],
        "predictions": [float(row[3]) for row in lines[1:]],
        "observed": [float(row[2]) for row in lines[1:]],
        "terminal_node_ids": [int(row[1]) for row in lines[1:]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--version", choices=["46.2", "46.3"], required=True)
    parser.add_argument("--source-url", required=True, help="URL of the downloaded binary archive")
    parser.add_argument("--left-size", type=int, choices=[5, 10, 15], default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new fixture directory")
    binary = args.binary.resolve()
    files = inputs(args.version, args.left_size)
    manifest = {
        "fixture_format": 1, "case": "least-squares-numeric-stump",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": args.source_url, "binary_sha256": sha(binary.read_bytes()),
        "requested_version": args.version, "template_generated_by": "46.3",
        "version_header_adapted": args.version != "46.3",
        "left_size": args.left_size, "completed": False,
        "settings": {"task": "least_squares", "node_model": "constant", "pruning": "none",
                     "max_depth": 1, "min_samples_leaf": 2, "split_search": "exhaustive"},
        "tolerances": {"cutpoint_absolute": 1e-9, "prediction_absolute": 1e-6},
    }
    with tempfile.TemporaryDirectory(prefix="guide-capture-") as temporary:
        work = Path(temporary)
        for name, content in files.items():
            (work / name).write_text(content)
        try:
            result = subprocess.run([str(binary)], input=files["input.txt"], text=True,
                                    capture_output=True, cwd=work, timeout=30,
                                    env={**os.environ, "TMPDIR": temporary})
            manifest["returncode"] = result.returncode
            files.update({"stdout.txt": result.stdout, "stderr.txt": result.stderr})
            for name in ["regression.out", "predictions.txt", "splits.txt"]:
                if (work / name).exists():
                    files[name] = (work / name).read_text()
            if result.returncode != 0:
                raise ValueError("reference process did not complete successfully")
            manifest["result"] = parse_result(result.stdout, files, args.version)
            manifest["completed"] = True
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as error:
            manifest["error"] = str(error)
    manifest["file_sha256"] = {name: sha(content.encode()) for name, content in files.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (args.output / name).write_text(content)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["completed"]:
        parser.exit(1, "Saved an incomplete diagnostic record; it is not a golden fixture.\n")
    print(f"Captured GUIDE {args.version}: {args.output}")


if __name__ == "__main__":
    main()
