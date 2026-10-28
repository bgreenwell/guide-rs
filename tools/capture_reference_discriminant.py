#!/usr/bin/env python3
"""Capture version-checked single-node GUIDE LDA fits.

The batch template follows the 46.3 interactive writer for classification with
"trees with higher accuracy" and LDA node models (code 4 in 46.2 and 46.3).
Existing fixtures are never overwritten. Failures are retained as diagnostics.
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

from reference_discriminant_cases import cases


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inputs(value, version):
    names = value["features"] + ["y"]
    records = [[("NA" if v is None else repr(v)) for v in row] + [t]
               for row, t in zip(value["rows"], value["targets"])]
    lines = ["GUIDE (46.x single-node LDA reference)", f"{version} (version)",
             "1 (model fitting)", '"report.out"', "1 (single tree)", "1 (classification)",
             "4 (LDA node models)", "3 (no pruning)", '"data.dsc"',
             "1 (estimated priors)", "1 (unit costs)", "1 (default split count)",
             "1 (default depth)", "2 (explicit minimum node size)", str(value["min_node"]),
             "0 (no LaTeX)", "2 (store coefficients)", '"coefficients.txt"',
             "2 (store fitted values)", '"fitted.txt"', "1 (skip R export)", "2 (no plot code)"]
    return {
        "input.txt": "\n".join(lines) + "\n",
        "data.dsc": "data.txt\nNA\n2\n" + "".join(
            f"{i} {n} {'d' if n == 'y' else 'n'}\n" for i, n in enumerate(names, 1)),
        "data.txt": " ".join(names) + "\n" + "".join(" ".join(r) + "\n" for r in records),
        "case.json": json.dumps(value, indent=2) + "\n",
    }


def number(token):
    return float(token.replace("D", "E"))


def parse_result(stdout, files, version, value):
    report = files["report.out"]
    if f"Version {version}" not in stdout:
        raise ValueError("runtime version mismatch")
    if "LDA node models" not in report:
        raise ValueError("report is not an LDA fit")
    if not re.search(r"Number of terminal nodes of final tree:\s+1\b", report):
        raise ValueError("root was split; case is not a single node")
    coefficients = []
    for line in files["coefficients.txt"].splitlines()[1:]:
        t = line.split()
        if t:
            coefficients.append({"variable": t[1], "mean": number(t[2]),
                                 "coefficients": [None if x == "NA" else number(x) for x in t[3:]]})
    fitted = []
    for line in files["fitted.txt"].splitlines()[1:]:
        t = line.split()
        if t:
            fitted.append({"coordinates": [number(x) for x in t[2:-2]],
                           "observed": t[-2].strip('"'), "predicted": t[-1].strip('"')})
    if len(fitted) != len(value["rows"]):
        raise ValueError("fitted values do not cover every row")
    block = report.split("Node 1: Terminal node", 1)[1]
    stats = {}
    old = re.search(r"Class\s+Crimmean1(.*?)\n(.*?)\n\s*Variance\s+(.*?)\n", block, re.S)
    if old:
        # 46.2 prints one table of coordinate means and priors, then pooled variances.
        variances = [number(x) for x in old.group(3).split()]
        rows = [l.split() for l in old.group(2).splitlines() if l.split()]
        for k, v in enumerate(variances, 1):
            stats[f"coordinate{k}"] = {t[0]: {"mean": number(t[k]), "variance": v} for t in rows}
        stats["priors"] = {t[0]: number(t[-1]) for t in rows}
        return {"coefficients": coefficients, "fitted": fitted, "leaf": stats}
    for k in (1, 2):
        m = re.search(rf"Class\s+Crim{k}Mean\s+Crim{k}Variance\n(.*?)\n\s*\n", block, re.S)
        if m:
            stats[f"coordinate{k}"] = {t[0]: {"mean": number(t[1]), "variance": number(t[2])}
                                       for t in (l.split() for l in m.group(1).splitlines()) if t}
    m = re.search(r"Class\s+Prior\n(.*?)\n\s*-{5}", block, re.S)
    if not m:
        raise ValueError("leaf priors not found")
    stats["priors"] = {t[0]: number(t[1]) for t in (l.split() for l in m.group(1).splitlines()) if t}
    return {"coefficients": coefficients, "fitted": fitted, "leaf": stats}


def capture(binary, version, source_url, value, output):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    files = inputs(value, version)
    manifest = {
        "fixture_format": 2, "case": value["name"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url, "binary_sha256": sha(binary.read_bytes()),
        "requested_version": version, "template_generated_by": "46.3", "completed": False,
        # The coefficient file prints eight significant digits, leaf statistics
        # six and fitted coordinates four. These limits are fixed before comparison.
        "tolerances": {"coefficient_relative": 1e-6, "statistic_relative": 5e-6,
                       "coordinate_relative": 5e-4, "prior_absolute": 1e-7},
    }
    with tempfile.TemporaryDirectory(prefix="guide-lda-") as temporary:
        work = Path(temporary)
        for name, content in files.items():
            (work / name).write_text(content)
        try:
            process = subprocess.run([str(binary)], input=files["input.txt"], text=True,
                                     capture_output=True, cwd=work, timeout=30,
                                     env={**os.environ, "TMPDIR": temporary})
            manifest["returncode"] = process.returncode
            files.update({"stdout.txt": process.stdout, "stderr.txt": process.stderr})
            for name in ("report.out", "coefficients.txt", "fitted.txt"):
                if (work / name).exists():
                    files[name] = (work / name).read_text()
            if process.returncode != 0:
                raise ValueError("reference process did not complete successfully")
            manifest["result"] = parse_result(process.stdout, files, version, value)
            manifest["completed"] = True
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as error:
            manifest["error"] = str(error)
    manifest["file_sha256"] = {name: sha(s.encode()) for name, s in files.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (output / name).write_text(content)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


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
