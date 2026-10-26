#!/usr/bin/env python3
"""Capture version-checked constant regression/classification GUIDE cases.

The classification template follows the 46.3 interactive writer (simple models,
skip linear splits); the regression template follows capture_reference_stump.py.
Existing fixtures are never overwritten. Failures are retained as diagnostics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from reference_cases import all_cases, data_files


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inputs(value, version):
    files = data_files(value)
    opt = value["options"]
    pruning = opt["pruning"]
    kind = pruning["kind"]
    code = {"none": 3 if value["classification"] else 2,
            "cross_validation": 1, "fixed_leaves": 0}[kind]
    pruning_input = ([f"{code} (no pruning)"] if kind == "none" else
                     [f"{code} ({kind})"])
    if kind == "fixed_leaves":
        pruning_input.append(f'{pruning["max_leaves"]} (maximum terminal nodes)')
    lines = ["GUIDE (46.x constant-node reference)", f"{version} (version)",
             "1 (model fitting)", '"report.out"', "1 (single tree)"]
    if value["classification"]:
        lines += ["1 (classification)", "1 (simple node model)",
                  "2 (skip linear splits; keep interactions)"] + pruning_input + ['"data.dsc"']
        if kind == "cross_validation":
            lines += [str(pruning["folds"]), "1 (mean-based CV)", str(pruning["standard_errors"])]
        if opt["class_priors"]:
            files["priors.txt"] = "".join(f"{p}\n" for p in opt["class_priors"])
            lines += ["3 (explicit priors)", '"priors.txt"']
        else:
            lines += ["1 (estimated priors)"]
        if opt["class_costs"]:
            # GUIDE indexes C(predicted | true); Rust indexes [true][predicted].
            files["costs.txt"] = "".join(" ".join(map(str, row)) + "\n"
                                           for row in zip(*opt["class_costs"]))
            lines += ["2 (explicit costs)", '"costs.txt"']
        else:
            lines += ["1 (unit costs)"]
    elif opt["task"]["loss"]["kind"] in ("poisson", "quantile"):
        # These models have no cutpoint prompt: GUIDE uses its fixed grid.
        loss = opt["task"]["loss"]
        if kind != "none":
            raise ValueError("Poisson/quantile reference cases do not prune")
        lines += ["2 (regression)"]
        if loss["kind"] == "poisson":
            lines += ["3 (Poisson)", "3 (constant nodes)"]
        else:
            lines += ["2 (quantile)", "3 (constant nodes)", "1 (one quantile)",
                      f'{loss["probability"]} (quantile probability)']
        lines += ["1 (interaction tests)"] + pruning_input + ['"data.dsc"']
        lines += ["2 (explicit depth)", str(opt["max_depth"]),
                  "2 (explicit minimum node size)", "2", "0 (no LaTeX)",
                  "3 (store splits)", '"splits.txt"', "2 (store predictions)",
                  '"predictions.txt"', "1 (skip R export)", "1 (root variable rank)"]
        files["input.txt"] = "\n".join(lines) + "\n"
        files["dataset.json"] = json.dumps(value["dataset"], indent=2) + "\n"
        files["options.json"] = json.dumps(opt, indent=2) + "\n"
        return files
    else:
        lines += ["2 (regression)", "1 (linear)", "1 (least squares)",
                  "3 (constant nodes)", "1 (interaction tests)"] + pruning_input + ['"data.dsc"']
        if kind == "cross_validation":
            lines += [str(pruning["folds"]), "1 (mean-based CV)", str(pruning["standard_errors"])]
            if value["dataset"]["weights"]:
                lines += ["2 (weighted pruning errors)"]
    lines += ["2 (exhaustive cutpoints)", "2 (explicit depth)", str(opt["max_depth"]),
              "2 (explicit minimum node size)", "2", "0 (no LaTeX)",
              "3 (store splits)", '"splits.txt"', "2 (store predictions)",
              '"predictions.txt"', "1 (skip R export)", "1 (root variable rank)"]
    files["input.txt"] = "\n".join(lines) + "\n"
    files["dataset.json"] = json.dumps(value["dataset"], indent=2) + "\n"
    files["options.json"] = json.dumps(opt, indent=2) + "\n"
    return files


def parse_result(stdout, files, expected_version, value):
    banner = re.search(r"Version ([\d.]+) \(Build date: ([^)]+)\)", stdout)
    if banner is None or banner[1] != expected_version:
        raise ValueError("runtime version does not match the requested reference version")
    report = files["report.out"]
    if value["name"] == "class-constant-response":
        # GUIDE exits with status zero for this explicit input rejection.
        # This is an accepted rejection case, not a fitted-tree fixture.
        if ("Error: Only one class level in data" not in report
                or len(set(value["dataset"]["targets"])) != 1
                or "predictions.txt" in files or "splits.txt" in files):
            raise ValueError("expected explicit single-class rejection")
        return {"version": banner[1], "build_date": banner[2],
                "outcome": "rejected", "error_code": "single_class"}
    final = re.findall(r"Number of terminal nodes of final tree:\s*(\d+)", report)
    if len(final) != 1 or "Results are stored in report.out" not in stdout:
        raise ValueError("missing or ambiguous final tree size")
    for label, expected in [("Maximum number of split levels", value["options"]["max_depth"]),
                            ("Minimum node sample size", value["options"]["min_samples_leaf"])]:
        actual = re.findall(re.escape(label) + r":\s*(\d+)", report)
        if actual != [str(expected)]:
            raise ValueError(f"effective {label} does not match the requested setting")
    lines = [shlex.split(s) for s in files["predictions.txt"].splitlines() if s.strip()]
    if not lines or lines[0][:4] != ["train", "node", "observed", "predicted"]:
        raise ValueError("unexpected prediction header")
    if len(lines) != len(value["dataset"]["rows"]) + 1:
        raise ValueError("unexpected prediction count")
    classes = sorted(set(value["dataset"]["targets"])) if value["classification"] else []
    if lines[0][4:] != [f"P({c})" for c in classes]:
        raise ValueError("unexpected probability columns")
    predictions, observed, node_ids, probabilities = [], [], [], []
    # With an exposure, GUIDE reports Poisson observations and predictions as
    # rates: counts divided by exposure.
    exposures = value["dataset"]["offsets"] or [None] * len(value["dataset"]["rows"])
    for row, target, exposure in zip(lines[1:], value["dataset"]["targets"], exposures):
        if len(row) != len(lines[0]) or row[0] != "y":
            raise ValueError("incomplete prediction row")
        obs = row[2] if classes else float(row[2])
        expected_obs = target if exposure is None else target / exposure
        if obs != expected_obs and not (exposure and abs(obs - expected_obs) <= 1e-5 * max(1, abs(expected_obs))):
            raise ValueError("observed response does not match the input row")
        observed.append(obs)
        predictions.append(row[3] if classes else float(row[3]))
        node_ids.append(int(row[1]))
        if classes:
            p = list(map(float, row[4:]))
            if any(not math.isfinite(x) or x < 0 or x > 1 for x in p) or abs(sum(p) - 1) > 2e-5:
                raise ValueError("invalid class probabilities")
            if row[3] not in classes:
                raise ValueError("unknown predicted class")
            probabilities.append(p)
        elif not math.isfinite(predictions[-1]):
            raise ValueError("nonfinite prediction")
    raw_splits = [shlex.split(s) for s in files["splits.txt"].splitlines() if s.strip()]
    # GUIDE repeats a categorical node's identical encoding before each child.
    unique = {}
    occurrences = {}
    for row in raw_splits:
        key = int(row[0])
        if key in unique and (row != unique[key] or row[1] not in ("c", "b")):
            raise ValueError("conflicting or duplicate node IDs")
        unique[key] = row
        occurrences[key] = occurrences.get(key, 0) + 1
    if any(occurrences[k] != (2 if r[1] in ("c", "b") else 1) for k, r in unique.items()):
        raise ValueError("incomplete categorical split encoding")
    raw_splits = list(unique.values())
    splits, terminals = [], []
    for row in raw_splits:
        node_id = int(row[0])
        if row[1] == "t":
            terminals.append(node_id)
            continue
        # The first condition sends a case to the left child. Preserve it along
        # with the split file so categorical and missing conventions are auditable.
        conditions = re.findall(rf"^ +Node {node_id}: (.+)$", report, re.MULTILINE)
        if len(conditions) != 2:
            raise ValueError(f"ambiguous conditions for node {node_id}")
        entry = {"node_id": node_id, "feature": row[2], "native_fields": row[1:],
                 "conditions": conditions, "missing_left": "or NA" in conditions[0]}
        if row[1] in ("s", "n"):
            entry.update(kind="numeric", threshold=float(row[5]))
            branch = re.fullmatch(re.escape(row[2]) + r" <= ([\d.Ee+\-]+)( or NA)?", conditions[0])
            if branch is None or not math.isclose(float(branch[1]), entry["threshold"],
                                                  rel_tol=5e-7, abs_tol=5e-8):
                raise ValueError("numeric split and report disagree")
            if not conditions[1].startswith(row[2] + " > ") or (
                    ("or NA" in conditions[0]) == ("or NA" in conditions[1])):
                raise ValueError("inconsistent numerical missing direction")
        elif row[1] in ("c", "b"):
            if len(row[5:]) != int(row[4]):
                raise ValueError("incomplete category list")
            if (not conditions[0].startswith(row[2] + " = ")
                    or not conditions[1].startswith(row[2] + " /= ")
                    or re.findall(r'"([^"]+)"', conditions[0]) != row[5:]
                    or re.findall(r'"([^"]+)"', conditions[1]) != row[5:]):
                raise ValueError("categorical split and report disagree")
            entry.update(kind="categorical", left_categories=[s for s in row[5:] if s != "NA"],
                         missing_left="NA" in row[5:], unknown_left=False)
        else:
            raise ValueError(f"unsupported split encoding: {row}")
        splits.append(entry)
    if len(set(int(row[0]) for row in raw_splits)) != len(raw_splits):
        raise ValueError("duplicate node IDs")
    if len(terminals) != int(final[0]) or len(raw_splits) != 2 * len(terminals) - 1:
        raise ValueError("inconsistent final tree")
    all_ids = {int(r[0]) for r in raw_splits}
    split_ids = {s["node_id"] for s in splits}
    if 1 not in all_ids or any(2*s["node_id"] not in all_ids or
                              2*s["node_id"]+1 not in all_ids for s in splits) or any(
            n > 1 and n // 2 not in split_ids for n in all_ids):
        raise ValueError("disconnected tree")
    if any(n not in terminals for n in node_ids):
        raise ValueError("prediction references a nonterminal node")
    result = {"version": banner[1], "build_date": banner[2], "outcome": "fitted", "classes": classes,
            "leaf_count": len(terminals), "splits": splits, "predictions": predictions,
            "observed": observed, "terminal_node_ids": node_ids,
            "probabilities": probabilities}
    if value["options"]["pruning"]["kind"] == "cross_validation":
        folds = re.findall(r"Pruning by v-fold cross-validation, with v =\s*(\d+)", report)
        if folds != [str(value["options"]["pruning"]["folds"])]:
            raise ValueError("effective CV fold count does not match the requested setting")
        result["pruning_scores"] = parse_pruning(report)
        # Native weighted errors are sum(w*loss)/N. Rust reports sum(w*loss)/sum(w),
        # retaining invariance to weight units. Both mean and naive SE scale by
        # mean weight; retain the printed values and make this conversion explicit.
        if value["dataset"]["weights"] and not value["classification"]:
            mean_weight = sum(value["dataset"]["weights"]) / len(value["dataset"]["rows"])
            result["pruning_mean_weight"] = mean_weight
            for row in result["pruning_scores"]:
                for field in ["mean_loss", "standard_error"]:
                    row["native_" + field] = row[field]
                    row[field] /= mean_weight
                    row[field + "_tolerance"] /= mean_weight
        selected = [r for r in result["pruning_scores"] if r["selected"]]
        if len(selected) != 1 or selected[0]["leaves"] != len(terminals):
            raise ValueError("CV selection disagrees with the final tree")
    if value["options"]["pruning"]["kind"] == "fixed_leaves":
        result["pruning_path"] = parse_fixed_path(report)
        selected = next(row for row in result["pruning_path"]
                        if row["leaves"] <= value["options"]["pruning"]["max_leaves"])
        if selected["leaves"] != len(terminals):
            raise ValueError("fixed-size selection disagrees with the final tree")
    return result


def parse_pruning(report):
    sections = re.findall(r"Size and CV [^\n]+:\n(.*?)(?=\n\s*0-SE tree)", report, re.DOTALL)
    if len(sections) != 1:
        raise ValueError("missing or ambiguous CV table")
    rows = []
    for line in sections[0].splitlines()[1:]:
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*(\d+)([*+\-]*)\s+(\d+)\s+(.+?)\s*", line)
        if match is None:
            raise ValueError("malformed CV row")
        numbers = list(map(float, match[4].split()))
        if len(numbers) != 5 or any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError("invalid CV statistics")
        rows.append({"index": int(match[1]), "leaves": int(match[3]),
                     "mean_loss": numbers[0], "standard_error": numbers[1],
                     "mean_loss_tolerance": printed_tolerance(match[4].split()[0]),
                     "standard_error_tolerance": printed_tolerance(match[4].split()[1]),
                     "selected": "**" in match[2]})
    if not rows or [r["index"] for r in rows] != list(range(len(rows))) or rows[-1]["leaves"] != 1:
        raise ValueError("incomplete CV path")
    if any(a["leaves"] <= b["leaves"] for a, b in zip(rows, rows[1:])):
        raise ValueError("CV path is not nested")
    return rows


def printed_tolerance(token):
    match = re.fullmatch(r"[+\-]?\d+\.(\d*)(?:[Ee]([+\-]?\d+))?", token)
    if match is None:
        raise ValueError("unexpected CV numeric precision")
    return 0.5 * 10.0 ** (int(match[2] or 0) - len(match[1]))


def parse_fixed_path(report):
    blocks = re.findall(r" Pruning sequence\n[^\n]+\n[^\n]+\n(.*?)(?=\n\s*\n)", report, re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("missing fixed-size pruning path")
    rows = []
    for line in blocks[0].splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("malformed fixed-size pruning row")
        alpha = float(fields[3])
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("invalid pruning alpha")
        rows.append({"index": int(fields[0]), "pruned_node": int(fields[1]),
                     "leaves": int(fields[2]), "alpha": alpha,
                     "alpha_tolerance": printed_tolerance(fields[3])})
    if (not rows or [r["index"] for r in rows] != list(range(len(rows)))
            or rows[0]["pruned_node"] != 0 or rows[-1]["leaves"] != 1
            or any(r["leaves"] < 1 or r["pruned_node"] < 1 for r in rows[1:])
            or any(a["leaves"] <= b["leaves"] or a["alpha"] > b["alpha"] for a, b in zip(rows, rows[1:]))):
        raise ValueError("inconsistent fixed-size pruning path")
    return rows


def capture(binary, version, source_url, value, output, *, input_builder=inputs,
            result_parser=parse_result, extra_output_files=()):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    files = input_builder(value, version)
    manifest = {
        "fixture_format": 2, "case": value["name"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url, "binary_sha256": sha(binary.read_bytes()),
        "requested_version": version, "template_generated_by": "46.3",
        "completed": False, "settings": value["options"],
        # Native predictions/probabilities use five significant digits; split
        # files use ten. These limits are fixed before comparison with Rust.
        "tolerances": {"cutpoint_absolute": 1e-9, "prediction_absolute": 5e-5,
                       "prediction_relative": 5e-6, "probability_absolute": 5e-6},
    }
    with tempfile.TemporaryDirectory(prefix="guide-core-") as temporary:
        work = Path(temporary)
        for name, content in files.items():
            (work / name).write_text(content)
        try:
            process = subprocess.run([str(binary)], input=files["input.txt"], text=True,
                                     capture_output=True, cwd=work, timeout=30,
                                     env={**os.environ, "TMPDIR": temporary})
            manifest["returncode"] = process.returncode
            files.update({"stdout.txt": process.stdout, "stderr.txt": process.stderr})
            for name in ("report.out", "splits.txt", "predictions.txt", *extra_output_files):
                if (work / name).exists():
                    files[name] = (work / name).read_text()
            if process.returncode != 0:
                raise ValueError("reference process did not complete successfully")
            manifest["result"] = result_parser(process.stdout, files, version, value)
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
    parser.add_argument("--case", choices=sorted(all_cases()), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new fixture directory")
    manifest = capture(args.binary.resolve(), args.version, args.source_url,
                       all_cases()[args.case], args.output)
    if not manifest["completed"]:
        parser.exit(1, f"Incomplete diagnostic record: {manifest.get('error')}\n")
    print(f"Captured GUIDE {args.version}: {args.output}")


if __name__ == "__main__":
    main()
