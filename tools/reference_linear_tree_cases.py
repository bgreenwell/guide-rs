"""Multiple-linear growth/pruning, isolated from deferred stepwise work."""
import copy
from reference_cases import case, data_files as constant_data_files


def shallow_cases():
    result = {}
    for kind in ("split-only", "weighted", "split-and-fit"):
        value = case(f"linear-tree-{kind}", False,
                     [("x", "numeric"), ("z", "numeric")],
                     [[i, 2 * (i % 4) - 3] for i in range(40)],
                     [2 * (2 * (i % 4) - 3) + (10 if i >= 20 else 0) + (i % 3 - 1)
                      + (3 * i if kind == "split-and-fit" else 0) for i in range(40)],
                     weights=[1 + i % 5 for i in range(40)] if kind == "weighted" else None,
                     depth=1)
        value["dataset"]["schema"][0]["role"] = "both" if kind == "split-and-fit" else "split"
        value["dataset"]["schema"][1]["role"] = "fit"
        value["options"]["node_model"] = {"kind": "linear"}
        # Keep every evaluated child above the native small-sample fallback
        # region; that fitting-policy difference remains separately documented.
        value["options"]["min_samples_leaf"] = 10
        result[value["name"]] = value
    for se, suffix in [(0.25, "default"), (100.0, "wide-se")]:
        value = copy.deepcopy(result["linear-tree-split-only"])
        value["name"] = f"linear-tree-loocv-{suffix}"
        value["options"]["pruning"] = {"kind": "cross_validation", "folds": 40,
                                        "standard_errors": se}
        result[value["name"]] = value
    return result


def multilevel_cases():
    result = {}
    for kind, depth in [("depth2", 2), ("depth3", 3), ("weighted", 3),
                        ("fit-missing", 3), ("weighted-fit-missing", 3), ("split-and-fit", 3)]:
        levels = [0, 3, 15, 21] if depth == 2 else [0, 2, 10, 14, 30, 34, 50, 56]
        n = 16 * len(levels)
        rows, targets = [], []
        for i in range(n):
            z = None if "missing" in kind and i % 7 == 0 else 2 * (i % 4) - 3
            rows.append([i, z])
            targets.append(levels[i // 16] + (6 if z is None else 2 * z)
                           + (i % 3 - 1) + (3 * i if kind == "split-and-fit" else 0))
        value = case(f"linear-multilevel-{kind}", False,
                     [("x", "numeric"), ("z", "numeric")], rows, targets,
                     weights=[1+i % 5 for i in range(n)] if "weighted" in kind else None,
                     depth=depth)
        value["dataset"]["schema"][0]["role"] = "both" if kind == "split-and-fit" else "split"
        value["dataset"]["schema"][1]["role"] = "fit"
        value["options"]["node_model"] = {"kind": "linear"}
        value["options"]["min_samples_leaf"] = 12
        result[value["name"]] = value
    for weighted in (False, True):
        value = copy.deepcopy(result["linear-multilevel-depth3"])
        value["name"] = "linear-multilevel-" + ("weighted-" if weighted else "") + "split-missing"
        for i, row in enumerate(value["dataset"]["rows"]):
            if i % 13 == 0:
                row[0] = None
        if weighted:
            value["dataset"]["weights"] = [1+i % 5 for i in range(128)]
        result[value["name"]] = value
    for size in [3, 5]:
        value = copy.deepcopy(result["linear-multilevel-depth3"])
        value["name"] = f"linear-multilevel-fixed-{size}"
        value["options"]["pruning"] = {"kind": "fixed_leaves", "max_leaves": size}
        result[value["name"]] = value
    for weighted in (False, True):
        value = copy.deepcopy(result["linear-multilevel-depth2"])
        value["name"] = "linear-multilevel-loocv" + ("-weighted" if weighted else "")
        if weighted:
            value["dataset"]["weights"] = [1+i % 5 for i in range(64)]
        value["options"]["pruning"] = {"kind": "cross_validation", "folds": 64, "standard_errors": 0.25}
        result[value["name"]] = value
    return result


def cases():
    return {**shallow_cases(), **multilevel_cases()}


def data_files(value):
    files = constant_data_files(value)
    codes = {"split": "s", "fit": "f", "both": "n"}
    files["data.dsc"] = "data.txt\nNA\n2\n" + "".join(
        f'{i} {f["name"]} {codes[f["role"]]}\n'
        for i, f in enumerate(value["dataset"]["schema"], 1)) + "3 y d\n"
    if value["dataset"]["weights"]:
        files["data.dsc"] += "4 weight w\n"
    return files
