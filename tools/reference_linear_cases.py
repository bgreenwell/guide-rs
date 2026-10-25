"""Root-only numerical least-squares leaf acceptance cases (no truncation)."""
from reference_cases import case, data_files as constant_data_files


def cases():
    result = {}
    for stepwise in (False, True):
        for kind in ("complete", "weighted", "missing", "all-missing", "collinear", "small", "moderate"):
            n = 4 if kind == "small" else 40
            rows, targets = [], []
            for i in range(n):
                x = None if kind == "all-missing" or kind == "missing" and i % 5 == 0 else i
                z = 2 * i if kind == "collinear" else i % 3
                y = 3 + 2 * i - 4 * (i % 3) + (i % 5 - 2) * 0.1
                if kind == "missing" and x is None:
                    y = 100 + i
                if kind == "moderate":
                    x, z = (1 if i % 2 else -1), (1 if i // 2 % 2 else -1)
                    y = x + 0.02 * z + 2.9 * (1 if i // 4 % 2 else -1)
                rows.append([x, z])
                targets.append(y)
            name = f'ls-{"stepwise" if stepwise else "linear"}-{kind}'
            value = case(name, False, [("x", "numeric"), ("z", "numeric")], rows, targets,
                         weights=[1 + i % 4 for i in range(n)] if kind == "weighted" else None,
                         depth=1)
            for feature in value["dataset"]["schema"]:
                feature["role"] = "fit"
            value["options"]["node_model"] = (
                {"kind": "stepwise", "alpha": 0.05, "max_variables": 0, "backward": True}
                if stepwise else {"kind": "linear"})
            value["native_difference"] = (
                "small_multiple_fallback" if not stepwise and kind == "small" else
                "collinear_stepwise_fallback" if stepwise and kind == "collinear" else
                "grouped_stepwise_selection" if stepwise and kind == "missing" else None)
            result[name] = value
    # Both univariate F tails underflow in ordinary probability. The variable
    # limit makes the first selected feature observable without later additions.
    value = case("ls-stepwise-extreme-tails", False,
                 [("x", "numeric"), ("z", "numeric")],
                 [[-1 if i % 2 == 0 else 1, -1 if i // 2 % 2 == 0 else 1] for i in range(3200)],
                 [(-1 if i % 2 == 0 else 1) + 1.1 * (-1 if i // 2 % 2 == 0 else 1)
                  + 0.01 * (-1 if i // 4 % 2 == 0 else 1) for i in range(3200)], depth=1)
    for feature in value["dataset"]["schema"]:
        feature["role"] = "fit"
    value["options"]["node_model"] = {"kind": "stepwise", "alpha": 0.05,
                                       "max_variables": 1, "backward": True}
    value["native_difference"] = None
    result[value["name"]] = value
    return result


def data_files(value):
    files = constant_data_files(value)
    # Root-only cases: f variables fit the model but cannot split the tree.
    files["data.dsc"] = files["data.dsc"].replace(" x s\n", " x f\n").replace(" z s\n", " z f\n")
    return files
