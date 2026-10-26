"""Deterministic, small datasets for the constant-node GUIDE acceptance milestone."""
import csv
import io
import math
import random


def case(name, classification, features, rows, targets, *, weights=None,
         priors=None, costs=None, depth=2, loss=None, offsets=None):
    schema = [{"name": n, "feature_type": {"kind": k}, "role": "split"}
              for n, k in features]
    options = {
        "task": {"kind": "classification"} if classification else
                {"kind": "regression", "loss": loss or {"kind": "squared_error"}},
        "node_model": {"kind": "constant"}, "max_depth": depth,
        "min_samples_leaf": 2, "min_gain": 0.0, "interactions": True,
        "linear_splits": False, "significance_level": 0.1,
        "max_split_candidates": 0, "pruning": {"kind": "none"},
        "seed": 0, "class_priors": priors, "class_costs": costs, "mtry": None,
    }
    return {"name": name, "classification": classification,
            "dataset": {"schema": schema, "rows": rows, "targets": targets,
                        "weights": weights or [], "offsets": offsets or []},
            "options": options}


def cases():
    result = {}
    def add(*args, **kwargs):
        value = case(*args, **kwargs)
        result[value["name"]] = value
    numeric = [("x", "numeric")]
    categorical = [("group", "categorical")]
    for left in (5, 10, 15):
        add(f"ls-stump-{left}", False, numeric, [[i] for i in range(20)],
            [2 if i < left else 8 for i in range(20)], depth=1)
    rows = [[i if i % 7 else None] for i in range(80)]
    add("ls-numeric-missing", False, numeric, rows,
        [2 if r[0] is not None and r[0] < 40 else 8 for r in rows], depth=1)
    rows = [[None if i % 4 == 3 else "abc"[i % 4]] for i in range(80)]
    add("ls-categorical-missing", False, categorical, rows,
        [2 if r[0] in ("a", "b") else 8 for r in rows], depth=1)
    add("ls-mixed-predictors", False,
        [("noise", "numeric"), ("group", "categorical"), ("signal", "numeric")],
        [[(i * 37) % 83, "abcd"[i % 4], i] for i in range(80)],
        [2 if i < 40 else 8 for i in range(80)], depth=1)
    xor = [[(i // 20) % 2, (i // 40) % 2] for i in range(80)]
    for classification in (False, True):
        add("class-interaction" if classification else "ls-interaction", classification,
            [("a", "numeric"), ("b", "numeric")], xor,
            [("yes" if a != b else "no") if classification else
             (8 if a != b else 2) for a, b in xor])
    add("ls-weights", False, numeric, [[i] for i in range(40)],
        [i % 3 + (4 if i >= 19 else 0) for i in range(40)],
        weights=[1 + i % 5 for i in range(40)], depth=1)
    add("ls-tied-predictors", False, [("first", "numeric"), ("second", "numeric")],
        [[i, i] for i in range(40)], [2 if i < 20 else 8 for i in range(40)], depth=1)
    add("ls-constant-response", False, numeric, [[i] for i in range(40)], [3] * 40)
    add("class-balanced", True, numeric, [[i] for i in range(80)],
        ["a" if i < 40 else "b" for i in range(80)], depth=1)
    add("class-imbalanced", True, numeric, [[i] for i in range(80)],
        ["a" if i < 60 else "b" for i in range(80)], depth=1)
    add("class-multiclass", True, numeric, [[i] for i in range(90)],
        ["abc"[i // 30] for i in range(90)])
    add("class-categorical", True, categorical, [["abcd"[i % 4]] for i in range(80)],
        ["a" if i % 4 < 2 else "b" for i in range(80)], depth=1)
    add("class-numeric-missing", True, numeric, [[i if i % 7 else None] for i in range(80)],
        ["a" if i % 7 and i < 40 else "b" for i in range(80)], depth=1)
    add("class-categorical-missing", True, categorical, rows,
        ["a" if r[0] in ("a", "b") else "b" for r in rows], depth=1)
    # Both classes occur in both regions: exercise prior adjustment and costs,
    # rather than merely rechecking an unchanged pure split.
    target = ["a" if i % 10 < (8 if i < 40 else 3) else "b" for i in range(80)]
    add("class-priors", True, numeric, [[i] for i in range(80)], target,
        priors=[0.2, 0.8], depth=1)
    add("class-costs", True, numeric, [[i] for i in range(80)], target,
        costs=[[0.0, 1.0], [4.0, 0.0]], depth=1)
    add("class-constant-response", True, numeric, [[i] for i in range(40)], ["a"] * 40)
    # Zero-gain splits: every cut leaves both child means at 5, yet the
    # residual signs still associate with x, so GUIDE splits anyway.
    pattern = [(0, [0, 10]), (1, [5, 5]), (2, [5, 5]), (3, [0, 10])]
    zero_x = [x for x, pair in pattern for _ in range(3) for _ in pair]
    zero_y = [t for _, pair in pattern for _ in range(3) for t in pair]
    add("ls-zero-gain-numeric", False, numeric, [[x] for x in zero_x], zero_y, depth=1)
    add("ls-zero-gain-categorical", False, categorical, [["abcd"[x]] for x in zero_x], zero_y, depth=1)
    # Same-class sibling leaves merge bottom-up, regardless of association
    # strength; a split survives only if some leaf below changes the class.
    shares = [39, 21, 21, 39]
    add("class-same-class-stump", True, numeric, [[x] for x in range(4) for _ in range(40)],
        ["a" if k < shares[x] else "b" for x in range(4) for k in range(40)], depth=1)
    pair = [("A", "numeric"), ("B", "numeric")]
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    def merge_case(name, last, shares=None, **kwargs):
        share = dict(zip(cells, shares or [40, 40, 38, last]))
        add(name, True, pair, [[a, b] for a, b in cells for _ in range(40)],
            ["a" if k < share[a, b] else "b" for a, b in cells for k in range(40)],
            depth=2, **kwargs)
    merge_case("class-same-class-kept", 12)
    merge_case("class-same-class-cascade", 24)
    merge_case("class-same-class-costs", 24, costs=[[0.0, 1.0], [3.0, 0.0]])
    # With priors or unequal costs, count-weighted Gini can rise at a split
    # whose children predict different classes; GUIDE still makes the split.
    merge_case("class-impurity-increase-priors", 24, priors=[0.3, 0.7])
    merge_case("class-impurity-increase-priors-2", None, shares=[38, 38, 40, 12],
               priors=[0.3, 0.7])
    merge_case("class-impurity-increase-costs", None, shares=[0, 12, 36, 0],
               costs=[[0.0, 1.0], [3.0, 0.0]])
    # Categorical orientation: the side with fewer levels goes left, whatever
    # its row count or response. Each spec lists (level, rows, low response).
    for k, spec in enumerate([[("p", 10, 1), ("q", 10, 1), ("r", 40, 0)],
                              [("p", 40, 1), ("q", 10, 0), ("r", 10, 0)],
                              [("p", 20, 0), ("q", 20, 0), ("r", 20, 1)],
                              [("p", 20, 0), ("q", 20, 1), ("r", 20, 1)]], 1):
        rows = [[level] for level, n, _ in spec for _ in range(n)]
        low = [flag for _, n, flag in spec for _ in range(n)]
        add(f"ls-categorical-orientation-{k}", False, categorical, rows,
            [0 if f else 10 for f in low], depth=1)
        add(f"class-categorical-orientation-{k}", True, categorical, rows,
            ["a" if f else "b" for f in low], depth=1)
    # Multiclass splits on categorical variables with more than 11 levels
    # search cuts on the first two discriminant coordinates. Designs use an
    # odd total number of categories (counting NA), so the fewer-levels-left
    # orientation never ties. random.random() is platform independent.
    def many_levels(n, classes, seed, missing=0.0):
        rnd, rows, ys = random.Random(seed), [], []
        for level in range(n):
            p = [rnd.random() ** 2 for _ in range(classes)]
            for _ in range(12):
                rows.append([f"L{level:02d}"])
                ys.append("abcdefghijkl"[rnd.choices(range(classes), p)[0]])
        holes = random.Random(seed + 999)
        return [[None] if holes.random() < missing else r for r in rows], ys
    for name, (n, classes, seed, missing, depth) in {
            "class-multiclass-many-13": (13, 3, 511, 0.0, 1),
            "class-multiclass-many-21": (21, 3, 502, 0.0, 1),
            "class-multiclass-many-second-coordinate": (19, 3, 50, 0.0, 1),
            "class-multiclass-many-12-classes": (15, 12, 503, 0.0, 1),
            "class-multiclass-many-missing": (14, 3, 505, 0.15, 1),
            "class-multiclass-many-depth2": (25, 4, 504, 0.0, 2)}.items():
        rows, ys = many_levels(n, classes, seed, missing)
        add(name, True, categorical, rows, ys, depth=depth)
    return result


def pruning_cases():
    """LOOCV removes dependence on the native and Rust random-number streams."""
    import copy
    result = {}
    def add(value, name, pruning):
        value = copy.deepcopy(value)
        value.update(name=name, family="pruning")
        value["options"]["pruning"] = pruning
        result[name] = value
    original = cases()
    for se, suffix in [(0.25, "default"), (4.6, "se-boundary"), (5.0, "wide-se")]:
        add(original["ls-stump-10"], f"ls-loocv-{suffix}",
            {"kind": "cross_validation", "folds": 20, "standard_errors": se})
    weighted = copy.deepcopy(original["ls-stump-10"])
    weighted["dataset"]["weights"] = [1 + i % 5 for i in range(20)]
    add(weighted, "ls-loocv-weights", {"kind": "cross_validation", "folds": 20, "standard_errors": 0.25})
    for tied, means in [(True, [0, 2, 8, 10]), (False, [0, 1, 7, 10])]:
        value = case("unused", False, [("x", "numeric")], [[i] for i in range(24)],
                     [means[i // 6] for i in range(24)], depth=2)
        add(value, "ls-loocv-" + ("tied-path" if tied else "path"),
            {"kind": "cross_validation", "folds": 24, "standard_errors": 0.25})
        if tied:
            add(value, "ls-fixed-ties", {"kind": "fixed_leaves", "max_leaves": 3})
    for left, suffix in [(10, "balanced"), (15, "imbalanced")]:
        value = case("unused", True, [("x", "numeric")], [[i] for i in range(20)],
                     ["a" if i < left else "b" for i in range(20)], depth=1)
        add(value, "class-loocv-" + suffix,
            {"kind": "cross_validation", "folds": 20, "standard_errors": 0.25})
    for setting in ["priors", "costs"]:
        value = case("unused", True, [("x", "numeric")], [[i] for i in range(20)],
                     ["a" if i < 10 else "b" for i in range(20)], depth=1,
                     priors=[0.2, 0.8] if setting == "priors" else None,
                     costs=[[0.0, 1.0], [4.0, 0.0]] if setting == "costs" else None)
        add(value, "class-loocv-" + setting,
            {"kind": "cross_validation", "folds": 20, "standard_errors": 0.25})
    value = case("unused", True, [("x", "numeric")], [[i] for i in range(24)],
                 ["abcd"[i // 6] for i in range(24)], depth=2)
    add(value, "class-loocv-multiclass", {"kind": "cross_validation", "folds": 24, "standard_errors": 0.25})
    add(value, "class-fixed-root-tie", {"kind": "fixed_leaves", "max_leaves": 3})
    for name in ["ls-interaction", "class-interaction"]:
        add(original[name], name.replace("interaction", "fixed-interaction"),
            {"kind": "fixed_leaves", "max_leaves": 3})
    return result


def glm_cases():
    """Poisson and quantile constant trees: GUIDE's fixed cutpoint grid and
    Poisson's positive-count children."""
    result = {}
    def add(name, rows, targets, loss, depth=1, offsets=None, features=None):
        value = case(name, False, features or [("x", "numeric")], rows, targets,
                     depth=depth, loss=loss, offsets=offsets)
        result[name] = value
    poisson, median = {"kind": "poisson"}, {"kind": "quantile", "probability": 0.5}
    rng = random.Random(46)
    xs = [rng.random() for _ in range(200)]
    rows = [[round(x, 4)] for x in xs]
    def counts(rate, n=40):
        return sum(1 for _ in range(n) if rng.random() < rate / n)
    add("poisson-cut-grid", rows, [counts(2.5 if r[0] > 0.5 else 1.0) for r in rows], poisson)
    add("poisson-zero-child", [[i + 1] for i in range(40)],
        [0 if i < 12 else 6 for i in range(40)], poisson)
    exposure = [round(rng.uniform(0.2, 3.0), 4) for _ in rows[:150]]
    add("poisson-exposure", rows[:150],
        [counts(e * (2.5 if r[0] > 0.5 else 1.0), 80) for r, e in zip(rows, exposure)],
        poisson, offsets=exposure)
    add("quantile-cut-grid", rows,
        [round((2 if r[0] > 0.5 else 0) + rng.gauss(0, 1), 4) for r in rows], median)
    pairs = [[round(rng.random(), 4), round(rng.random(), 4)] for _ in range(150)]
    add("quantile-depth-2", pairs,
        [round((2 if a > 0.5 else 0) + (1 if b > 0.3 else 0) + rng.gauss(0, 1), 4)
         for a, b in pairs], median, depth=2, features=[("a", "numeric"), ("b", "numeric")])
    return result


def all_cases():
    return {**cases(), **pruning_cases(), **glm_cases()}


def data_files(value):
    data = value["dataset"]
    names = [f["name"] for f in data["schema"]] + ["y"]
    codes = ["c" if f["feature_type"]["kind"] == "categorical" else "s"
             for f in data["schema"]] + ["d"]
    records = [row + [y] for row, y in zip(data["rows"], data["targets"])]
    if data["weights"]:
        names.append("weight")
        codes.append("w")
        records = [r + [w] for r, w in zip(records, data["weights"])]
    if data["offsets"]:
        # GUIDE's Poisson offset is the log exposure.
        names.append("offset")
        codes.append("z")
        records = [r + [repr(math.log(o))] for r, o in zip(records, data["offsets"])]
    records = [["NA" if v is None else v for v in r] for r in records]
    csv_text = io.StringIO(newline="")
    writer = csv.writer(csv_text, lineterminator="\n")
    writer.writerows([names] + records)
    return {
        "data.dsc": "data.txt\nNA\n2\n" + "".join(
            f"{i} {n} {c}\n" for i, (n, c) in enumerate(zip(names, codes), 1)),
        "data.txt": " ".join(names) + "\n" + "".join(
            " ".join(map(str, r)) + "\n" for r in records),
        "data.csv": csv_text.getvalue(),
    }
