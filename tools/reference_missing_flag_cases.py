"""Constant-node designs with GUIDE missing-value flag (M) variables."""
from reference_cases import case


def make(name, classification, kind, rows, targets, *, depth=1, extra=(), competitor=False,
         difference=None):
    """`rows` hold [parent, flag] (plus [z] with a competitor). The flag column
    follows its parent, as GUIDE's DSC format requires."""
    features = [("x", kind), ("x_", "categorical")] + ([("z", "numeric")] if competitor else [])
    rows = [list(r) for r in rows] + [list(r) for r in extra]
    targets = list(targets) + [None] * len(extra)
    value = case("missing-flag-" + name, classification, features, rows, targets, depth=depth)
    flag = value["dataset"]["schema"][1]
    flag["role"] = "flag"
    flag["flags"] = "x"
    # Prediction-only rows: an unseen code, no code, then every training code.
    value["prediction_rows"] = len(extra)
    value["native_difference"] = difference
    return value


# GUIDE only considers flag code subsets that contain the node's first code,
# so when the best split leaves that code out it chooses a worse one.
ANCHORED = "first-code-anchored-subsets"


def probe_rows(codes):
    tail = [[None, "Z"], [None, None]] + [[None, c] for c in codes]
    return tail


def cases():
    result = {}

    def add(value):
        result[value["name"]] = value

    # Only missing values with code A differ; a depth-2 tree then separates
    # the remaining missing values from observed ones.
    rows, ys = [], []
    for i in range(120):
        if i % 10 < 3:
            code = "A" if i % 2 == 0 else "C"
            rows.append([None, code]); ys.append(20 if code == "A" else 0)
        else:
            rows.append([i % 60, "D"]); ys.append(10)
    add(make("numeric-code-only", False, "numeric", rows, ys, depth=2, extra=probe_rows("AC")))

    def three_codes(left, kind, classification=False):
        rows, ys = [], []
        for i in range(300):
            if i % 5 < 2:
                code = "ABC"[(i // 5) % 3]
                rows.append([None, code]); low = code in left
            elif kind == "numeric":
                x = (i * 7) % 50
                rows.append([x, "D"]); low = x <= 25
            else:
                g = "pqrs"[i % 4]
                rows.append([g, "D"]); low = g in "pq"
            ys.append(("a" if low else "b") if classification else (0 if low else 10))
        return rows, ys

    for kind in ["numeric", "categorical"]:
        for left in ["A", "AB", "ABC", ""]:
            rows, ys = three_codes(left, kind)
            difference = (ANCHORED if kind == "numeric" and left == "" else None)
            add(make(f"{kind}-left-{left or 'none'}", False, kind, rows, ys,
                     extra=probe_rows("ABC") + ([[60, "D"], [-5, "D"]] if kind == "numeric"
                                                else [["t", "D"], ["p", "D"]]),
                     difference=difference))
        rows, ys = three_codes("A", kind, classification=True)
        add(make(f"{kind}-classification", True, kind, rows, ys, extra=probe_rows("ABC")))

    # Missing values without a code: they never satisfy a code condition.
    rows, ys = three_codes("A", "numeric")
    for r in rows[::25]:
        if r[0] is None:
            r[1] = None
    add(make("numeric-uncoded", False, "numeric", rows, ys, extra=probe_rows("ABC")))

    # Selection: x carries signal only through its missing codes. With the
    # flag, x outranks a competitor z that wins when the codes are ignored.
    # Exact decimal noise in [-1.5, 1.5]: platform math libraries would
    # change the last digits of random.gauss, and so the captured inputs.
    rows, ys = [], []
    for i in range(300):
        z, e = i % 2, ((i * 7919) % 301 - 150) / 100
        if i % 5 < 2:
            code = "A" if (i // 5) % 2 == 0 else "C"
            rows.append([None, code, z]); ys.append((11.5 if code == "A" else 8.5) + 0.6 * z + e)
        else:
            rows.append([(i * 7) % 50, "D", z]); ys.append(10 + 0.6 * z + e)
    add(make("selection", False, "numeric", rows, ys, competitor=True,
             extra=[r + [0] for r in probe_rows("AC")], difference=ANCHORED))
    for left in ["B", "BC"]:
        rows, ys = three_codes(left, "numeric")
        add(make(f"numeric-left-{left}", False, "numeric", rows, ys,
                 extra=probe_rows("ABC"), difference=ANCHORED))
    rows, ys = [], []
    for i in range(240):
        if i % 4 == 0:
            code = "ABC"[(i // 4) % 3]
            rows.append([None, code]); ys.append((0 if code == "C" else 10) + (i // 4) % 5 * 0.01)
        else:
            rows.append([i % 6, "D"]); ys.append(10 + i % 5 * 0.01)
    add(make("missing-only-C", False, "numeric", rows, ys, extra=probe_rows("ABC"),
             difference=ANCHORED))

    # Held-out designs, captured after the routing and orientation rules
    # above were fixed, to check that those rules generalize.
    rows, ys = three_codes("AB", "categorical")
    for r in rows[::25]:
        if r[0] is None:
            r[1] = None
    add(make("holdout-categorical-uncoded", False, "categorical", rows, ys,
             extra=probe_rows("ABC") + [["t", "D"]]))
    rows, ys = [], []
    for i in range(240):
        g = "pqrstu"[i % 6] if i % 4 else None
        code = None if g else "ABC"[(i // 4) % 3]
        rows.append([g, code if g is None else "D"])
        ys.append(0 if g in ("p",) or code == "B" else 10)
    add(make("holdout-categorical-unbalanced", False, "categorical", rows, ys,
             extra=probe_rows("ABC") + [["v", "D"]]))
    rows, ys = [], []
    for i in range(360):
        x = (i * 11) % 90
        if i % 6 < 2:
            code = "ABCD"[(i // 6) % 4]
            rows.append([None, code]); ys.append({"A": 0, "B": 5, "C": 5, "D": 20}[code])
        else:
            rows.append([x, "D"]); ys.append(5 if x <= 40 else 20)
    add(make("holdout-numeric-depth2", False, "numeric", rows, ys, depth=2,
             extra=probe_rows("ABCD") + [[100, "D"]]))
    rows, ys = three_codes("AC", "numeric", classification=True)
    add(make("holdout-numeric-classification", True, "numeric", rows, ys, depth=2,
             extra=probe_rows("ABC")))
    return result
