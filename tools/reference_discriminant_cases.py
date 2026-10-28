"""Single-node LDA cases: discriminant coordinates, variable selection and leaf rules.

Each design has fewer rows than ceil(2.5 * minimum node size), so native GUIDE
cannot split the root and the whole fit is one LDA leaf.
"""
import random


def make(name, n, shifts, *, classes=2, seed=1, missing=(), duplicate=None, constant=False,
         min_node=None):
    """`shifts[j]` is the class-index multiple added to numeric column j.

    `missing` lists columns whose every seventh value (offset by the column
    index) is missing; `duplicate` copies an existing column; `constant` adds a
    constant column.
    """
    rng = random.Random(seed)
    labels = [f"k{i % classes}" for i in range(n)]
    rng.shuffle(labels)
    names = [f"x{j}" for j in range(len(shifts))]
    rows = []
    for i, label in enumerate(labels):
        c = int(label[1:])
        row = [round(rng.gauss(0.0, 1.0) + s * c + 10.0, 4) for s in shifts]
        for j in missing:
            if (i + j) % 7 == 0:
                row[j] = None
        rows.append(row)
    if duplicate is not None:
        names.append(f"x{duplicate}copy")
        for row in rows:
            row.append(row[duplicate])
    if constant:
        names.append("flat")
        for row in rows:
            row.append(5.0)
    return {
        "name": "lda-leaf-" + name,
        "features": names,
        "rows": rows,
        "targets": labels,
        "min_node": min_node if min_node is not None else n // 2,
    }


def exact(name, f_values, *, n=60, seed=11):
    """Two balanced classes; column j has one-way ANOVA F statistic f_values[j].

    Within-class residuals are rescaled so the within mean square is 1, and the
    class shift then gives the requested between sum of squares. The F values
    are F(1, 58) upper quantiles of chosen p-values, computed once with
    scipy.stats.f.isf so these tools need only the standard library.
    """
    rng = random.Random(seed)
    labels = [i % 2 for i in range(n)]
    rng.shuffle(labels)
    columns = []
    for f in f_values:
        e = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for c in (0, 1):
            members = [i for i in range(n) if labels[i] == c]
            m = sum(e[i] for i in members) / len(members)
            for i in members:
                e[i] -= m
        scale = ((n - 2) / sum(v * v for v in e)) ** 0.5
        shift = (f * n / ((n // 2) * (n - n // 2))) ** 0.5
        columns.append([round(10.0 + scale * e[i] + shift * labels[i], 6) for i in range(n)])
    return {
        "name": "lda-leaf-" + name,
        "features": [f"x{j}" for j in range(len(f_values))],
        "rows": [[col[i] for col in columns] for i in range(n)],
        "targets": [f"k{c}" for c in labels],
        "min_node": n // 2,
    }


# F(1, 58) upper quantiles: p = 1e-6, 0.10, 0.12, 0.15, 0.19, 0.24, 0.30, 0.90.
F_P = {1e-6: 29.921420828403722, 0.10: 2.794088637670471, 0.12: 2.490147782394021,
       0.15: 2.128254313688923, 0.19: 1.7585839915185515, 0.24: 1.4093733911128097,
       0.30: 1.0936627458815595, 0.90: 0.015929667069233898}


def cases():
    result = {}

    def add(value):
        result[value["name"]] = value
    # Two classes: one coordinate; the count rule's three branches.
    add(make("two-strong", 60, [1.5, 1.2, 0.0, 0.0]))
    add(make("two-one-signal", 60, [1.5, 0.3, 0.2, 0.0, 0.0]))
    add(make("two-null", 40, [0.0, 0.0, 0.0, 0.0], seed=3))
    add(make("two-wide", 80, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0], seed=4))
    # Three and four classes: two coordinates.
    add(make("three", 90, [1.0, 0.6, 0.3, 0.0], classes=3, seed=5))
    add(make("three-weak", 60, [0.4, 0.2, 0.0, 0.0, 0.0], classes=3, seed=6))
    add(make("four", 100, [0.8, 0.5, 0.5, 0.0], classes=4, seed=7))
    # Node-mean imputation and missing-value indicator columns.
    add(make("missing", 70, [1.2, 0.8, 0.4, 0.0], classes=3, seed=8, missing=(0, 2)))
    # Collinear and constant columns.
    add(make("duplicate", 60, [1.2, 0.6, 0.0], classes=3, seed=9, duplicate=0))
    add(make("constant", 60, [1.2, 0.6, 0.0], classes=3, seed=10, constant=True))
    # Fallback thresholds with K = 5: one column below 0.4/5 keeps the others
    # below 0.8/3 (0.24 in, 0.30 out); none keeps those below 0.8/4 (0.19 in, 0.24 out).
    add(exact("relaxed-one", [F_P[1e-6], F_P[0.12], F_P[0.24], F_P[0.30], F_P[0.90]]))
    add(exact("relaxed-none", [F_P[0.10], F_P[0.15], F_P[0.19], F_P[0.24], F_P[0.90]]))
    return result
