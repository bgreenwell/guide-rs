"""Matplotlib drawing of fitted GUIDE trees."""

import json

from sklearn.utils.validation import check_is_fitted


def _number(x, precision):
    return f"{x:.{precision}g}"


def _split_label(split, names, precision):
    kind = split["kind"]
    if kind == "linear":
        (a, b), (u, v) = split["coefficients"], split["features"]
        text = (f"{_number(a, precision)} {names[u]} + {_number(b, precision)} {names[v]}"
                f" <= {_number(split['threshold'], precision)}")
    elif kind == "missing":
        return f"{names[split['feature']]} = NA"
    elif kind == "numeric":
        text = f"{names[split['feature']]} <= {_number(split['threshold'], precision)}"
    elif kind == "categorical":
        levels = split["left_categories"]
        shown = ", ".join(levels[:4]) + (", ..." if len(levels) > 4 else "")
        text = f"{names[split['feature']]} in {{{shown}}}"
    else:
        start, end = (_number(split[k], precision) for k in ("start", "end"))
        text = (f"{names[split['feature']]} mod {_number(split['period'], precision)}"
                f" in ({start}, {end}]")
    if split.get("missing_codes"):
        codes = split["missing_codes"]
        text += f"\nor NA with {names[codes['flag']]} in {{{', '.join(codes['left'])}}}"
    elif split.get("missing_left"):
        text += " or NA"
    return text


def _leaf_label(model, classes, names, precision):
    kind = model["kind"]
    if kind == "classification":
        return classes[model["predicted_class"]]
    if kind == "constant":
        return _number(model["value"], precision)
    if kind == "quantiles":
        return f"({_number(model['lower'], precision)}, {_number(model['upper'], precision)})"
    terms = model["terms"]
    if len(terms) > 2 or any(t["power"] != 1 or t["category"] is not None
                             or t["missing_indicator"] for t in terms):
        return f"linear ({len(terms)} term{'s' * (len(terms) != 1)})"
    # Terms are standardized, (x - center) / scale; show the raw-scale equation.
    intercept = model["intercept"] - sum(
        c * t["center"] / t["scale"] for c, t in zip(model["coefficients"], terms))
    text = _number(intercept, precision)
    for c, t in zip(model["coefficients"], terms):
        slope = c / t["scale"]
        text += f" {'-' if slope < 0 else '+'} {_number(abs(slope), precision)} {names[t['feature']]}"
    return text


def plot_tree(estimator, *, tree=0, max_depth=None, precision=3, fontsize=None, ax=None):
    """Draw a fitted GUIDE tree with matplotlib.

    Internal nodes show the split condition; samples satisfying it go left
    ("yes"). Leaves show the predicted class, mean response, quantiles or
    linear model, with the training sample count. Classification leaves are
    colored by predicted class.

    Parameters
    ----------
    estimator : fitted guide-rs estimator
    tree : int, default=0
        Which tree of a forest to draw.
    max_depth : int, default=None
        Draw only this many levels; deeper subtrees are shown as "...".
    precision : int, default=3
        Significant digits for thresholds and predictions.
    fontsize : float, default=None
        Text size; by default it shrinks with the number of leaves.
    ax : matplotlib Axes, default=None
        Axes to draw on; the current axes by default.

    Returns
    -------
    list of matplotlib Annotation
        One per drawn node.
    """
    import matplotlib.pyplot as plt

    check_is_fitted(estimator)
    model = json.loads(estimator.to_json())
    names = [f["name"] for f in model["schema"]]
    classes = model["classes"]
    root = model["trees"][tree]["root"]
    ax = plt.gca() if ax is None else ax

    def children(node, depth):
        if node.get("left") is None or (max_depth is not None and depth >= max_depth):
            return []
        return [node["left"], node["right"]]

    # Leaves take consecutive x positions; parents sit above their children.
    positions, edges, order = {}, [], []

    def place(node, depth, next_x):
        kids = children(node, depth)
        if not kids:
            positions[id(node)] = (next_x, depth)
            order.append((node, depth))
            return next_x + 1
        for kid in kids:
            next_x = place(kid, depth + 1, next_x)
        xs = [positions[id(k)][0] for k in kids]
        positions[id(node)] = ((xs[0] + xs[-1]) / 2, depth)
        order.append((node, depth))
        edges.extend((node, k, side) for k, side in zip(kids, ("yes", "no")))
        return next_x

    width = place(root, 0, 0)
    depth = max(d for _, d in positions.values())
    if fontsize is None:
        fontsize = max(6.0, min(11.0, 60.0 / max(width, 1)))
    colors = plt.get_cmap("tab10")
    annotations = []
    for node, d in order:
        x, y = positions[id(node)]
        truncated = node.get("left") is not None and not children(node, d)
        if children(node, d):
            text, face = _split_label(node["split"], names, precision), "white"
        elif truncated:
            text, face = "...", "whitesmoke"
        else:
            text = _leaf_label(node["model"], classes, names, precision)
            face = "whitesmoke"
            if node["model"]["kind"] == "classification":
                # A light opaque tint, so edges do not show through the box.
                rgb = colors(node["model"]["predicted_class"] % 10)[:3]
                face = tuple(1 - 0.35 * (1 - c) for c in rgb)
        annotations.append(ax.annotate(
            f"{text}\nn = {node['sample_count']}", (x, -y), ha="center", va="center",
            fontsize=fontsize, bbox={"boxstyle": "round", "fc": face, "ec": "0.4"},
            zorder=2,
        ))
    for parent, child, side in edges:
        (x0, y0), (x1, y1) = positions[id(parent)], positions[id(child)]
        ax.plot([x0, x1], [-y0, -y1], color="0.5", lw=1, zorder=1)
        ax.text((x0 + x1) / 2, -(y0 + y1) / 2, side, ha="center", va="center",
                fontsize=fontsize * 0.85, color="0.3",
                bbox={"fc": "white", "ec": "none", "pad": 0.5}, zorder=1.5)
    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(-depth - 0.5, 0.5)
    ax.set_axis_off()
    return annotations
