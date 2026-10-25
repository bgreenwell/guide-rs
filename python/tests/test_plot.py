import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

from guide_rs import GuideClassifier, GuideForestRegressor, GuideRegressor, plot_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def texts(annotations):
    return [a.get_text() for a in annotations]


def test_classification_tree_shows_splits_classes_and_counts():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(size=200), "g": rng.choice(["a", "b", "c"], 200)})
    y = np.where((X.x < 0.5) & (X.g != "c"), "low", "high")
    model = GuideClassifier(pruning="none", max_depth=2).fit(X, y)
    labels = texts(plot_tree(model))
    assert len(labels) == 2 * model.get_n_leaves() - 1
    assert any(label.startswith("x <= ") for label in labels)
    assert any(label.startswith("g in {") for label in labels)
    assert {label.split("\n")[0] for label in labels} >= {"low", "high"}
    assert labels[-1].endswith("n = 200")  # the root is drawn last


def test_simple_linear_leaves_show_their_equation():
    x = np.linspace(-1, 1, 200)
    X = pd.DataFrame({"x": x, "z": np.repeat([0.0, 1.0], 100)})
    y = np.where(X.z == 0, 1 + 2 * x, 3 - 4 * x)
    model = GuideRegressor(node_model="linear", pruning="none", max_depth=1,
                           feature_roles={"x": "fit", "z": "split"}).fit(X, y)
    labels = texts(plot_tree(model))
    assert any(label.startswith("1 + 2 x\n") for label in labels)
    assert any(label.startswith("3 - 4 x\n") for label in labels)


def test_max_depth_truncates_and_forest_trees_are_selectable():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(300, 3))
    y = X[:, 0] + (X[:, 1] > 0.5) + rng.normal(scale=0.05, size=300)
    model = GuideRegressor(pruning="none", max_depth=4).fit(X, y)
    assert "...\n" in "".join(texts(plot_tree(model, max_depth=1)))
    forest = GuideForestRegressor(n_estimators=3, max_depth=2).fit(X, y)
    roots = [json.loads(forest.to_json())["trees"][t]["root"]["sample_count"] for t in range(3)]
    for t, n in enumerate(roots):
        assert texts(plot_tree(forest, tree=t))[-1].endswith(f"n = {n}")
