import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guide import GuideClassifier, importance

ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / "tests/fixtures/discriminant/guide-46.3"
IRIS = ROOT / "tests/fixtures/datasets/iris.csv"


def captures():
    for path in sorted(CAPTURES.glob("lda-*/case.json")):
        yield pytest.param(path.parent, id=path.parent.name)


@pytest.mark.parametrize("directory", list(captures()))
def test_lda_reproduces_native_captures(directory):
    # The core reproduces every frozen LDA capture; the estimator must give
    # native's training predictions from the same settings.
    case = json.loads((directory / "case.json").read_text())
    result = json.loads((directory / "manifest.json").read_text())["result"]
    X = pd.DataFrame(case["rows"], columns=case["features"])
    for name in case.get("categorical", []):
        X[name] = X[name].astype("category")
    model = GuideClassifier(
        node_model="lda_splits" if case.get("constant_leaves") else "lda",
        max_depth=case.get("depth", 1),
        min_samples_leaf=case["min_node"],
        pruning="none",
    ).fit(X, case["targets"])
    if "predicted" in result:
        native = result["predicted"]
    else:
        native = [row["predicted"] for row in result["fitted"]]
    assert model.predict(X).tolist() == native


def test_lda_classifier_fits_iris_and_round_trips():
    data = pd.read_csv(IRIS)
    X, y = data.drop(columns="Species"), data["Species"]
    model = GuideClassifier(node_model="lda", min_samples_leaf=10, pruning="cv").fit(X, y)
    assert model.score(X, y) > 0.9
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.predict_proba(X), proba)
    assert '"discriminant"' in model.to_json()


def test_lda_importance_scores_with_constant_trees():
    data = pd.read_csv(IRIS)
    X, y = data.drop(columns="Species"), data["Species"]
    lda = importance(GuideClassifier(node_model="lda"), X, y, permutations=20)
    constant = importance(GuideClassifier(), X, y, permutations=20)
    assert lda["scores"] == constant["scores"]


def test_lda_fits_categorical_features_as_dummies():
    data = pd.read_csv(IRIS)
    X, y = data.drop(columns="Species"), data["Species"]
    X = X.assign(group=np.where(X["Sepal.Length"] > 5.8, "a", "b"))
    model = GuideClassifier(node_model="lda").fit(X, y)
    assert model.score(X, y) > 0.9
    assert model._schema[-1]["role"] == "both"
    with pytest.raises(ValueError, match="role fit or both"):
        GuideClassifier(node_model="lda", feature_roles={"group": "split"}).fit(X, y)


def test_lda_splits_classify_with_constant_leaves():
    data = pd.read_csv(IRIS)
    X, y = data.drop(columns="Species"), data["Species"]
    model = GuideClassifier(node_model="lda_splits", max_depth=6, pruning="none").fit(X, y)
    assert model.score(X, y) > 0.95
    fitted = model.to_json()
    assert '"discriminant"' in fitted and '"classification"' in fitted


def test_lda_rejects_unsupported_options():
    data = pd.read_csv(IRIS)
    X, y = data.drop(columns="Species"), data["Species"]
    with pytest.raises(ValueError, match="node_model must be"):
        GuideClassifier(node_model="qda").fit(X, y)
    with pytest.raises(ValueError, match="unequal class costs"):
        GuideClassifier(node_model="lda", class_costs=[[0, 1, 2], [1, 0, 1], [1, 1, 0]]).fit(X, y)
