import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_estimators import column, estimator_for

from guide import (
    GuideClassifier,
    GuideForestClassifier,
    GuideForestRegressor,
    GuideRegressor,
    _core,
    importance,
)
from guide._core import Model

ROOT = Path(__file__).resolve().parents[2]
IMPORTANCE = sorted((ROOT / "tests/fixtures/importance/guide-46.3").iterdir())
pytestmark = pytest.mark.filterwarnings("ignore:Some training rows have no out-of-bag")


@pytest.fixture
def iris():
    data = pd.read_csv(ROOT / "tests/fixtures/datasets/iris.csv")
    return data.drop(columns="Species"), data["Species"]


def test_feature_roles():
    rng = np.random.default_rng(5)
    n = 400
    X = pd.DataFrame({"x": rng.uniform(-1, 1, n), "z": rng.uniform(-1, 1, n),
                      "noise": rng.uniform(-1, 1, n)})
    y = np.where(X["z"] < 0, 2.0, -2.0) * X["x"] + 0.5 * X["noise"]
    # z only splits and x only fits: the tree splits on z, and leaves regress on x.
    model = GuideRegressor(
        node_model="linear", pruning="none", max_depth=1,
        feature_roles={"x": "fit", "z": "split", "noise": "excluded"},
    ).fit(X, y)
    saved = json.loads(model.to_json())
    assert [f["role"] for f in saved["schema"]] == ["fit", "split", "excluded"]
    split = saved["trees"][0]["root"]["split"]
    assert split["feature"] == 1
    leaf = saved["trees"][0]["root"]["left"]["model"]
    assert {t["feature"] for t in leaf["terms"]} == {0}
    # Indices work as keys, and roles override the node-model default.
    by_index = GuideRegressor(node_model="linear", pruning="none", max_depth=1,
                              feature_roles={0: "fit", 1: "split", 2: "excluded"}).fit(X, y)
    assert by_index.to_json() == model.to_json()
    with pytest.raises(ValueError, match="feature role must be one of"):
        GuideRegressor(feature_roles={"x": "flag"}).fit(X, y)
    with pytest.raises(ValueError, match="not a column name"):
        GuideRegressor(feature_roles={"w": "split"}).fit(X, y)


def test_forest_matches_the_core_ensemble(iris):
    X, y = iris
    forest = GuideForestClassifier(n_estimators=10, max_samples=0.8, random_state=3).fit(X, y)
    options = {"task": {"kind": "classification"}, "pruning": {"kind": "none"}, "seed": 3}
    schema = [{"name": c, "feature_type": {"kind": "numeric"}} for c in X.columns]
    core = Model.fit(
        json.dumps(schema), [X[c].tolist() for c in X.columns], list(y), json.dumps(options),
        ensemble=json.dumps({"kind": "forest", "trees": 10, "sample_fraction": 0.8}),
    )
    assert json.loads(forest.to_json()) == json.loads(core.to_json())
    assert (forest.predict(X) == y).mean() > 0.9
    np.testing.assert_allclose(forest.predict_proba(X).sum(axis=1), 1.0)
    assert forest.apply(X).shape == (150, 10)
    assert 0 <= forest.oob_loss_ < 0.2
    restored = pickle.loads(pickle.dumps(forest))
    np.testing.assert_array_equal(restored.predict_proba(X), forest.predict_proba(X))


def test_forest_options(iris):
    features, _ = iris
    X, target = features.iloc[:, 1:], features.iloc[:, 0]
    half = GuideForestRegressor(n_estimators=5, max_features=0.5).fit(X, target)
    assert json.loads(half.to_json())["options"]["mtry"] == 2
    fixed = GuideForestRegressor(n_estimators=5, max_features=1).fit(X, target)
    assert json.loads(fixed.to_json())["options"]["mtry"] == 1
    default = GuideForestRegressor(n_estimators=5).fit(X, target)
    assert json.loads(default.to_json())["options"]["mtry"] == 1  # a third of 3 features
    bagged = GuideForestRegressor(kind="bagging", n_estimators=5, pruning="cost_complexity",
                                  ccp_alpha=0.01).fit(X, target)
    assert json.loads(bagged.to_json())["ensemble"]["kind"] == "bagging"
    assert np.corrcoef(bagged.predict(X), target)[0, 1] > 0.8
    with pytest.raises(ValueError, match="max_features requires kind='forest'"):
        GuideForestRegressor(kind="bagging", max_features=2).fit(X, target)
    with pytest.raises(ValueError, match="cross-validation"):
        GuideForestRegressor(kind="bagging", pruning="cv").fit(X, target)
    with pytest.raises(ValueError, match="kind must be"):
        GuideForestRegressor(kind="boosting").fit(X, target)


def fixture_data(path):
    options = json.loads((path / "options.json").read_text())
    dataset = json.loads((path / "dataset.json").read_text())
    schema = dataset["schema"]
    categorical = [f["feature_type"]["kind"] == "categorical" for f in schema]
    columns = [column([r[j] for r in dataset["rows"]], cat) for j, cat in enumerate(categorical)]
    X = pd.DataFrame({
        f["name"]: pd.Series(values, dtype=object if cat else float)
        for f, values, cat in zip(schema, columns, categorical)
    })
    y = column(dataset["targets"], options["task"]["kind"] == "classification")
    return options, schema, columns, X, y


@pytest.mark.parametrize("path", IMPORTANCE, ids=[p.name for p in IMPORTANCE])
def test_importance_matches_core_and_native_sets(path):
    options, schema, columns, X, y = fixture_data(path)
    io = json.loads((path / "importance.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    result = importance(estimator_for(options, schema), X, y, permutations=io["permutations"],
                        max_depth=io["max_depth"], random_state=io["seed"])
    core = _core.importance(json.dumps(schema), columns, y, json.dumps(options), json.dumps(io))
    assert result == json.loads(core)
    native = {s["feature"] for s in manifest["result"]["scores"] if s["group"] in "AB"}
    ours = {s["feature"] for s in result["scores"]
            if s["group"] in ("highly_important", "likely_important")}
    assert ours == native == set(manifest["expected_important"])


def test_importance_requires_a_supported_single_tree(iris):
    X, y = iris
    with pytest.raises(TypeError, match="GuideRegressor or GuideClassifier"):
        importance(GuideForestClassifier(), X, y)
    with pytest.raises(ValueError, match="default priors and costs"):
        importance(GuideClassifier(class_priors=[0.2, 0.3, 0.5]), X, y, permutations=2)
    estimator = GuideClassifier()
    result = importance(estimator, X, y, permutations=20)
    assert {s["feature"] for s in result["scores"]} == set(X.columns)
    assert not hasattr(estimator, "_model")


def test_n_jobs_does_not_change_results(iris):
    X, y = iris
    fits = [GuideForestClassifier(n_estimators=30, n_jobs=n).fit(X, y).to_json()
            for n in (None, 1, 3, -2)]
    assert len(set(fits)) == 1
    trees = [GuideClassifier(cv_folds=5, n_jobs=n).fit(X, y).to_json() for n in (1, 4)]
    assert trees[0] == trees[1]
    scores = [importance(GuideClassifier(n_jobs=n), X, y, permutations=10) for n in (1, 3)]
    assert scores[0] == scores[1]
    for bad in (0, 1.5, "all", True):
        with pytest.raises(ValueError, match="n_jobs must be"):
            GuideClassifier(n_jobs=bad).fit(X, y)
