import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from guide_rs import ALGORITHM_VERSION, GuideClassifier, GuideRegressor
from guide_rs._core import Model

ROOT = Path(__file__).resolve().parents[2]
IRIS = ROOT / "tests/fixtures/datasets/iris.csv"
PRUNING = {
    "none": lambda p: {"pruning": "none"},
    "cross_validation": lambda p: {
        "pruning": "cv",
        "cv_folds": p["folds"],
        "cv_standard_errors": p["standard_errors"],
    },
    "fixed_leaves": lambda p: {"pruning": "max_leaves", "max_leaves": p["max_leaves"]},
    "cost_complexity": lambda p: {"pruning": "cost_complexity", "ccp_alpha": p["alpha"]},
}


def rust_label(v):
    # The core labels numeric categories with Rust's float formatting.
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return None if v is None else str(v)


def estimator_for(options, schema):
    """The estimator equivalent to a core configuration, or None if the
    estimator cannot express it."""
    task, node = options["task"], options["node_model"]
    if (
        options.get("mtry") is not None
        or options.get("prediction_truncation", "none") != "none"
        or options["pruning"]["kind"] not in PRUNING
        or any(f["feature_type"]["kind"] == "periodic" for f in schema)
    ):
        return None
    params = {
        "max_depth": options["max_depth"],
        "min_samples_leaf": options["min_samples_leaf"],
        "min_gain": options["min_gain"],
        "interactions": options["interactions"],
        "significance_level": options["significance_level"],
        "max_split_candidates": options["max_split_candidates"],
        "random_state": options["seed"],
        **PRUNING[options["pruning"]["kind"]](options["pruning"]),
    }
    if task["kind"] == "classification":
        roles = {"split"}
        estimator = GuideClassifier(
            linear_splits=options["linear_splits"],
            class_priors=options["class_priors"],
            class_costs=options["class_costs"],
            **params,
        )
    else:
        loss = task["loss"]
        if options["linear_splits"] or loss["kind"] not in ("squared_error", "quantile", "poisson"):
            return None
        if node["kind"] == "constant":
            roles = {"split"}
        elif node["kind"] in ("linear", "polynomial"):
            roles = {"both"}
        else:
            return None
        estimator = GuideRegressor(
            loss=loss["kind"],
            quantile=loss.get("probability", 0.5),
            node_model=node["kind"],
            degree=node.get("degree", 2),
            **params,
        )
    expected = {
        "split" if f["feature_type"]["kind"] == "categorical" else role for f in schema
        for role in roles
    }
    if {f["role"] for f in schema} != expected:
        return None
    return estimator


def reference_cases():
    for path in sorted(ROOT.glob("tests/fixtures/*/guide-46.3/*/options.json")):
        options = json.loads(path.read_text())
        dataset = json.loads((path.parent / "dataset.json").read_text())
        if estimator_for(options, dataset["schema"]) is not None:
            yield pytest.param(options, dataset, id=path.parent.name)


CASES = list(reference_cases())


def test_reference_corpus_is_covered():
    assert len(CASES) >= 50


@pytest.mark.filterwarnings("ignore:Positive classification weights")
@pytest.mark.parametrize("options,dataset", CASES)
def test_estimator_fits_the_core_model(options, dataset):
    # The core's fits are validated against the frozen GUIDE captures; the
    # estimator must produce the same model from the same configuration.
    schema, rows, targets = dataset["schema"], dataset["rows"], dataset["targets"]
    classifying = options["task"]["kind"] == "classification"

    def column(values, categorical):
        if categorical:
            return [rust_label(v) for v in values]
        return [np.nan if v is None else float(v) for v in values]

    def fit_core():
        return Model.fit(
            json.dumps(schema),
            [column([r[j] for r in rows], f["feature_type"]["kind"] == "categorical")
             for j, f in enumerate(schema)],
            column(targets, classifying),
            json.dumps(options),
            weights=dataset.get("weights") or None,
            exposure=dataset.get("offsets") or None,
        )

    # The estimator takes only rows with an observed response.
    active = [i for i, y in enumerate(targets) if y is not None]
    X = pd.DataFrame({
        f["name"]: pd.Series(
            column([rows[i][j] for i in active], f["feature_type"]["kind"] == "categorical"),
            dtype=object if f["feature_type"]["kind"] == "categorical" else float,
        )
        for j, f in enumerate(schema)
    })
    y = column([targets[i] for i in active], classifying)
    kwargs = {}
    if dataset.get("weights"):
        kwargs["sample_weight"] = [dataset["weights"][i] for i in active]
    if dataset.get("offsets"):
        kwargs["exposure"] = [dataset["offsets"][i] for i in active]
    estimator = estimator_for(options, schema)
    try:
        core = fit_core()
    except ValueError as error:
        with pytest.raises(ValueError, match=re.escape(str(error))):
            estimator.fit(X, y, **kwargs)
        return
    estimator.fit(X, y, **kwargs)
    ours, theirs = json.loads(estimator.to_json()), json.loads(core.to_json())
    for model in (ours, theirs):
        del model["report"]["training_rows"], model["report"]["excluded_rows"]
    assert ours == theirs


@pytest.fixture
def iris():
    data = pd.read_csv(IRIS)
    return data.drop(columns="Species"), data["Species"]


def test_classifier_matches_cli_configuration(iris):
    X, y = iris
    model = GuideClassifier(random_state=42).fit(X, y)
    saved = json.loads(model.to_json())
    assert saved["algorithm_version"] == ALGORITHM_VERSION
    assert saved["classes"] == ["setosa", "versicolor", "virginica"]
    assert model.get_n_leaves() == saved["report"]["fitted_leaves"]
    assert (model.predict(X) == y).mean() > 0.9
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_[proba.argmax(axis=1)]) == list(model.predict(X))


def test_integer_labels_keep_numeric_order():
    # The core orders classes by label string ("10" < "2"); the estimator
    # reports them in numeric order and aligns probabilities accordingly.
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(300, 2))
    y = np.where(X[:, 0] < 0.33, 2, np.where(X[:, 0] < 0.66, 10, 30))
    model = GuideClassifier(pruning="none", max_depth=3).fit(X, y)
    assert list(model.classes_) == [2, 10, 30]
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], model.predict(X))
    assert (model.predict(X) == y).mean() > 0.95


def test_priors_and_costs_follow_classes_order():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(200, 1))
    y = np.where(X[:, 0] + rng.normal(scale=0.3, size=200) > 0.5, 10, 2)
    costs = [[0, 1], [1, 0]]
    base = GuideClassifier(pruning="none", max_depth=1, class_costs=costs).fit(X, y)
    # Making errors on class 2 expensive must predict class 2 at least as often.
    heavy = GuideClassifier(pruning="none", max_depth=1, class_costs=[[0, 20], [1, 0]]).fit(X, y)
    assert (heavy.predict(X) == 2).sum() >= (base.predict(X) == 2).sum()
    assert (heavy.predict(X) == 2).sum() > (y == 2).sum()
    flat = GuideClassifier(pruning="none", max_depth=1, class_priors=[0.9, 0.1]).fit(X, y)
    assert (flat.predict(X) == 2).mean() > 0.5


def test_categorical_and_missing_values():
    rng = np.random.default_rng(2)
    n = 400
    group = rng.choice(["a", "b", "c", "d"], size=n)
    x = rng.normal(size=n)
    y = np.where(np.isin(group, ["a", "c"]), 5.0, 0.0) + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"group": group, "x": x})
    X.loc[::10, "x"] = np.nan
    X.loc[::17, "group"] = None
    model = GuideRegressor(pruning="none", max_depth=2).fit(X, y)
    assert "group" in model.to_text()
    new = pd.DataFrame({"group": ["a", "b", "z", None], "x": [0.0, np.nan, 1.0, 2.0]})
    pred = model.predict(new)
    assert pred[0] > 4 and pred[1] < 1
    assert np.isfinite(pred).all()


def test_categorical_features_for_arrays():
    X = np.array([["a", 1.0], ["b", 2.0], ["a", 3.0], ["b", 4.0]] * 25, dtype=object)
    y = np.array([1.0, 0.0, 1.0, 0.0] * 25)
    model = GuideRegressor(pruning="none", max_depth=1, categorical_features=[0]).fit(X, y)
    np.testing.assert_allclose(model.predict(X), y)
    with pytest.raises(ValueError, match="could not convert"):
        GuideRegressor(pruning="none").fit(X, y)


def test_poisson_exposure():
    rng = np.random.default_rng(3)
    n = 500
    X = rng.uniform(size=(n, 1))
    exposure = rng.uniform(1, 5, size=n)
    y = rng.poisson(exposure * np.where(X[:, 0] < 0.5, 1.0, 4.0))
    model = GuideRegressor(loss="poisson", pruning="none", max_depth=1).fit(
        X, y, exposure=exposure
    )
    rate = model.predict([[0.2], [0.8]], exposure=[1.0, 1.0])
    np.testing.assert_allclose(rate, [1.0, 4.0], rtol=0.2)
    with pytest.raises(ValueError, match="exposure"):
        model.predict([[0.2]])


def test_linear_leaves():
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = np.where(X[:, 1] < 0, 2 * X[:, 0], -3 * X[:, 0]) + rng.normal(scale=0.01, size=300)
    model = GuideRegressor(node_model="linear", pruning="none", max_depth=1).fit(X, y)
    assert model.get_n_leaves() == 2
    assert np.abs(model.predict(X) - y).mean() < 0.05


def test_pickle_clone_and_json_round_trip(iris):
    X, y = iris
    model = GuideClassifier().fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
    np.testing.assert_array_equal(restored.apply(X), model.apply(X))
    assert Model.from_json(model.to_json()).to_json() == model.to_json()
    assert clone(model).get_params() == model.get_params()
    assert "digraph" in model.to_dot()


def test_works_with_model_selection(iris):
    X, y = iris
    search = GridSearchCV(GuideClassifier(), {"max_depth": [1, 3]}, cv=3).fit(X, y)
    assert search.best_params_["max_depth"] == 3
    pipe = make_pipeline(FunctionTransformer(), GuideRegressor(pruning="none", max_depth=2))
    assert cross_val_score(pipe, X.iloc[:, 1:], X.iloc[:, 0], cv=KFold(3, shuffle=True, random_state=0)).mean() > 0.5


def test_invalid_options_are_reported():
    X, y = np.arange(20.0).reshape(-1, 1), np.arange(20.0)
    with pytest.raises(ValueError, match="pruning must be one of"):
        GuideRegressor(pruning="bogus").fit(X, y)
    with pytest.raises(ValueError, match="significance_level"):
        GuideRegressor(significance_level=2.0).fit(X, y)
    with pytest.raises(ValueError, match="requires max_leaves"):
        GuideRegressor(pruning="max_leaves").fit(X, y)
