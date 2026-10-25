import json

import numpy as np
import pandas as pd
import pytest

from guide_rs import GuideClassifier, GuideForestRegressor, GuideRegressor
from guide_rs._core import Model


def core_fit(schema, columns, target, options, **kwargs):
    return json.loads(Model.fit(json.dumps(schema), columns, target, json.dumps(options),
                                **kwargs).to_json())


def test_periodic_features_wrap_around():
    rng = np.random.default_rng(0)
    hour = rng.uniform(0, 24, 400)
    # High at night: hours after 22 or before 4, an interval across midnight.
    y = np.where((hour > 22) | (hour < 4), 5.0, 0.0) + rng.normal(0, 0.1, 400)
    X = pd.DataFrame({"hour": hour})
    model = GuideRegressor(periodic_features={"hour": 24}, pruning="none", max_depth=1).fit(X, y)
    saved = json.loads(model.to_json())
    assert saved["schema"][0]["feature_type"] == {"kind": "periodic", "period": 24.0}
    assert saved["schema"][0]["role"] == "split"
    assert saved["trees"][0]["root"]["split"]["kind"] == "periodic"
    np.testing.assert_allclose(model.predict(pd.DataFrame({"hour": [23.0, 1.0, 12.0]})),
                               [5.0, 5.0, 0.0], atol=0.1)
    core = core_fit(saved["schema"], [hour.tolist()], y.tolist(), saved["options"])
    assert core["trees"] == saved["trees"]
    with pytest.raises(ValueError, match="positive and finite"):
        GuideRegressor(periodic_features={"hour": 0}).fit(X, y)
    with pytest.raises(ValueError, match="must be numeric"):
        GuideRegressor(periodic_features={"g": 7}).fit(
            pd.DataFrame({"g": ["a", "b"] * 5}), np.arange(10.0))


def test_missing_flags_route_codes():
    rng = np.random.default_rng(1)
    n = 300
    income = rng.uniform(0, 100, n)
    reason = np.array([None] * n, dtype=object)
    missing = rng.random(n) < 0.3
    income[missing] = np.nan
    reason[missing] = rng.choice(["refused", "unknown"], missing.sum())
    y = np.where(reason == "refused", 1, 0).astype(str)
    X = pd.DataFrame({"income": income, "income_reason": reason})
    model = GuideClassifier(missing_flags={"income_reason": "income"}, pruning="none").fit(X, y)
    saved = json.loads(model.to_json())
    assert saved["schema"][1] == {"name": "income_reason",
                                  "feature_type": {"kind": "categorical"},
                                  "role": "flag", "flags": "income"}
    assert (model.predict(X) == y).mean() > 0.99
    # Arrays work too; flag columns are categorical even without categorical_features.
    array_model = GuideClassifier(missing_flags={1: 0}, pruning="none").fit(X.to_numpy(), y)
    assert json.loads(array_model.to_json())["schema"][1]["role"] == "flag"
    with pytest.raises(ValueError, match="cannot also have a feature role"):
        GuideClassifier(missing_flags={"income_reason": "income"},
                        feature_roles={"income_reason": "split"}).fit(X, y)
    with pytest.raises(ValueError, match="cannot flag itself"):
        GuideClassifier(missing_flags={"income": "income"}).fit(X, y)


def test_logistic_and_dual_quantile_losses():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 600)
    X = pd.DataFrame({"x": x})
    p = np.where(x > 0, 0.8, 0.2)
    y = (rng.random(600) < p).astype(float)
    logistic = GuideRegressor(loss="logistic", pruning="none", max_depth=1).fit(X, y)
    np.testing.assert_allclose(logistic.predict(pd.DataFrame({"x": [-0.5, 0.5]})),
                               [0.2, 0.8], atol=0.06)
    spread = np.where(x > 0, 3.0, 0.5)
    z = rng.normal(0, 1, 600) * spread
    dual = GuideRegressor(loss="dual_quantile", quantiles=(0.1, 0.9), pruning="none",
                          max_depth=1).fit(X, z)
    bounds = dual.predict(X)
    assert bounds.shape == (600, 2) and (bounds[:, 0] < bounds[:, 1]).all()
    coverage = ((z >= bounds[:, 0]) & (z <= bounds[:, 1])).mean()
    assert 0.75 < coverage < 0.85
    width = bounds[:, 1] - bounds[:, 0]
    assert width[x > 0].mean() > 3 * width[x <= 0].mean()
    with pytest.raises(ValueError, match="loss must be"):
        GuideRegressor(loss="hinge").fit(X, z)


def test_validation_pruning_matches_the_core():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 500), "b": rng.uniform(0, 1, 500)})
    y = np.where(X["a"] > 0.5, 2.0, 0.0) + rng.normal(0, 1, 500)
    train, valid = slice(0, 300), slice(300, 500)
    model = GuideRegressor(pruning="validation").fit(
        X[train], y[train], validation_data=(X[valid], y[valid]))
    saved = json.loads(model.to_json())
    columns = lambda frame: [frame[c].tolist() for c in frame.columns]
    core = core_fit(saved["schema"], columns(X[train]), y[train].tolist(), saved["options"],
                    validation=(columns(X[valid]), y[valid].tolist(), None, None))
    assert core == saved
    assert saved["report"]["pruning_scores"] and model.get_n_leaves() >= 2
    weighted = GuideRegressor(pruning="validation").fit(
        X[train], y[train], validation_data=(X[valid], y[valid], np.ones(200)))
    assert weighted.to_json() == model.to_json()
    labels = np.where(y > 1, "hi", "lo")
    classifier = GuideClassifier(pruning="validation").fit(
        X[train], labels[train], validation_data=(X[valid], labels[valid]))
    assert classifier.score(X[valid], labels[valid]) > 0.6
    with pytest.raises(ValueError, match="requires validation_data"):
        GuideRegressor(pruning="validation").fit(X, y)
    with pytest.raises(ValueError, match="requires pruning='validation'"):
        GuideRegressor().fit(X, y, validation_data=(X, y))
    with pytest.raises(ValueError, match="ensemble validation pruning"):
        GuideForestRegressor(kind="bagging", pruning="validation").fit(
            X, y, validation_data=(X, y))


def test_prediction_truncation_bounds_linear_leaves():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 200)
    y = 10 * x + rng.normal(0, 0.1, 200)
    X = pd.DataFrame({"x": x})
    far = pd.DataFrame({"x": [5.0]})
    free = GuideRegressor(node_model="linear", pruning="none", max_depth=1).fit(X, y)
    bounded = GuideRegressor(node_model="linear", pruning="none", max_depth=1,
                             prediction_truncation="global_range").fit(X, y)
    assert free.predict(far)[0] > 40
    assert bounded.predict(far)[0] == pytest.approx(y.max())
    with pytest.raises(ValueError, match="prediction_truncation|unknown variant"):
        GuideRegressor(node_model="linear", prediction_truncation="clip").fit(X, y)
