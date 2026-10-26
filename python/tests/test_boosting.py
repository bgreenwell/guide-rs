import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_friedman1

from guide import GuideBoostingClassifier, GuideBoostingRegressor, plot_tree
from guide._core import Model


def test_boosting_regressor_learns_and_stops_early():
    X, y = make_friedman1(n_samples=600, noise=0.5, random_state=0)
    train, valid = slice(0, 400), slice(400, 600)
    model = GuideBoostingRegressor(n_estimators=400, early_stopping_rounds=15).fit(
        X[train], y[train], validation_data=(X[valid], y[valid]))
    assert 1 <= model.n_estimators_ < 400
    assert model.score(X[valid], y[valid]) > 0.85
    assert model.apply(X[:5]).shape == (5, model.n_estimators_)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X[valid]), model.predict(X[valid]))
    with pytest.raises(ValueError, match="used together"):
        GuideBoostingRegressor(early_stopping_rounds=5).fit(X, y)
    with pytest.raises(ValueError, match="loss must be"):
        GuideBoostingRegressor(loss="quantile").fit(X, y)


def test_boosting_matches_the_core_fit():
    X, y = make_friedman1(n_samples=200, random_state=1)
    model = GuideBoostingRegressor(n_estimators=20, subsample=0.8, max_features=0.6).fit(X, y)
    saved = json.loads(model.to_json())
    assert saved["options"]["mtry"] == 6
    core = Model.fit(json.dumps(saved["schema"]), [X[:, j].tolist() for j in range(10)],
                     y.tolist(), json.dumps(saved["options"]),
                     boosting=json.dumps(saved["boosting"]["options"]))
    assert json.loads(core.to_json()) == saved


def test_boosting_classifier_and_poisson():
    X, y = load_iris(return_X_y=True, as_frame=True)
    labels = y.map(dict(enumerate(["setosa", "versicolor", "virginica"])))
    clf = GuideBoostingClassifier(n_estimators=30).fit(X, labels)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (clf.predict(X) == labels).mean() > 0.95
    assert len(json.loads(clf.to_json())["trees"]) == 90
    rng = np.random.default_rng(2)
    Xp = pd.DataFrame({"a": rng.uniform(0, 1, 500)})
    exposure = rng.uniform(0.5, 2, 500)
    counts = rng.poisson(exposure * np.where(Xp["a"] > 0.5, 4.0, 1.0))
    poisson = GuideBoostingRegressor(loss="poisson").fit(Xp, counts, exposure=exposure)
    rate = poisson.predict(pd.DataFrame({"a": [0.25, 0.75]}), exposure=[1.0, 1.0])
    np.testing.assert_allclose(rate, [1.0, 4.0], rtol=0.15)
    plot_tree(poisson, tree=3)
