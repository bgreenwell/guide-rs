"""Estimators pickled by published releases must keep their predictions.

The fixtures come from tools/freeze_saved_models.py, run under each release.
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parents[2]
RELEASES = sorted((ROOT / "tests/fixtures/saved-models").iterdir())
CASES = [
    (release, name, family)
    for release in RELEASES
    for name, family in json.loads((release / "manifest.json").read_text())["families"].items()
]


def test_every_family_of_the_first_release_is_frozen():
    assert sum(release.name == "0.2.0" for release, _, _ in CASES) == 19


@pytest.mark.parametrize(
    "release, name, family", CASES, ids=[f"{r.name}-{name}" for r, name, _ in CASES]
)
def test_pickled_estimators_keep_their_predictions(release, name, family):
    with warnings.catch_warnings():
        # scikit-learn warns when a pickle comes from another of its versions.
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        model = pickle.loads((release / "pickles" / f"{name}.pkl").read_bytes())
    assert type(model).__name__ == family["estimator"]
    X = np.array(family["rows"], dtype=object)
    expected = family["predictions"]
    if expected[0]["kind"] == "classification":
        proba = model.predict_proba(X)
        assert list(model.predict(X)) == [p["label"] for p in expected]
        wanted = [[dict(p["probabilities"])[str(c)] for c in model.classes_] for p in expected]
        np.testing.assert_allclose(proba, wanted, rtol=1e-12, atol=1e-12)
    elif expected[0]["kind"] == "quantiles":
        wanted = [[p["lower"], p["upper"]] for p in expected]
        np.testing.assert_allclose(model.predict(X), wanted, rtol=1e-12, atol=1e-12)
    else:
        predicted = model.predict(X, exposure=family["exposure"])
        wanted = [p["value"] for p in expected]
        np.testing.assert_allclose(predicted, wanted, rtol=1e-12, atol=1e-12)
