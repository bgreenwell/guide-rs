#!/usr/bin/env python3
"""Freeze models saved by a released guide-rs wheel as compatibility fixtures.

Run this under the released package, not a development build, so the fixtures
are what users of that release have on disk:

    uv venv /tmp/frozen && uv pip install --python /tmp/frozen guide-rs==X.Y.Z
    /tmp/frozen/bin/python tools/freeze_saved_models.py

It writes tests/fixtures/saved-models/X.Y.Z/ and refuses to replace an existing
release. tests/saved_models.rs and python/tests/test_saved_models.py then require
each model to keep its predictions, through JSON and pickle respectively.
"""
import hashlib
import json
import math
import pickle
from importlib.metadata import version
from pathlib import Path

import guide
import numpy as np
from guide import (
    GuideBoostingClassifier,
    GuideBoostingRegressor,
    GuideClassifier,
    GuideForestClassifier,
    GuideForestRegressor,
    GuideRegressor,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/saved-models"
SEED = 20261102
TRAINING_ROWS = 160
PREDICTION_ROWS = 30


def make_data():
    """Features and responses covering numeric, missing, periodic and categorical input."""
    rng = np.random.RandomState(SEED)
    n = TRAINING_ROWS
    x0 = rng.normal(size=n)
    x1 = rng.uniform(0, 10, size=n)
    hour = rng.uniform(0, 24, size=n)
    group = rng.choice(["a", "b", "c"], size=n).astype(object)
    score = 2.0 * (x0 > 0) + 0.5 * x1 + 3.0 * (group == "b")
    x1[::11] = np.nan
    group[::17] = None
    reason = np.where(np.isnan(x1), rng.choice(["refused", "unknown"], size=n), "observed")
    exposure = rng.uniform(0.5, 2.0, size=n)
    numeric = np.column_stack([x0, x1])
    return {
        "numeric": numeric,
        "periodic": np.column_stack([numeric, hour]),
        "mixed": np.column_stack([x0, x1, group.astype(object)]),
        "flagged": np.column_stack([x0, x1, reason.astype(object)]),
        "response": score + rng.normal(scale=0.5, size=n),
        "counts": rng.poisson(np.exp(0.2 + 0.1 * score) * exposure).astype(float),
        "binary": (score + rng.normal(size=n) > np.median(score)).astype(float),
        "classes": np.array(["low", "mid", "high"], dtype=object)[
            np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        ],
        "exposure": exposure,
    }


def families(data):
    """Family name, estimator, feature set, response and whether exposure applies."""
    tree = {"pruning": "none", "max_depth": 3, "min_samples_leaf": 5}
    return {
        "regression-constant": (GuideRegressor(**tree, categorical_features=[2]), "mixed", "response"),
        "regression-linear": (GuideRegressor(**tree, node_model="linear"), "numeric", "response"),
        "regression-polynomial": (
            GuideRegressor(**tree, node_model="polynomial"), "numeric", "response"),
        "regression-stepwise": (GuideRegressor(**tree, node_model="stepwise"), "numeric", "response"),
        "regression-periodic": (
            GuideRegressor(**tree, periodic_features={2: 24}), "periodic", "response"),
        "quantile-constant": (
            GuideRegressor(**tree, loss="quantile", quantile=0.9), "numeric", "response"),
        "quantile-linear": (
            GuideRegressor(**tree, loss="quantile", quantile=0.9, node_model="linear"),
            "numeric", "response"),
        "quantile-dual": (
            GuideRegressor(**tree, loss="dual_quantile", quantiles=(0.1, 0.9)), "numeric", "response"),
        "poisson-constant": (GuideRegressor(**tree, loss="poisson"), "numeric", "counts"),
        "poisson-exposure": (GuideRegressor(**tree, loss="poisson"), "numeric", "counts", True),
        "logistic-constant": (GuideRegressor(**tree, loss="logistic"), "numeric", "binary"),
        "classification-constant": (
            GuideClassifier(**tree, categorical_features=[2]), "mixed", "classes"),
        "classification-lda": (GuideClassifier(**tree, node_model="lda"), "numeric", "classes"),
        "classification-lda-splits": (
            GuideClassifier(**tree, node_model="lda_splits"), "numeric", "classes"),
        "classification-flags": (GuideClassifier(**tree, missing_flags={2: 1}), "flagged", "classes"),
        "forest-regressor": (
            GuideForestRegressor(n_estimators=5, max_depth=3, categorical_features=[2]),
            "mixed", "response"),
        "forest-classifier": (
            GuideForestClassifier(n_estimators=5, max_depth=3, categorical_features=[2]),
            "mixed", "classes"),
        "boosting-regressor": (
            GuideBoostingRegressor(n_estimators=5, max_depth=2), "numeric", "response"),
        "boosting-classifier": (
            GuideBoostingClassifier(n_estimators=5, max_depth=2), "numeric", "classes"),
    }


def core_rows(columns):
    """Columns as rows in the core's JSON format, with null for a missing value."""
    def value(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    return [[value(x) for x in row] for row in zip(*columns)]


def predictions(model, columns, exposure):
    """Predictions in the core's JSON format."""
    if model._model.classes:
        labels, probabilities = model._model.predict_classes(columns)
        return [
            {"kind": "classification", "label": label,
             "probabilities": [list(p) for p in zip(model._model.classes, row)]}
            for label, row in zip(labels, probabilities)
        ]
    if getattr(model, "loss", None) == "dual_quantile":
        return [{"kind": "quantiles", "lower": lower, "upper": upper}
                for lower, upper in model._model.predict_intervals(columns)]
    return [{"kind": "regression", "value": value}
            for value in model._model.predict_values(columns, exposure)]


def main():
    out = FIXTURES / version("guide-rs")
    if out.exists():
        raise SystemExit(f"{out} exists; frozen fixtures are never replaced")
    (out / "models").mkdir(parents=True)
    (out / "pickles").mkdir()
    data = make_data()
    manifest = {
        "release": out.name,
        "algorithm_version": guide.ALGORITHM_VERSION,
        "generator": "tools/freeze_saved_models.py",
        "seed": SEED,
        "families": {},
    }
    for name, spec in families(data).items():
        estimator, features, response, *flags = spec
        uses_exposure = bool(flags)
        X = data[features]
        fit_kwargs = {"exposure": data["exposure"].tolist()} if uses_exposure else {}
        model = estimator.fit(X, data[response], **fit_kwargs)
        X_new = X[:PREDICTION_ROWS]
        if features == "mixed":
            # A row with missing values and a category unseen in training.
            X_new = np.vstack([X_new, np.array([[np.nan, np.nan, "unseen"]], dtype=object)])
        columns, _ = model._columns(X_new, reset=False)
        exposure = data["exposure"][:PREDICTION_ROWS].tolist() if uses_exposure else None
        model_path = out / "models" / f"{name}.json"
        model_path.write_text(model.to_json() + "\n")
        pickle_path = out / "pickles" / f"{name}.pkl"
        pickle_path.write_bytes(pickle.dumps(model, protocol=4))
        manifest["families"][name] = {
            "estimator": type(model).__name__,
            "rows": core_rows(columns),
            "exposure": exposure,
            "predictions": predictions(model, columns, exposure),
            "sha256": {
                path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (model_path, pickle_path)
            },
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(manifest['families'])} families to {out}")


if __name__ == "__main__":
    main()
