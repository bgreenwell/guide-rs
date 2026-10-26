# guide-rs for Python

scikit-learn estimators for GUIDE-style classification and regression trees, backed by the [guide-rs](https://github.com/bgreenwell/guide-rs) Rust core.

The bindings are experimental; see the core's [coverage and remaining work](https://github.com/bgreenwell/guide-rs/blob/main/docs/coverage.md) for what is validated against GUIDE 46.3.

## Features

- `GuideRegressor`, `GuideClassifier`, `GuideForestRegressor` and `GuideForestClassifier`, compatible with scikit-learn's estimator API, pipelines, cloning and model selection
- Unbiased split-variable selection with interaction tests, as in GUIDE
- Native categorical features and missing values (NaN or None); no one-hot encoding or imputation
- Constant, linear, polynomial and (experimental) stepwise leaves; squared-error, quantile, dual-quantile, Poisson and logistic losses; class priors and costs
- Per-feature roles (`feature_roles`): split only, fit only, both or excluded; periodic features (`periodic_features`) and missing-value flags (`missing_flags`)
- Pruning by cross-validation, cost complexity, leaf count or a validation sample (`fit(..., validation_data=...)`)
- Bagged ensembles and forests with out-of-bag loss, and bias-adjusted GUIDE importance scores (`guide.importance`)
- Gradient boosting (`GuideBoostingRegressor`, `GuideBoostingClassifier`) whose split variables come from GUIDE's tests on gradient signs, so split counts and gain are not biased toward variables with many values
- Tree drawings with matplotlib (`guide.plot_tree`), plus text, Graphviz DOT and JSON output, the JSON matching models saved by the guide-rs CLI

## Installation

From a clone (requires a Rust toolchain):

```sh
cd python
uv sync
```

The distribution is named `guide-rs`; the package imports as `guide`. `plot_tree` needs matplotlib (the `plot` extra).

## Quick start

```python
from sklearn.datasets import load_iris
from guide import GuideClassifier

X, y = load_iris(return_X_y=True, as_frame=True)
model = GuideClassifier().fit(X, y)
print(model.predict_proba(X.head()))

from guide import plot_tree
plot_tree(model)
```

DataFrame columns with object, string, categorical or boolean dtypes are treated as categorical; for arrays, name them with `categorical_features`.

Importance scores refit shallow trees with the estimator's configuration:

```python
from guide import importance

report = importance(GuideClassifier(), X, y)
for score in report["scores"]:
    print(score["feature"], score["adjusted_score"], score["group"])
```

Cross-validation pruning, forests and importance permutations run in parallel, with the same results for any thread count. `n_jobs` sets the threads; unlike most scikit-learn estimators, its default (None) uses all cores. Lower it when `GridSearchCV` also runs jobs in parallel.

## Documentation

- Example notebooks comparing GUIDE with scikit-learn's CART trees on its bundled datasets: [classification](examples/classification.ipynb), [regression](examples/regression.ipynb), and [selection bias and importance](examples/selection_bias.ipynb)
- [User guide and API reference](https://bgreenwell.github.io/guide-rs/), built from `docs/` and published on each push to `main`
- [Usage and configuration of the core](https://github.com/bgreenwell/guide-rs/blob/main/docs/usage.md)

## Development

```sh
uv sync                                  # build the extension in editable mode
uv run ruff check . && uv run pytest     # quality gate
uv run --group examples python examples/execute.py   # re-run the notebooks
uv run --group docs sphinx-build -W -b html docs docs/_build/html   # build the docs
```

Branch and commit conventions are in the repository's [AGENTS.md](https://github.com/bgreenwell/guide-rs/blob/main/AGENTS.md).
