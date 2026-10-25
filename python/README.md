# guide-rs for Python

scikit-learn estimators for GUIDE-style classification and regression trees, backed by the [guide-rs](https://github.com/bgreenwell/guide-rs) Rust core.

The bindings are experimental and cover single trees; see the core's [coverage and remaining work](https://github.com/bgreenwell/guide-rs/blob/main/docs/coverage.md) for what is validated against GUIDE 46.3.

## Features

- `GuideRegressor` and `GuideClassifier`, compatible with scikit-learn's estimator API, pipelines, cloning and model selection
- Unbiased split-variable selection with interaction tests, as in GUIDE
- Native categorical features and missing values (NaN or None); no one-hot encoding or imputation
- Constant, linear and polynomial leaves; squared-error, quantile and Poisson losses; class priors and costs
- Text, Graphviz DOT and JSON output, the JSON matching models saved by the guide-rs CLI

## Installation

From a clone (requires a Rust toolchain):

```sh
cd python
uv sync
```

## Quick start

```python
from sklearn.datasets import load_iris
from guide_rs import GuideClassifier

X, y = load_iris(return_X_y=True, as_frame=True)
model = GuideClassifier().fit(X, y)
print(model.to_text())
print(model.predict_proba(X.head()))
```

DataFrame columns with object, string, categorical or boolean dtypes are treated as categorical; for arrays, name them with `categorical_features`.

## Documentation

- Estimator parameters: `help(GuideRegressor)`, `help(GuideClassifier)`
- [Usage and configuration of the core](https://github.com/bgreenwell/guide-rs/blob/main/docs/usage.md)

## Development

```sh
uv sync                                  # build the extension in editable mode
uv run ruff check . && uv run pytest     # quality gate
```

Branch and commit conventions are in the repository's [AGENTS.md](https://github.com/bgreenwell/guide-rs/blob/main/AGENTS.md).
