# Getting started

## Installation

From a clone of the repository (requires a Rust toolchain and [uv](https://docs.astral.sh/uv/)):

```sh
cd python
uv sync
```

The distribution is named `guide-rs`; the package imports as `guide`. `plot_tree` needs matplotlib, available as the `plot` extra.

## Fitting a tree

```python
from sklearn.datasets import load_iris
from guide import GuideClassifier, plot_tree

X, y = load_iris(return_X_y=True, as_frame=True)
model = GuideClassifier().fit(X, y)
model.predict_proba(X.head())
plot_tree(model)
```

The estimators follow the scikit-learn API, so they work in pipelines, `clone`, grid search and cross-validation.

## Inputs

- **Categorical features** are used natively. DataFrame columns with object, string, categorical or boolean dtypes are categorical; for arrays, name them with `categorical_features`.
- **Missing values** (NaN or None) are allowed in features. Splits send them left or right, and `missing_flags` can record why a value is missing.
- **Feature roles** (`feature_roles`) make a column split only, fit only (a regressor in linear leaves), both or excluded. `periodic_features` declares cyclic columns such as hour of day.

## Pruning and parallelism

Single trees are pruned by 10-fold cross-validation by default. `pruning` also accepts cost complexity, a fixed leaf count, a validation sample passed to `fit` as `validation_data`, or none.

Cross-validation folds, forest trees and importance permutations run in parallel. `n_jobs` sets the number of threads; unlike most scikit-learn estimators, its default of None uses all cores. Results are the same for any thread count.

## Importance scores

{func}`guide.importance` computes GUIDE's bias-adjusted variable importance from shallow unpruned trees and permutations of the response:

```python
from guide import importance

report = importance(GuideClassifier(), X, y)
for score in report["scores"]:
    print(score["feature"], score["adjusted_score"], score["group"])
```
