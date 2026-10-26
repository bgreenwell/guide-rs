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

Like GUIDE, classifiers choose the tree size by held-out misclassification cost. `pruning_loss="log_loss"` or `"brier"` uses a proper scoring rule instead, which judges the predicted probabilities rather than only the predicted class; forests then report that loss as `oob_loss_`.

Cross-validation folds, forest trees and importance permutations run in parallel. `n_jobs` sets the number of threads; unlike most scikit-learn estimators, its default of None uses all cores. Results are the same for any thread count.

## Boosting

`GuideBoostingRegressor` and `GuideBoostingClassifier` fit gradient-boosted trees. Boosting is experimental: a guide-rs extension, not part of GUIDE, whose variable selection and defaults may change between releases. Each split variable comes from chi-square tests of the gradients against each predictor's bins (GUIDE's tests on the gradient signs for squared error), and the split point from the second-order gain, so variables with many distinct values are not favored. Early stopping keeps the round with the lowest loss on a validation set:

```python
from guide import GuideBoostingClassifier

model = GuideBoostingClassifier(n_estimators=500, early_stopping_rounds=20)
model.fit(X_train, y_train, validation_data=(X_valid, y_valid))
model.n_estimators_  # rounds kept
```

## Importance scores

{func}`guide.importance` computes GUIDE's bias-adjusted variable importance from shallow unpruned trees and permutations of the response:

```python
from guide import importance

report = importance(GuideClassifier(), X, y)
for score in report["scores"]:
    print(score["feature"], score["adjusted_score"], score["group"])
```
