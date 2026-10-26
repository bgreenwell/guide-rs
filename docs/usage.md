# Using guide-rs

This page covers configuration details for the CLI and Rust API. Defaults and numerical conventions are specified in [the algorithm specification](algorithms.md); run `guide-rs config` to print a complete default configuration.

## Run configuration

A run configuration has `data` (CSV options), `fit` (model options), and optional `ensemble` and `importance` sections. A minimal regression configuration:

```json
{
  "data": { "target": "mpg" },
  "fit": {
    "task": { "kind": "regression", "loss": { "kind": "squared_error" } },
    "pruning": { "kind": "cross_validation", "folds": 5, "standard_errors": 0.25 },
    "seed": 42
  }
}
```

Output files are never overwritten unless `--force` is given. Each subcommand documents its flags under `--help`.

## CSV input and feature types

CSV types are inferred from all nonmissing entries unless `data.features` is specified. With inference, every non-response column becomes a split feature, so list features explicitly when the file contains identifiers or other columns that should not be used. Response, weight and offset columns are never inferred as predictors. Blank lines are skipped.

An explicit feature looks like:

```json
{"name": "age", "feature_type": {"kind": "numeric"}, "role": "both"}
```

- Feature types: `numeric`, `categorical`, or `periodic` (with `period`).
- Roles: `split`, `fit`, `both`, or `excluded`. Linear and polynomial node models require explicit `fit`/`both` roles.

Names, category labels, missing-value tokens, model parameters and prediction routing are saved with the model. Prediction CSV columns are matched by name, so their order may differ from training. Newly fitted categorical splits send unknown levels right; missing values use a separately stored route. Numerical threshold cuts use observed split-value ranks before missing routing: at least the configured minimum on the left and strictly more on the right, matching the native endpoint convention.

## Missing-value flags

A flag column records why values of another feature are missing, like GUIDE's M variables. Declare it as a categorical feature with role `flag` and name its parent in `flags`:

```json
{"name": "income", "feature_type": {"kind": "numeric"}, "role": "split"},
{"name": "income_reason", "feature_type": {"kind": "categorical"}, "role": "flag", "flags": "income"}
```

Codes only matter where the parent is missing. Each code is a separate missing group for variable selection, and splits can send any subset of codes left, for example `income <= 50000; missing -> yes if income_reason in ["refused"]`. At prediction time, missing values with no code or an unseen code go right. Flags require constant node models and numeric or categorical parents, with at most 10 codes. See [the flag validation record](missing-flag-validation.md).

## Linear node models

Multiple-linear squared-error fits with numeric fitting columns use the verified sample-size fallback policy. Set `fit.prediction_truncation` to `none`, `node_range`, `node_range_expanded` (10% padding) or `global_range`; see [the example configuration](../examples/linear-limits.json). guide-rs defaults to `none` for compatibility; GUIDE's native default is `global_range`. Global bounds use the full active training sample across CV folds; node bounds are fold-local. [The linear validation record](linear-validation.md) documents thresholds, exclusions, saved-model compatibility and a native least-squares correctness difference.

Stepwise leaves are an experimental, opt-in compatibility option with known native selection and fallback differences; see [the recorded decision](linear-validation.md#stepwise-decision-2026-10-23).

## Poisson, classification and quantile options

- **Poisson:** `data.offset` names a positive exposure, not its logarithm. Models fitted with exposures require them again at prediction time.
- **Classification:** positive observation weights are treated as 1, as GUIDE does; priors and misclassification costs are separate options. Cost matrices are indexed `[true_class][predicted_class]`, in sorted class-label order.
- **Pruning loss:** cross-validation and validation pruning of classification trees choose the subtree on the cost-complexity sequence by held-out misclassification cost, as GUIDE does. `fit.pruning_loss` can instead be `"log_loss"` or `"brier"` (summed over classes), proper scoring rules that judge the predicted probabilities; forests then report the same loss out of bag. These are guide-rs extensions and cannot be combined with class costs. Held-out probabilities include GUIDE's zero-posterior correction, so log loss stays finite.
- **Quantile regression:** supports constant, linear and simple polynomial node models. Set `fit.task.loss` to `{"kind": "quantile", "probability": 0.5}` for conditional medians and `fit.node_model` to `{"kind": "linear"}` for linear nodes; see [the median-regression example](../examples/mtcars-quantile.json). The optimizer records convergence status in the model's report warnings. Simultaneous lower/upper quantiles support constant nodes only.

## Ensembles and importance

`ensemble` fits bagged trees or forests with per-node feature sampling and out-of-bag loss. Forests require `pruning: none`; bagged trees may use `none`, `cost_complexity` or `fixed_leaves`, but not cross-validation, because bootstrap samples repeat rows.

`importance` returns raw, null-mean, bias-adjusted and optional normalized scores. It can report an `indeterminate` group when score ties prevent assigning individual memberships.

All stochastic operations use the recorded seed with ChaCha8; they do not reproduce Fortran random streams.

Ensemble trees, cross-validation folds and importance permutations are fitted in parallel on all cores. Results do not depend on the number of threads; `--threads` (or `RAYON_NUM_THREADS`) limits it.

## Exports

`guide-rs export --format text|dot|latex` writes a text summary, a Graphviz DOT graph, or a LaTeX document using the `forest` package. Rendering DOT or LaTeX requires the corresponding external tools.
