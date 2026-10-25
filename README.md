# guide-rs

A native Rust implementation of statistical trees, with **GUIDE 46.3 as the feature and behavior target**. It separates association-based variable selection from split optimization and runs without GUIDE, Fortran, BLAS, or LAPACK.

**Status: experimental and incomplete.** The core implementation works, but this is not yet a full or numerically validated replacement for GUIDE. [Coverage and remaining work](docs/coverage.md) distinguish implemented capabilities, deliberate differences, and missing methods. In particular, survival, longitudinal, treatment-effect, and several advanced node models are not implemented. Do not use this release to reproduce published GUIDE analyses without independent validation.

## Build and use

```sh
cargo build --release
cargo run -- config
cargo run --release -- fit --data tests/fixtures/datasets/iris.csv --config examples/iris.json --model /tmp/iris-model.json
cargo run --release -- predict --model /tmp/iris-model.json --data tests/fixtures/datasets/iris.csv
cargo run --release -- inspect --model /tmp/iris-model.json
cargo run --release -- export --model /tmp/iris-model.json --format dot --output /tmp/iris-tree.dot
```

Use `--help` for all commands. Files are created without overwriting existing files; `--force` permits replacement. Graphviz DOT and LaTeX (`forest` package) exports describe the fitted trees; their renderers are optional external tools.

A minimal regression configuration is:

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

CSV types are inferred from all nonmissing entries unless `data.features` is specified. Names, category labels, missing-value tokens, model parameters, and prediction routing are preserved. Response, weight, and offset columns are never inferred as predictors. CSV prediction columns are matched by name, so their order may differ from training. When specifying features, their format is:

```json
{"name":"age", "feature_type":{"kind":"numeric"}, "role":"both"}
```

Feature types are `numeric`, `categorical`, or `periodic` (with `period`). Roles are `split`, `fit`, `both`, or `excluded`. Linear and polynomial node models require explicit `fit`/`both` roles. Newly fitted categorical splits send unknown levels right; missing values use a separately stored route. Defaults and numerical conventions are documented in [the algorithm specification](docs/algorithms.md). Numerical threshold cuts use observed split-value ranks before missing routing: at least the configured minimum on the left and strictly more on the right, matching the native endpoint convention.

Multiple-linear squared-error fits with numeric fitting columns now use the verified sample-size fallback policy. Set `fit.prediction_truncation` to `none`, `node_range`, `node_range_expanded` (10% padding), or `global_range`; see [the example configuration](examples/linear-limits.json). Rust defaults to `none` for compatibility; GUIDE's native default is `global_range`. Global bounds use the full active training sample across CV folds; node bounds are fold-local. [The linear validation record](docs/linear-validation.md) documents thresholds, exclusions, saved-model compatibility and a native least-squares correctness difference.

Stepwise leaves are an experimental, opt-in compatibility option with known native selection/fallback differences. Exact stepwise parity work is deferred while multiple-linear split selection and pruning are validated; see [the recorded decision](docs/linear-validation.md#stepwise-decision-2026-10-23).

For Poisson regression, `data.offset` names a **positive exposure**, not its logarithm. Models fitted with exposures require them again at prediction time. Classification treats positive observation weights as 1, as GUIDE does; priors and misclassification costs are separate options. Cost matrices are indexed `[true_class][predicted_class]`, in sorted class-label order.

Quantile regression supports constant, linear and simple polynomial node models. Set `fit.task.loss` to `{"kind":"quantile","probability":0.5}` for conditional medians, and `fit.node_model` to `{"kind":"linear"}` for linear nodes. [The median-regression example](examples/mtcars-quantile.json) uses explicit fitting roles. The pure Rust optimizer records convergence status; check the model's report warnings. Simultaneous lower/upper quantiles currently support constant nodes only.

## Rust API

```rust
use guide_rs::{CsvOptions, Dataset, FitOptions, FittedModel, Pruning, Task, dataset_from_csv};

let options = FitOptions { pruning: Pruning::None, ..Default::default() };
let data: Dataset = dataset_from_csv(
    "x,target\n1,0\n2,0\n3,1\n4,1\n",
    &CsvOptions::default(), &Task::default(),
)?;
let model = FittedModel::fit(&data, &options)?;
let predictions = model.predict(&data.rows)?;
let restored = FittedModel::from_json(&model.to_json()?)?;
assert_eq!(predictions, restored.predict(&data.rows)?);
# Ok::<(), guide_rs::Error>(())
```

`FittedModel::fit_with_validation` accepts a separate pruning dataset; `fit_ensemble` fits bagging/forest models. `importance` returns raw, null-mean, bias-adjusted, and optional normalized importance scores. All stochastic operations use a recorded seed and ChaCha8; they do not reproduce Fortran random streams.

## Verification

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --test interoperability -- --ignored  # requires Rscript, quantreg, rpart
cargo test --release --test statistical -- --ignored --nocapture
python3 tools/inspect_reference.py               # portable static Mach-O inspection
python3 -m unittest discover -s tools -p 'test_reference_*.py'
```

Tests cover numerical reference values, interaction-only signals, missing values, categorical predictors, weighted models, pruning, reproducibility, serialization, CLI workflows, and the bundled Iris data. Optional R tests compare extreme chi-square log tails/inverses against R stats, fitting against `lm`/`glm`/`quantreg::rq`, and one-predictor cost-complexity pruning against `rpart`. [The validation record](docs/validation.md) includes numerical audit fixes, statistical smoke-test results and their limitations. Newly fitted models and importance reports record the numerical behavior revision in `algorithm_version`. Importance can report an `indeterminate` group when score ties prevent assigning individual memberships. A release should not claim GUIDE parity until the outstanding differential and statistical acceptance tests in the coverage matrix pass.

The manual and papers are local-only references under the ignored `assets/` directory. Pinned native executables and their download metadata remain in the ignored `.reference/upstream/` cache. The retired `guidevi` wrapper and its old executable have been removed. The small Iris and mtcars example datasets live in `tests/fixtures/datasets/`; building and testing Rust requires neither reference directory. No original Fortran source or proprietary runtime code has been copied into the Rust implementation.

Working GUIDE 46.2/46.3 Apple Silicon references supply 476 frozen captures across 238 distinct cases: 70 constant fitting/pruning captures, 16 importance captures, 30 linear/stepwise leaf captures, 34 multiple-linear tree captures, 94 fallback/truncation captures, 82 categorical-linear captures, 72 selection-boundary captures and 78 orientation captures; see [upstream reference provenance](docs/reference-upstream.md). The ordinary Rust tests compare these outputs without executing GUIDE. [Linear leaf validation](docs/linear-validation.md) records numerical fitting and explicit native CV differences. [Categorical validation](docs/categorical-linear-validation.md) covers dummy references, mixed-design fallbacks and a corrected subset search. [Selection validation](docs/selection-validation.md) resolves the single-predictor zero-score stopping gap while preserving weak positive and interaction searches; [Orientation validation](docs/orientation-validation.md) fixes complementary child orientations and unseen routing, and preserves an independently verified categorical CV discrepancy. Multi-predictor zero-score fallback remains open. The primary oracle is the pinned 46.3 build; 46.2 remains a secondary regression reference. The bundled **46.1 manual (January 24, 2026)** supplies documentation, supplemented by the published history through 46.2 and direct observations of 46.3. The coverage matrix includes the newer features and remaining default/behavior differences.

The [enhancement roadmap](docs/GUIDE-RS%20Enhancement%20Plan.md) prioritizes validation and completing GUIDE before additional extensions. R prediction-function export is excluded from the product feature list and roadmap; R is used for independent numerical validation.

Pruning checks retain correct weakest-link selection where native CV ignores a cheaper branch because of node-number ordering. The [documented counterexamples](docs/native-cv-difference.md) preserve and test the native difference explicitly.

[Association and importance checks](docs/importance-validation.md) cover extreme-tail stability and important-variable sets. Exact native importance calibration and normalization remain unverified.
