# GUIDE 46.3 coverage and remaining work

Updated **2026-10-24**, through algorithm revision **`guide-rs-0.1-experimental-reference12`**. Repository cleanup is complete; validation work resumes with the core checklist below.

**guide-rs is a working subset of GUIDE, not a full replacement.** Its strongest native validation is for constant classification/regression, multiple-linear squared-error trees, and pruning. Other implemented losses and ensembles have narrower independent checks. No complete subsystem is certified equivalent to every GUIDE option.

The primary oracle is the pinned GUIDE **46.3** build of July 11, 2026; **46.2** is a secondary regression reference. Documentation comes from the **46.1** manual and history through 46.2. The fixed acceptance corpus contains **556 native captures across 278 distinct cases**, plus separately recorded diagnostics. See [reference provenance](reference-upstream.md) and [validation results](validation.md).

A checked box below means the stated capability exists with tests. It does **not** certify full native parity. Unchecked boxes identify remaining work, including validation of existing implementations.

## Implemented today

- [x] Pure Rust library and CLI; CSV input, typed feature roles, weights/exposures, named prediction columns and JSON model round trips.
- [x] Constant classification with probabilities, priors, costs, numeric/categorical splits, residual/response association tests and interaction search.
- [x] Weighted constant and multiple-linear least-squares leaves; pivoted QR, missing-data terms, numeric/categorical fitting and verified sample guards.
- [x] Multiple-linear SSE prediction limits: none, node range, expanded node range and global range.
- [x] Bounded exhaustive categorical linear search, local child orientation, and checked missing/unseen routes.
- [x] Simple polynomial leaves; experimental, opt-in grouped stepwise leaves with known compatibility gaps.
- [x] Single-quantile constant/linear/polynomial fits; simultaneous lower/upper **constant** quantiles.
- [x] Basic Poisson and Bernoulli logistic constant/linear/polynomial fits, including positive Poisson exposures.
- [x] Periodic interval splits and optional bivariate discriminant splits.
- [x] Missing-value flag (M) variables for constant node models, with per-code selection groups and code-subset split routing.
- [x] Cost-complexity, fixed-size, validation-sample and cross-validation pruning; independently verified correct behavior in documented native counterexamples.
- [x] Basic bagging/forest ensembles, per-node feature sampling, OOB loss and deterministic seeds.
- [x] Constant classification/SSE importance scoring, permutation adjustment and scoped important-variable checks.
- [x] Prediction, JSON conversion, text/DOT/LaTeX tree output, and macOS/Linux/Windows CI configuration.
- [x] Python bindings (`python/`): scikit-learn `GuideRegressor`, `GuideClassifier`, `GuideForestRegressor` and `GuideForestClassifier` with feature roles, plus `importance`. They pass `sklearn.utils.estimator_checks`, reproduce the core model exactly on the 233 reference cases they can express, and reproduce the native important sets on the importance captures.
- [x] Standalone source-tree verification: all ordinary Rust tests pass in CI from a clean checkout without local manuals, native executables or reference caches.

## Comparison by GUIDE family

| GUIDE family / manual chapters | guide-rs status | Principal remaining gap |
|---|---|---|
| Data, operation, missing flags (3, 5) | Modern CSV/JSON interfaces, missing handling and constant-node missing-value flags | Flags with linear node models (M-to-B conversion) and periodic parents |
| Classification (4) | Constant-node trees, priors/costs, interactions, optional projected splits and high-cardinality multiclass search | Kernel, nearest-neighbor, LDA/QDA **node models** and full discriminant-coordinate options |
| Least squares (6) | Constant/multiple-linear core has the strongest differential evidence; polynomial and stepwise subsets exist | Broader selection/default combinations, native polynomial rules, ANCOVA/all-subsets and population mean estimation |
| Quantiles (7) | Constant/linear/polynomial single quantiles; dual constant quantiles | Nonconstant simultaneous quantiles and native growth/model-selection conventions |
| Periodic predictors (8) | Circular split search and modulo prediction | Association rotation invariance and native routing/default validation |
| Poisson, logistic (9, 15) | Basic GLM fitting and prediction | Native residual grouping, growth, inference and specialized options |
| Censored responses (10) | Missing; deferred until a Python wrapper exists | Survival data, proportional hazards and restricted mean event time |
| Treatments (11–12) | Missing | Randomized/observational subgroup methods, adjustment and inference |
| Multiresponse, longitudinal (13–14) | Missing | Joint responses, repeated measures, time-varying models and smoothing |
| Importance (16) | Scoped constant classification/SSE scoring | Exact calibration/normalization, 50% thresholds, flag aggregation and specialized responses |
| Propensity, DIF, bootstrap intervals (17–19) | Missing | Their specialized estimators, data structures and inference |
| Ensembles (20) | Basic bagging/forest/OOB implementation | Native defaults, random missing routing, broader validation and bootstrap/CV handling |
| Pruning and other features (21) | Four pruning interfaces and modern model/tree output | Random multifold/native test-sample acceptance, broader combinations and general feature generation |
| Least median of squares | Missing | Robust LMS fitting; ordinary median/quantile loss is a different method |
| Portability | Pure Rust; local macOS checks and CI configuration | Observed Linux/Windows CI results and complete release acceptance |

## Remaining checklist: core compatibility and correctness

These are the next priorities before expanding into the missing specialized families.

- [ ] Establish native tie-breaking among exactly tied multi-predictor scores. Zero-gain splitting (`min_gain = 0`) and same-class leaf merging are resolved; the singleton zero-score rule is resolved. Native tie order is not column order and is recorded as a known difference.
- [ ] Explain the classification-with-priors difference: on a depth-2 design with priors 0.3/0.7, native keeps three leaves where Rust keeps two.
- [ ] Validate high-cardinality categorical linear searches, near-tied partitions, broader missing patterns and projected/periodic split defaults. Multiclass categorical search above 11 levels is implemented and validated; when both sides have the same number of levels its orientation follows native eigenvector signs, which are not reproduced.
- [ ] Finish importance calibration/normalization, zero-null fallback conventions, 50% thresholds and missing-flag aggregation.
- [ ] Missing-value flags (M variables) with linear/polynomial/stepwise node models via the documented M-to-B conversion, and on periodic parents. Constant-node flags for numeric and categorical parents are implemented and validated ([flag validation](missing-flag-validation.md)).
- [ ] Validate GUIDE's polynomial degree selection, missing handling and model-specific fallback rules.
- [ ] Validate native quantile/Poisson/logistic split selection, residual grouping, model selection and fitting-policy combinations.
- [ ] Validate periodic association rotation invariance and missing/unknown routing.
- [ ] Validate random multifold CV, native test-sample pruning, broader model-based pruning and prediction-limit combinations.
- [ ] Explain the additional categorical CV-table discrepancy while preserving independently verified CV. Its output, direct native refits and independent values are already frozen and tested.
- [ ] Validate ensemble-specific defaults, fitting policies and random missing routes. CV pruning on bootstrap samples is now rejected; out-of-bag pruning remains unimplemented.
- [ ] Audit 45.x–46.3 defaults and option combinations, including multiple-linear depth, expanded classification settings, priors/costs and weight conventions.
- [ ] Expand fixed-seed smoke checks into prespecified acceptance studies. Root selection bias and weak-signal power now have a [prespecified study](selection-bias-study.md), which passes every criterion and runs in CI. Interaction power and importance familywise calibration remain.

## Remaining checklist: open audit findings

Carried over from the 2026-10-23 repository review; fixed items (log-scale selection, bagging with CV, CSV/CLI diagnostics) are omitted.

- [ ] Non-SSE linear and polynomial nodes (quantile, Poisson, logistic) accept children with only `terms + 1` rows, which can interpolate and win the split search; validate GUIDE's minimum node sizes for these models.
- [ ] Selection contingency tables count rows without weights, and class priors do not enter the tests; confirm against the native binary and document in `algorithms.md`.
- [ ] Unseen-category direction and the larger-child missing route are chosen by row count rather than weight; stepwise F tests use the unweighted n.
- [ ] Excluded numeric columns must still parse as numbers at fit and prediction time.
- [ ] Performance at scale: O(n²) CV fold construction, per-call `Context` rebuilds in external losses, refitting both child models for every split candidate (the classification interaction lookahead now sweeps class counts instead), O(cuts²) periodic intervals, and full model revalidation on every `predict` call.

## Remaining checklist: missing statistical capabilities

- [ ] Classification kernel and nearest-neighbor node models.
- [ ] Classification LDA/QDA node models and full discriminant-coordinate split options. The existing bivariate split is not an LDA/QDA node model.
- [ ] Nonconstant simultaneous lower/upper quantile models.
- [ ] Poisson/logistic statistical inference and specialized estimated-probability/treatment behavior.
- [ ] Survival: event/time validation, proportional hazards, restricted mean event time and survival predictions. Deferred by user decision (2026-10-25) until a Python wrapper is in place; native probe findings so far are in [survival notes](survival-notes.md).
- [ ] Randomized treatment subgroup methods and inference.
- [ ] Observational treatment subgroup methods, prognostic adjustment and inference.
- [ ] Joint multiresponse data, fitting and residual grouping.
- [ ] Longitudinal/repeated-measure models, time dependence and lowess/spline smoothing.
- [ ] Propensity grouping, causal estimators and missing-response imputation.
- [ ] DIF/item-response scoring and pairwise p-value matrices.
- [ ] Calibrated bootstrap post-selection confidence intervals.
- [ ] Least median of squares fitting.
- [ ] Population mean estimation.
- [ ] General power/product feature generation beyond existing polynomial leaves.

## Remaining checklist: acceptance and release

- [ ] Exercise every in-scope manual option and allowed combination under the pinned native versions.
- [ ] Extend per-case comparisons of intermediate statistics, selected variables, coefficients, split structure, pruning paths and held-out predictions.
- [x] Observe passing macOS/Linux/Windows CI runs (first observed 2026-10-25, PR #1).
- [ ] Complete release/package review and verify a committed clean checkout before claiming a GUIDE replacement.

## Deferred and deliberately excluded

- **Deferred:** further exact stepwise selection/fallback parity, all-subsets and stepwise ANCOVA work. Stepwise remains experimental and opt-in by user decision; its existing regression evidence remains supported.
- **Excluded:** R prediction-function export (removed from the CLI and API), legacy interactive menus/DSC/batch compatibility and byte-for-byte native output. Frozen upstream R output can remain evidence; independent R numerical validation remains in scope.
- **Outside the current phase:** R bindings and extra enhancement-roadmap features. Python bindings do not yet expose missing-value flags, periodic features, prediction truncation, stepwise leaves, dual quantiles, logistic loss or validation-sample pruning. Because `fit` rejects missing responses, prediction-only rows cannot contribute to the dummy coding of categorical fit variables as they do in GUIDE and the CLI.
- **Intentional numerical differences:** keep correct weakest-link CV and QR optimality in independently established native counterexamples. These are documented differences, not unchecked tasks to reproduce native defects.

## Evidence and implementation choices

- [Linear fitting and pruning](linear-validation.md): numeric fitting, fallback guards, limits, multilevel trees and the native QR counterexample.
- [Categorical fitting](categorical-linear-validation.md), [selection](selection-validation.md) and [orientation](orientation-validation.md): global/local category rules, the singleton gate, unseen routes and a separate CV-table discrepancy.
- [Importance](importance-validation.md) and [native CV differences](native-cv-difference.md): scoped agreement and explicitly retained differences.
- Double precision, deterministic ties and seeded ChaCha8 randomness are implementation choices; native random streams are not reproduced.
- Unsupported named methods/configurations are rejected. Local manuals/papers and native binaries are optional references; ordinary Rust tests use frozen fixtures and require neither.

Keep this checklist aligned with source changes and their evidence. Passing the current corpus is progress toward GUIDE coverage, not a percentage estimate of full feature parity.
