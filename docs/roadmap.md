# guide-rs roadmap

Oct 23, 2026 · @Brandon

## Overview and goals

GUIDE-RS is an independent Rust reimplementation of Wei-Yin Loh's GUIDE tree algorithm. Gradient boosting and modern ML tooling are planned extensions. Selection bias and uncertainty calibration require empirical and mathematical validation; they are goals, not established properties of the current crate.

The project should keep GUIDE's defining properties (unbiased split-variable selection, interaction detection, model-based leaves) while adding what current practitioners expect (boosting, SHAP, Python bindings, fast columnar input). This document collects every enhancement idea discussed so far, with design notes and references, to use when planning releases.

**Goals and ideas**

- Faithfully reproduce GUIDE's single-tree behavior for regression (Loh 2002) and classification (Loh 2009), validated against the official binary.
- Add a boosting mode that keeps unbiased variable selection, making boosted importance scores trustworthy.
- Provide statistically valid uncertainty: honest leaf estimates, conformal intervals, post-selection inference.
- Reach practitioners through a Python binding with familiar APIs.

**Non-goals**

- Beating LightGBM or XGBoost on raw accuracy or speed. The differentiator is trustworthy selection and interpretation; competitive speed is enough.
- Bit-for-bit reproduction of the GUIDE binary's output. Behavioral agreement on split variables, tree sizes and importance rankings is the target.

## Background: how GUIDE works

GUIDE separates *which variable to split on* from *where to split it*, which removes the selection bias of exhaustive-search trees. CART and C4.5 pick the split that maximizes impurity reduction over all variables and thresholds, so variables with more candidate split points (continuous, many-level categorical) win more often even when they are pure noise (Breiman et al. 1984; Loh & Shih 1997; Strobl et al. 2007). That bias distorts both the tree structure and any importance scores derived from it.

**Core per-node procedure (regression, Loh 2002)**

1. Fit the node model (constant, simple linear, multiple linear, polynomial, Poisson, quantile, proportional hazards, etc.) and compute residuals.
2. Classify each residual by sign (positive vs non-positive).
3. For each predictor, group its values into bins (roughly quartiles for numeric; levels for categorical) and form a 2 × k contingency table of residual sign against bin.
4. Compute a chi-square curvature test per predictor. Convert each statistic to a comparable scale (Wilson–Hilferty transformation to an approximate normal quantile) so predictors with different degrees of freedom compete fairly.
5. Run pairwise interaction tests on the joint bins of two predictors. If an interaction is more significant than every main effect, split on one of the pair (chosen by a secondary criterion).
6. With the split variable fixed, search its thresholds (or category subsets) for the split that best reduces node loss.
7. Grow a large tree, then prune with CART-style cost-complexity pruning and cross-validation (with the 0.5-SE or 1-SE rule options).

**Classification (Loh 2009)** replaces residual signs with class labels in the contingency tables, adds options for kernel and nearest-neighbor node models, linear-combination splits, and uses priors and misclassification costs. Missing values are handled by treating "missing" as its own bin in the tests and sending missing values down a learned direction.

**Other GUIDE capabilities to keep in view:** censored and longitudinal responses, multi-response trees, subgroup identification for treatment effects (Loh et al. 2015; Loh, Man & Wang 2019), unbiased importance scores with bias correction (Loh & Zhou 2021), periodic variables, and GUIDE forests. The pinned GUIDE 46.3 executable is the primary behavioral reference; 46.2 is a secondary regression reference. The bundled manual is version 46.1 (January 24, 2026, 359 pages). See [reference provenance](reference-upstream.md).

See the GUIDE home page and user manual (Loh 2026) for the full option list.

## Validation and testing

Validation comes first: a port is only as credible as the evidence that it behaves like GUIDE. Three layers of tests cover this.

### 1. Unbiasedness suite (headline result)

Show that GUIDE-RS selects split variables without bias, and that exhaustive search does not.

- **Null simulation:** response independent of all predictors. Predictors mix types, for example one binary, one 4-level categorical, one 50-level categorical, one continuous, one continuous with 30% missing.
- **Measure:** frequency each predictor is chosen at the root over 1,000+ replications. Under no bias each should be chosen about 1/p of the time; report a chi-square goodness-of-fit test against uniform.
- **Comparison:** run the same simulation with a CART-style exhaustive search (built into GUIDE-RS as a baseline mode, or via scikit-learn) to show its bias toward the 50-level and continuous variables.
- **Power check:** add a weak true signal on one low-cardinality predictor and confirm GUIDE-RS picks it more often than CART does.
- **Repeat for:** classification, each leaf-model type, the interaction tests, and the boosting mode (selection frequency across all trees).

The resulting plot is the README's headline figure. This design follows the simulations in Loh (2002), Loh (2009), Hothorn, Hornik & Zeileis (2006) and Strobl et al. (2007).

### 2. Oracle tests against the GUIDE binary

- Build a corpus of 20–50 datasets: the manual's worked examples (e.g., RHC, CE, NHTSA data), UCI classics, and synthetic sets with known structure.
- Compare: root split variable, first two levels of split variables, pruned tree size, CV error estimate, and importance-score rankings (Spearman correlation).
- Treat exact agreement as a goal only for split variables; thresholds and tree sizes may differ slightly. Investigate any systematic disagreement as a likely bug.
- Ordinary CI compares frozen, versioned native outputs without distributing or running GUIDE. Native recapture uses locally acquired, pinned executables on compatible hosts; automated redistribution requires reviewing the actual license terms first.

### 3. Property-based tests (`proptest`)

- Test positive affine changes of numerical predictor units on untied, unpruned constant-leaf cases, with floating-point tolerances. Arbitrary monotone transforms can change GUIDE's mean/SD test bins and are not a general invariant.
- Test row/column permutations on untied deterministic cases without resampling. Feature-index ties and seeded CV/bootstrap assignments can change results under reordering.
- Test identical results across thread counts once parallel fitting exists.
- Test category relabeling on cases with unique optimal partitions; tie ordering and binary branch orientation need explicit treatment.
- Test uniform weight rescaling where supported. Duplicating rows with half weights changes row-count tests, minimum sample rules and smoothing, so it is not an unconditional invariant.
- Serialization round-trips preserve predictions and saved routes, including models from earlier algorithm revisions.

### 4. Benchmarks

- Predictive performance (log loss, Brier score and AUROC for classification; squared-error and quantile loss for regression) and training time against LightGBM, XGBoost, CatBoost, scikit-learn trees, `rpart`, `partykit::ctree` and `mboost::blackboost`.
- Use a standard tabular benchmark suite (e.g., the Grinsztajn, Oyallon & Varoquaux 2022 suite) plus datasets with many high-cardinality categoricals, where unbiased selection should matter most.
- Track importance fidelity on synthetic data: rank correlation between estimated and true variable importance.

## Boosting mode

**Status (2026-10-25):** a first version is implemented (`FittedModel::fit_boosting`, `GuideBoostingRegressor`, `GuideBoostingClassifier`): GUIDE tests on gradient signs for squared error and score tests for bin effects for the other losses (reference16), second-order gain, Newton leaves, subsampling, per-node feature sampling and early stopping, for squared error, Poisson, logistic and multiclass. Not yet: Hessian-weighted tables, histogram binning, gradient-sorted categorical search, linear leaves, monotone constraints and the blackboost benchmark.

GUIDE trees plug naturally into gradient boosting, because GUIDE's split tests already operate on residual signs, and boosting fits each tree to pseudo-residuals (negative gradients). The hybrid design: GUIDE's tests choose the split variable; second-order boosting gain chooses the threshold.

### Why it fits

In gradient boosting (Friedman 2001), round m computes the gradient g\_i and hessian h\_i of the loss for each row at the current prediction, fits a tree to them, and adds the shrunken tree to the ensemble. GUIDE's curvature test only needs a residual-like quantity per row, so the signs of −g\_i drop straight in. Squared error, logistic and Poisson provide initial differentiable-loss candidates. Pinball loss is nondifferentiable with zero Hessian away from its kink; quantile boosting needs a separate design. Cox partial likelihood also needs loss-specific treatment rather than this generic rowwise Hessian recipe.

### Per-node algorithm

1. **Variable selection (GUIDE):** form 2 × k tables of sign(−g\_i) against each predictor's precomputed bins; compute chi-square statistics, Wilson–Hilferty normalize, optionally run interaction tests among top candidates. Optionally weight table counts by h\_i, so the test reflects the Newton-step geometry (a variant worth benchmarking against unweighted counts).
2. **Threshold selection (XGBoost-style):** for the chosen variable only, scan bin boundaries and maximize the regularized second-order gain below.
3. **Leaf values:** Newton step w = −G / (H + λ), with G and H the sums of g and h in the leaf.

```latex
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
```

Here λ is the L2 penalty on leaf weights and γ the minimum gain to split (Chen & Guestrin 2016). Only one variable's thresholds are scanned per node, so threshold search is p times cheaper than in XGBoost; the chi-square tests replace that cost.

### Design decisions

| Decision | Recommendation | Rationale |
| --- | --- | --- |
| Leaf model | Constant (Newton step) by default; linear leaves opt-in | Linear leaves over hundreds of trees overfit easily. If offered, regularize with ridge and restrict to a few features, as in LightGBM's `linear_tree` and Shi, Li & Li (2019) |
| Pruning | None; use max depth 2–6, min child hessian, γ | Per-tree CV pruning is too costly and unnecessary with shrinkage and early stopping |
| Regularization | Learning rate 0.03–0.3, row and column subsampling, early stopping on a validation set | Standard stochastic gradient boosting (Friedman 2002) |
| Interaction tests | On for the first rounds or top-k features only | They are O(p²) per node and dominate cost |
| Significance stopping | Optional: stop splitting a node when no test is significant at α | Mirrors conditional inference trees; acts as data-driven depth control |
| Categoricals | GUIDE-style level bins for tests; gradient-sorted level ordering for the split | Sorting levels by G/H reduces the subset search to a linear scan (Fisher 1958; used in LightGBM) |
| Missing values | "Missing" as its own bin in tests; learned default direction in splits | Matches both GUIDE and XGBoost |

### Interpretation caveat

After the first tree, the chi-square p-values lose inferential meaning: the residuals are fitted, rows are reused, and tests are repeated thousands of times. Treat the statistics as a ranking criterion. Do not report them as significance tests.

### Positioning and prior art

- Closest precedent: `blackboost` in R's mboost, which boosts conditional inference trees, another unbiased-selection family (Bühlmann & Hothorn 2007; Hofner et al. 2014; Hothorn, Hornik & Zeileis 2006). Use it as the primary benchmark.
- Expect predictive performance roughly on par with LightGBM, not better: selection bias mainly harms interpretation, not prediction. The value is importance scores (split counts, gain, SHAP) that are not inflated for high-cardinality or continuous noise variables, plus native interaction reporting.
- Benchmark where the difference should show: datasets mixing many-level categoricals with genuine low-cardinality signal.

### Performance notes

- Quantize all features once to `u8` bin indices (≤ 256 bins); GUIDE's test bins are a coarsening of these.
- Build sign-count tables and G/H histograms in a single pass per node.
- Histogram subtraction: compute the smaller child, derive the sibling from the parent.
- Parallelize per-feature tests with Rayon.

## Statistical enhancements

These extend GUIDE's inferential focus: valid uncertainty, stable leaf models, and correct tests on small nodes.

### Honest trees

Split the training data into a structure sample and an estimation sample. Grow the tree on the first; compute leaf estimates on the second. Because leaf values are no longer chosen by the same data that chose the splits, standard confidence intervals for leaf means and leaf regression coefficients become valid (Athey & Imbens 2016; Wager & Athey 2018).

- API: `honest: bool`, `honest_fraction: f64` (default 0.5).
- Report leaf-level standard errors and intervals in tree output.
- Cost: fewer observations for each task. Offer cross-fitting (swap roles and average) as an option.

### Post-selection inference for subgroups

GUIDE's subgroup methods identify treatment-effect subgroups; inference on those effects must account for the search. Loh, Man & Wang (2019) give a bootstrap calibration for confidence intervals of subgroup treatment effects with adjustment for prognostic effects. Implement it alongside the Gi/Gs subgroup models (Loh, He & Man 2015).

### Conformal prediction intervals

Add distribution-free prediction intervals with finite-sample marginal coverage.

- **Split conformal** (Lei et al. 2018): hold out a calibration set, compute absolute residuals, use their (1 − α) quantile as the interval half-width.
- **Conformalized quantile regression** (Romano, Patterson & Candès 2019): fit GUIDE quantile trees for α/2 and 1 − α/2, then conformalize. Gives adaptive-width intervals.
- **Leaf-conditional (Mondrian) conformal:** calibrate within each leaf for intervals that adapt to the tree's partition (Vovk et al. 2005), when leaves have enough calibration points.

### Regularized leaf models (stretch goal)

GUIDE's stepwise selection in linear leaves is unstable, and the reported t-statistics are optimistic after selection.

- Add ridge, lasso and elastic-net leaf fits (Tibshirani 1996; Zou & Hastie 2005), with λ chosen by CV or a fixed default.
- Report coefficients with a warning, or honest-sample estimates, rather than post-selection t-statistics.

### Small-node tests

Chi-square approximations break down when expected cell counts are small (common rule: below 5).

- Merge adjacent bins until expected counts are adequate (GUIDE does a version of this; document the rule found by experiment).
- Fall back to Monte Carlo permutation p-values for very small nodes. Permute residual signs, recompute statistics, count exceedances. Cost is bounded because these nodes are small.
- Optionally, exact Fisher tests for 2 × 2 tables.

### Monotonic constraints

Allow `+1` / `−1` / `0` constraints per feature. Enforce at split time (reject splits whose child values violate order) and propagate bounds down the tree, as in XGBoost and LightGBM. Common requirement in credit, insurance and pricing models.

### Weights and offsets

- Sample weights in all tests (weighted contingency counts), splits, leaf fits and pruning.
- Offsets for Poisson and other GLM-type leaf models (exposure adjustment).

### Probability calibration

Classification trees, forests and boosted models are often miscalibrated.

- Platt scaling (Platt 1999) and isotonic regression (Zadrozny & Elkan 2002) on a held-out set; compare by reliability diagrams and Brier score (Niculescu-Mizil & Caruana 2005).
- Expose as a wrapper so it composes with any model in the crate.

## Ensembles beyond boosting

GUIDE forests already exist in the official software; three ensemble extensions add distributional and causal outputs at modest cost.

### GUIDE forests (baseline)

Bagged GUIDE trees with random feature subsets per node (Breiman 2001; Loh 2025 manual, section on tree ensembles). Unbiased selection inside each tree removes the importance bias documented for random forests (Strobl et al. 2007), which conditional inference forests also address (Hothorn, Hornik & Zeileis 2006). Provide out-of-bag error and permutation importance.

### Quantile regression forests

Store the training responses (or their indices) in each leaf instead of only the mean. A prediction's conditional distribution is the weighted empirical distribution of training responses sharing leaves with the query (Meinshausen 2006).

- Gives any quantile and full predictive distributions from one fitted forest.
- Memory cost: O(n) indices per tree. Offer compact storage (e.g., `u32` indices, or per-leaf quantile sketches).
- Pairs with conformal calibration for guaranteed coverage.

### Distributional boosting

Predict all parameters of a response distribution (e.g., mean and scale of a normal, or parameters of a negative binomial) by boosting each parameter with natural gradients (NGBoost: Duan et al. 2020). A simpler alternative is GAMLSS-style cyclic boosting of each parameter (Mayr et al. 2012).

- Builds directly on the boosting mode: same base learner, vector-valued gradients.
- Outputs probabilistic predictions scored by log-likelihood or CRPS.

### Causal forests (stretch goal)

GUIDE's subgroup models and honest trees together point toward generalized random forests for heterogeneous treatment effects (Wager & Athey 2018; Athey, Tibshirani & Wager 2019). This overlaps with the R package `grf`; only pursue if GUIDE's unbiased selection gives a measurable advantage in subgroup discovery benchmarks.

## Explainability

Interpretability is GUIDE-RS's main selling point, so its explanation tools should be first-class and exceed what mainstream libraries offer on interactions.

### GUIDE importance scores

Implement Loh & Zhou (2021) importance scores, including the bias-correction step that rescales scores so a value above 1 indicates importance beyond noise. These are the native, unbiased alternative to split-count and gain importance.

### TreeSHAP

- **Constant-leaf trees and ensembles:** exact polynomial-time TreeSHAP (Lundberg et al. 2020), O(T · L · D²) per prediction for T trees, L leaves, depth D. Include SHAP interaction values.
- **Linear-leaf trees:** SHAP values for piecewise-linear trees need an extension. Contributions decompose into a path part (as in TreeSHAP) and a within-leaf linear part (coefficient × deviation from a baseline). No standard algorithm exists for this case, so derive it and validate against model-agnostic KernelSHAP (Lundberg & Lee 2017). Separately, Linear TreeShap (Yu et al. 2022) computes exact constant-leaf SHAP in time linear in depth, a speedup worth adopting.
- Validate against the `shap` Python package on constant-leaf models.

### Partial dependence and ICE

- Partial dependence (Friedman 2001) and individual conditional expectation curves (Goldstein et al. 2015).
- Accumulated local effects (Apley & Zhu 2020) as the default when features are correlated, since PDPs extrapolate into empty regions.
- Tree-specific fast path: PD for trees can be computed by weighted traversal without re-predicting the full dataset.

### Interaction reporting

Export GUIDE's pairwise interaction-test statistics as a p × p matrix, aggregated over nodes (weighted by node size) and, for ensembles, over trees. Show alongside SHAP interaction values and Friedman's H-statistic (Friedman & Popescu 2008). No mainstream library offers a test-based interaction matrix; this plays to GUIDE's strengths.

### Tree output

- Text and JSON tree dumps with node sample sizes, leaf models, coefficients and (with honest trees) standard errors.
- LaTeX/TikZ and SVG tree diagrams, matching what the official GUIDE binary produces. We can modernize if possible and lean on the rust ecosystem.
- Rules export: each leaf as a readable rule (`age > 52 AND smoker = yes → ...`).

## Engineering, performance and ecosystem

Adoption depends on bindings and data interop more than on any single algorithmic feature; Python bindings are the highest-leverage engineering item.

### Bindings

| Target | Tooling | API shape | Notes |
| --- | --- | --- | --- |
| Python | [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs) | scikit-learn estimators: `GuideRegressor`, `GuideClassifier`, `GuideBoostRegressor`, etc. | Pass `sklearn.utils.estimator_checks`; accept NumPy, pandas, Polars and PyArrow inputs; release the GIL during fit |
| CLI | `clap` | Optional compatibility mode reading GUIDE description files | Eases migration for current GUIDE users |

### Data input

- [Apache Arrow](https://arrow.apache.org) / [Polars](https://pola.rs) columnar input with zero-copy where possible.
- Native categorical (dictionary-encoded) columns and nulls via Arrow validity bitmaps; no one-hot encoding.
- CSV and Parquet readers for the CLI.

### Serialization and deployment

- [serde](https://serde.rs) model serialization (JSON for readability, a compact binary format such as bincode or MessagePack for size), with a format version field.
- [ONNX](https://onnx.ai) export via the `TreeEnsemble` operators for constant-leaf models.
- Code generation: emit standalone Rust, C or SQL `CASE` expressions for a fitted tree, for deployment without the crate.

### Performance

- **Quantize once:** map each feature to `u8` bin indices (≤ 256 bins) before training, as in LightGBM (Ke et al. 2017). GUIDE's coarse test bins are merges of these.
- **SIMD histogram and contingency-table construction:** sign counts and G/H sums per bin in one pass.
- **Histogram subtraction:** build the smaller child, derive the larger.
- **Parallelism:** Rayon over features (tests), over CV folds (pruning), and over trees (forests). Keep results deterministic regardless of thread count.
- **Subsampled tests at large nodes:** chi-square statistics converge quickly, so tests at nodes with millions of rows can use a fixed-size random sample (e.g., 100k rows) with negligible change in the selected variable. Splits and leaf values still use all rows. Validate the effect empirically.
- **Interaction-test budget:** top-k candidates by main-effect statistic, k configurable (default \~10).
- **Memory:** row-index partitioning in place (no data copies per node); `u32` row indices.

### Reproducibility

- Seeded RNG (e.g., `rand_chacha`) passed explicitly; no global state.
- Documented tie-breaking for equal test statistics and equal gains.
- Model files record crate version, options and seed.

### WASM demo

Compile to WebAssembly (`wasm-bindgen`) and ship a browser demo: upload a CSV, fit a tree, explore it interactively (expand nodes, view leaf models, compare against a CART fit on the same data). A strong showcase for an interpretability-first library, with no server costs.

### Project hygiene

- CI: tests, `clippy`, `rustfmt`, the unbiasedness suite (short version), and oracle tests on a schedule.
- Benchmarks with `criterion`, tracked over time.
- Docs: a book (`mdBook`) explaining GUIDE's method, with the unbiasedness figure up front.

## Prioritized roadmap and open questions

The agreed priority is **validate, then complete GUIDE**, before independent extensions. R prediction-function export is excluded from the product feature list and roadmap by user decision (2026-10-24) and has been removed from the code; it is not an acceptance requirement for upcoming work. Independent R numerical/statistical validation and frozen upstream R output remain test evidence. Existing code already includes partial forests, importance and tree visualization exports; these need validation and completion, not reintroduction as new features.

| Phase | Deliverable and acceptance gate |
|---|---|
| 1. Validate existing core | The constant-node milestone has 20 distinct cases captured under both 46.3 and 46.2: 19 fits and one explicit single-class rejection per version. A further 15 fixed-size/LOOCV cases per version validate pruning conventions, with a demonstrated native CV flaw intentionally excluded from behavioral equivalence (see `native-cv-difference.md`). Eight importance cases per version now validate important sets and independent raw scores; log-tail underflow and missing interaction-copy updates are corrected (see `importance-validation.md`). Exact native calibration remains open. Fifteen root-only numerical linear/stepwise cases per version compare identifiable coefficients/predictions; the numeric fallback difference is resolved and two stepwise gaps remain (see `linear-validation.md`). Stepwise F-tail underflow is corrected. By user decision, stepwise remains experimental and opt-in; further stepwise parity work is deferred. Five subsequent multiple-linear tree cases per version now validate numerical stumps, weighted fits and LOOCV selecting a stump/root. Twelve further tree cases per version now check depth-2/3 growth, weights, missing fitting/split values and fixed paths. Numerical observed-rank bounds are corrected; independent per-row evidence preserves correct four-leaf CV where native chooses three. Forty-seven further cases per version now verify numeric sample fallback thresholds and prediction truncation, including saved-model compatibility. Rust preserves QR optimality in an independently proved native LS counterexample. Forty-one categorical-linear cases per version now validate global dummy references and mixed-design fallback guards, including saved-model compatibility. Model-based categorical subset search through 11 levels is corrected; 36 further selection cases per version resolve the singleton zero-score gap and preserve weak positive/interaction searches (see `selection-validation.md`). Thirty-nine orientation cases per version now verify child directions and missing/unseen routes, with a separately documented native categorical CV discrepancy and independent direct-refit evidence (see `orientation-validation.md`). Next validate multi-predictor zero-score/zero-gain fallback, then broader defaults. **Gate 1:** scoped native comparisons, independent numerical checks and prespecified statistical checks pass; unresolved discrepancies remain explicit. |
| 2. Complete GUIDE capabilities | Prioritize 46.x classification models and missing flags, then missing survival, treatment, multiresponse, longitudinal, propensity, DIF and GUIDE bootstrap/inference capabilities, plus remaining existing-family options. **Gate 2:** each in-scope manual option/combination has documented support or an explicit error, with independent and native acceptance evidence; no full-equivalence claim before coverage is complete. |
| 3. Practitioner interfaces and explanations | Add Python estimators, data interop and explanations after the GUIDE implementation gates. **Gate 3:** estimator compatibility, model round-trips, attribution identities and deployment examples pass. |
| 4. Independent extensions | Evaluate boosting, honest/conformal methods, regularized leaves and distributional/causal extensions as separately specified methods. Their statistical guarantees and importance behavior require independent evidence. |

Phase 3 has started: `python/` holds PyO3 bindings with scikit-learn estimators for single trees. Survival (censored responses) was resumed once the Python wrapper existed and parked again (2026-10-25, user decision) because it depends on native Poisson split points that are not yet understood; [survival notes](survival-notes.md) record the probe findings to resume from. This ordering supersedes the earlier audience-dependent phase ordering. Benchmark and headline claims remain hypotheses until their planned checks pass. In particular, unbiased base-tree selection alone does not establish unbiased boosted importance or SHAP values.

### Open questions

- [x] Priority: validate the existing core, then complete GUIDE before Python, boosting and other extensions. Audience-specific presentation can be decided later.
- [ ] Will Wei-Yin Loh share test cases or endorse the port? (Email pending.)
- [ ] Should boosting weight contingency counts by hessians? Benchmark both variants.
- [ ] Exact bin-merging and tie-breaking rules in the official binary: establish by experiment and log provenance.
- [ ] Default number of interaction-test candidates (k) in boosting: tune on the benchmark suite.
- [ ] Is subsampled testing at large nodes accurate enough? Measure selection agreement versus full-data tests.
- [ ] Crate name and license (MIT/Apache-2.0 assumed).

## References

Citations are from memory except the GUIDE pages, which were checked on 2026-10-23; verify volume and page details before citing in publications.

### GUIDE and Loh's tree methods

- Loh, W.-Y. (2002). Regression trees with unbiased variable selection and interaction detection. *Statistica Sinica*, 12, 361–386. The definitive GUIDE regression reference.
- Loh, W.-Y. (2009). Improving the precision of classification trees. *Annals of Applied Statistics*, 3(4), 1710–1737. The definitive GUIDE classification reference.
- Loh, W.-Y. (2014). Fifty years of classification and regression trees (with discussion). *International Statistical Review*, 82(3), 329–348.
- Loh, W.-Y., & Zhou, P. (2021). Variable importance scores. *Journal of Data Science*, 19(4), 569–592.
- Loh, W.-Y., He, X., & Man, M. (2015). A regression tree approach to identifying subgroups with differential treatment effects. *Statistics in Medicine*, 34(11), 1818–1833.
- Loh, W.-Y., Man, M., & Wang, S. (2019). Subgroups from regression trees with adjustment for prognostic effects and post-selection inference. *Statistics in Medicine*, 38, 545–557.
- Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees. *Statistica Sinica*, 7, 815–840. (QUEST)
- Kim, H., & Loh, W.-Y. (2001). Classification trees with unbiased multiway splits. *Journal of the American Statistical Association*, 96, 589–604. (CRUISE)
- Loh, W.-Y. (2026). *User Manual for GUIDE ver. 46.1*, January 24. University of Wisconsin–Madison. [guideman.pdf](https://pages.stat.wisc.edu/~loh/treeprogs/guide/guideman.pdf)
- [GUIDE Classification and Regression Trees and Forests](https://pages.cs.wisc.edu/~loh/guide.html) (software home page and license)
- [GUIDE version history](https://pages.stat.wisc.edu/~loh/treeprogs/guide/history.txt)

### Trees, selection bias and unbiased recursive partitioning

- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Hothorn, T., Hornik, K., & Zeileis, A. (2006). Unbiased recursive partitioning: A conditional inference framework. *Journal of Computational and Graphical Statistics*, 15(3), 651–674.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8, 25.
- Zeileis, A., Hothorn, T., & Hornik, K. (2008). Model-based recursive partitioning. *Journal of Computational and Graphical Statistics*, 17(2), 492–514.
- Hothorn, T., & Zeileis, A. (2015). partykit: A modular toolkit for recursive partytioning in R. *Journal of Machine Learning Research*, 16, 3905–3909.
- Wilson, E. B., & Hilferty, M. M. (1931). The distribution of chi-square. *Proceedings of the National Academy of Sciences*, 17(12), 684–688.
- Fisher, W. D. (1958). On grouping for maximum homogeneity. *Journal of the American Statistical Association*, 53, 789–798.

### Boosting

- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367–378.
- Bühlmann, P., & Hothorn, T. (2007). Boosting algorithms: Regularization, prediction and model fitting. *Statistical Science*, 22(4), 477–505.
- Hofner, B., Mayr, A., Robinzonov, N., & Schmid, M. (2014). Model-based boosting in R: A hands-on tutorial using the R package mboost. *Computational Statistics*, 29, 3–35.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785–794.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems 31*.
- Shi, Y., Li, J., & Li, Z. (2019). Gradient boosting with piece-wise linear regression trees. *Proceedings of IJCAI 2019*, 3432–3438.
- Duan, T., et al. (2020). NGBoost: Natural gradient boosting for probabilistic prediction. *Proceedings of ICML 2020*.
- Mayr, A., Fenske, N., Hofner, B., Kneib, T., & Schmid, M. (2012). Generalized additive models for location, scale and shape for high dimensional data: A flexible approach based on boosting. *Journal of the Royal Statistical Society: Series C*, 61(3), 403–427.

### Forests and causal inference

- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.
- Meinshausen, N. (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983–999.
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47(2), 1148–1178.

### Uncertainty and calibration

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111.
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems 32*.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*. MIT Press.
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of KDD 2002*.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of ICML 2005*.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.

### Explainability

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Yu, P., Bifet, A., Read, J., & Xu, C. (2022). Linear TreeShap. *Advances in Neural Information Processing Systems 35*.
- Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation. *Journal of Computational and Graphical Statistics*, 24(1), 44–65.
- Apley, D. W., & Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. *Journal of the Royal Statistical Society: Series B*, 82(4), 1059–1086.
- Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. *Annals of Applied Statistics*, 2(3), 916–954.

### Benchmarks

- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems 35* (Datasets and Benchmarks Track).

### Software

- [pyguide](https://github.com/bgreenwell/pyguide): a third-party Python implementation of GUIDE, useful for comparison.
- [PyO3](https://pyo3.rs), [maturin](https://www.maturin.rs), [extendr](https://extendr.github.io), [Apache Arrow](https://arrow.apache.org), [Polars](https://pola.rs), [serde](https://serde.rs), [ONNX](https://onnx.ai).
