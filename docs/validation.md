# Validation record

Local environment: Apple Silicon macOS, Rust 1.99.1. The runtime implementation is pure Rust; R is used only as an independent test oracle. GitHub Actions runs the ordinary suite on Linux, macOS and Windows plus the independent R job; all passed from a clean checkout on 2026-10-25. R prediction-function export was excluded from product scope on 2026-10-24 and removed from the code on 2026-10-25. Historical export checks below remain a record of earlier work; independent R numerical/statistical validation remains in scope.

## Poisson and quantile residual groups, 2026-10-26

New fits record **`guide-rs-0.1-experimental-reference18`**; saved format remains 1. GUIDE's published history (`history.txt`) records two selection rules Rust lacked: version 34.2 groups Poisson rows by Anscombe residuals above or below their mean, and version 28.1 puts zero quantile residuals in the smaller of the positive and negative groups. Probes also showed native leaves a Poisson node with constant response unsplit, where Rust split at a roundoff-level gain. With all three, on fresh random two-predictor designs (seeds 1000 and 2000, no gain threshold, as in native):

| Native 46.3 | Old whole trees | New | Old nodes | New |
|---|---:|---:|---:|---:|
| Poisson (120 designs) | 71 | 78 | 353 | 360 of 427 |
| Quantile (120) | 75 | 84 | 359 | 378 of 456 |
| Quantile, integer responses (80) | 37 | 42 | 193 | 257 of 337 |

Anscombe residuals against their mean did better than against zero (353) or Pearson residuals (355); zeros join the positive group when the groups are equal in size (253 against 246 nodes on integer data). Earlier probes fitted Rust with its default `min_gain = 1e-12`, which under-counted quantile agreement because native keeps zero-gain splits. Eight captures (`poisson-constant-response`, `poisson-anscombe-groups`, `quantile-zero-residuals`, `quantile-zero-residuals-integer`, both versions) bring the corpus to **588 captures across 294 cases**; all fail under reference17.

## Interaction pairs at every depth, 2026-10-26

New fits record **`guide-rs-0.1-experimental-reference17`**; saved format remains 1. In constant-leaf trees, selected interaction pairs now use the two-level search at the last level too, with parent cuts on GUIDE's fixed grid. Identical predictions on fresh pair designs rose from 0 to 64 of 126 (regression, depth 1), 76 to 87 (regression, depth 2), 33 to 116 (classification, depth 1) and 82 to 110 (classification, depth 2). Eight new captures bring the corpus to 580 across 290 cases. Details and the remaining gap are in the [selection validation](selection-validation.md#interaction-pairs-2026-10-26).

## Boosting selection by score tests, 2026-10-26

New fits record **`guide-rs-0.1-experimental-reference16`**; saved format remains 1. Boosted classification and logistic-loss trees selected split variables from the signs of the negative gradients, which equal the class labels in every round, so each node's variable never changed across rounds: on covertype (classes 1 and 2, 50,000 sampled rows, 25% held out, 300 rounds, learning rate 0.1, depth 3) all 300 roots split on Elevation. Those losses and Poisson now select by score tests for bin effects on the current fit; squared error keeps the sign tests and is unchanged (California housing R² 0.8154 before and after).

| Covertype, held out | log loss | AUROC |
|---|---|---|
| guide-rs, sign tests | 0.4818 | 0.8416 |
| guide-rs, score tests | 0.4204 | 0.8851 |
| XGBoost 3.4.1, same settings | 0.4128 | 0.8898 |
| LightGBM 4.7.0, same settings, 8 leaves | 0.4131 | 0.8897 |
| scikit-learn HistGradientBoosting | 0.4150 | 0.8885 |

Null check (five pure-noise predictors with 2, 5, 10, 50 and continuous values, n = 200, 400 replicates): the first tree's root variable is consistent with uniform selection for binary, three-class and Poisson losses (chi-square p = 0.32, 0.96, 0.96). The root of the 20th tree is not (p < 0.005); GUIDE's sign tests for squared error drift the same way (p = 0.0007), because later rounds test residuals of a fit grown on the same predictors. `classification_boosting_moves_on_from_a_fitted_variable` fails under sign selection and passes now.

## Poisson and quantile cut grids, 2026-10-25

New fits record **`guide-rs-0.1-experimental-reference15`**; saved format remains 1. Step-response probes (x = 1..n, response stepping after row k, one run for every k) mapped the cuts native GUIDE 46.3 can choose in Poisson and quantile trees:

- **Fixed grid.** Up to about 100 admissible cuts every position is available; beyond that, exactly 102 grid positions `s - 1 + floor(j (m - 1) / 101)` rows from the left, `m = n - 2s + 1`, confirmed for n = 101 to 300 and minimum node sizes 2, 3, 5 and 10. The report still says "exhaustive search". Least-squares and classification trees are unaffected.
- **Positive counts.** Poisson cuts that leave a child with only zero counts are never chosen; a 0-to-6 step is cut one row late.
- **Sample quantile.** Root-only quantile trees for 6 sample sizes and 5 probabilities match the type 2 sample quantile in all 30 runs; Rust used type 1, which differs at exact ties such as even-sized medians.

On 180 random single-predictor Poisson designs the grid explains 179 native splits, against 104 for exhaustive deviance search. On 40 random two-predictor depth-1 or depth-3 trees per model, root agreement rose from 24 to 37 (Poisson) and 26 to 40 (quantile), node agreement from 81/139 to 111/139 and 101/153 to 126/153, and whole-tree agreement from 16 to 24 and 17 to 27. The remaining differences are mostly the variable selected at deeper nodes. Ten new captures (`poisson-cut-grid`, `poisson-zero-child`, `poisson-exposure`, `quantile-cut-grid`, `quantile-depth-2`, both versions) bring the corpus to **572 captures across 286 cases**; GUIDE reports Poisson observations and predictions with an exposure as rates, which the corpus comparison now handles.

## Minimum rows for quantile linear leaves, 2026-10-25

New fits record **`guide-rs-0.1-experimental-reference14`**; saved format remains 1. Black-box probes of GUIDE 46.3 median-regression trees (depth 3, minimum node size 2, 25 datasets for each of 1, 2 and 3 complete numeric regressors) show that native fits a linear model in a leaf with at least `p + 4` rows and a constant below that: the largest constant and smallest linear leaves were 4 and 5 rows for p = 1, 5 and 6 for p = 2, and 6 and 7 for p = 3. Rust had fitted linear quantile models from `p + 1` rows, which can interpolate small children and win the split search. Quantile leaves with complete numeric regressors now use the native threshold, and `quantile_linear_leaves_need_p_plus_four_rows` covers it. The same probes found no size threshold for Poisson (linear leaves with two rows, including rank-deficient fits) or logistic leaves (smallest linear leaf four rows; constant leaves of up to thirteen rows were degenerate); those keep the `terms + 1` requirement and remain open in the coverage checklist.

## Classification splits that raise impurity, 2026-10-25

New fits record **`guide-rs-0.1-experimental-reference13`**; saved format remains 1. With `min_gain = 0`, classification no longer rejects a split of the selected variable whose count-weighted Gini rises, which priors or costs can cause; this resolves the depth-2 priors difference. Six new captures bring the corpus to **562 captures across 281 cases**; see [selection validation](selection-validation.md#classification-splits-that-raise-impurity-2026-10-25).

## High-cardinality multiclass categorical splits, 2026-10-25

Multiclass trees no longer reject categorical predictors with more than 11 levels. They search cuts on the first two discriminant coordinates, as GUIDE does; fits that were previously accepted are unchanged, so the algorithm revision stays `reference12`. Twelve new captures bring the corpus to **556 captures across 278 cases**; see [reference provenance](reference-upstream.md#high-cardinality-multiclass-categorical-splits-2026-10-25). The prespecified selection bias study now passes all seven configurations and runs in CI.

Local checks pass: **88 regular Rust tests plus the README doctest, 12 R comparisons, 3 release statistical and study checks and 48 Python checks**, plus formatting and Clippy with warnings denied.

## Missing-value flags and categorical orientation, 2026-10-25

New fits record **`guide-rs-0.1-experimental-reference12`**; saved format remains 1, and saved splits gain an optional `missing_codes` field only when a flag is used. Constant-node fits support GUIDE missing-value flags, and categorical splits now put the side with fewer categories on the left, which changes where unseen categories go in unbalanced splits. Fifty-six new native captures bring the corpus to **544 captures across 272 cases**; see [flag validation](missing-flag-validation.md).

Local checks pass: **86 regular Rust tests plus the README doctest, 12 R comparisons, 2 release statistical checks and 48 Python checks**, plus formatting and Clippy with warnings denied.

## Zero-gain splits and same-class merging, 2026-10-25

New fits record **`guide-rs-0.1-experimental-reference11`**; saved format remains 1 and stored trees load unchanged. With `min_gain = 0` the selected variable is split even at zero gain, and classification fits merge same-class sibling leaves bottom-up. Twelve new native captures bring the corpus to **488 captures across 244 cases**; see [selection validation](selection-validation.md#multi-predictor-zero-scores-zero-gain-and-ties-2026-10-25).

The null-selection smoke check previously counted only surviving root splits; merged classification roots now count their selected variable, keeping 500 fits per configuration. Its bound is unchanged and all four configurations pass (p = 0.17, 0.93, 0.53, 0.21).

Local checks pass: **83 regular Rust tests plus the README doctest, 12 R comparisons, 2 release statistical checks and 44 Python checks**, plus formatting and Clippy with warnings denied.

## First commit and post-import fixes, 2026-10-25

The repository received its first commit and CI runs. Fixes since then: bagged ensembles reject CV pruning on bootstrap samples, CSV input skips blank lines and reports line/column locations, the CLI uses clap with file-path error context, and R prediction export was removed with its two export-only R tests. No numerical behavior changed; the algorithm revision remains **`guide-rs-0.1-experimental-reference10`** and saved format remains 1.

Local checks pass: **81 regular Rust tests plus the README doctest, 12 R comparisons, 2 release statistical checks and 44 Python checks**, plus formatting and Clippy 1.99.1 with warnings denied. The Python manual-provenance check skips when the ignored local manual is absent.

## Repository cleanup and standalone verification, 2026-10-24

Feature work was paused for cleanup. The retired `assets/guidevi` nested repository, obsolete 40.4-template fixture prototype, exploratory local probes/drafts, build output and caches were removed. Iris and mtcars CSV files were preserved byte-for-byte in `tests/fixtures/datasets`; the Iris test and README examples now use that standalone path. Static binary inspection defaults to the pinned 46.3 cache. The current native references/history, local PDFs, all 476 frozen acceptance captures and independent diagnostics remain available. Assets, caches and private metadata directories are ignored.

A temporary source copy excluding `.git`, `assets`, `.reference` and build/cache directories passed **all 78 ordinary Rust tests offline**. **44 Python integrity checks**, formatting and Clippy with warnings denied also passed. This cleanup introduced no statistical behavior revision; the algorithm remains reference10. The 14 R comparisons and two release statistical checks recorded below are from the preceding numerical milestone. [Coverage](coverage.md) now separates implemented capabilities, an actionable remaining checklist, and deferred/excluded scope.

## Categorical orientation and unseen routes, 2026-10-24

New fits/reports record **`guide-rs-0.1-experimental-reference10`**; saved format remains 1. Thirty-nine cases per executable add **78 captures**, bringing the fixed acceptance corpus to **476 captures across 238 distinct cases**. Both native versions agree on all new parsed results. Earlier captures remain unchanged.

Bounded exhaustive multiple-linear SSE categorical search now fixes the first locally observed lexical level on the left, including missingness as `NA`. Missing-only partitions use the same categorical metadata. This corrects complementary orientations for complete and missing data and matches actual unseen-level predictions. Tests cover split/both roles, all four-category two-versus-two partitions, case/numeric-looking/reordered labels, excluded prediction levels, weights, row order, local child levels, known/missing/unseen predictions and saved-model compatibility.

One categorical LOOCV table differs: native two-leaf mean 94.72/SE 6.616 versus independently verified Rust/R mean 40.2845495213/SE 4.3957332178. All 96 direct leave-one-out refits under each native version agree with the independent predictions within printed precision. The root score and final two-leaf tree agree. Correct CV is retained, both outcomes are checked explicitly and no native mechanism is claimed. See [orientation validation](orientation-validation.md) for hashed evidence and limitations.

Local checks pass: **78 regular Rust tests, 14 R comparisons, 2 release statistical checks and 44 Python checks**, plus formatting and Clippy with warnings denied. The categorical R check now covers 80 datasets and independently enumerates every categorical partition in every fold of the new CV control. Next investigate multi-predictor zero-score/zero-gain behavior. Stepwise parity remains deferred; R prediction export remains excluded.

## Singleton residual-sign selection, 2026-10-24

New fits/reports record **`guide-rs-0.1-experimental-reference9`**; saved format remains 1. Thirty-six cases per executable add **72 captures**, bringing the fixed acceptance corpus to **398 captures across 199 distinct cases**. Both versions agree on all new parsed results. Existing captures remain unchanged.

Multiple-linear SSE fitting now stops before search when its sole eligible nonconstant predictor has exactly zero residual-sign association. This resolves the original categorical one-leaf versus two-leaf difference, with no special refit or structural exception. The rule is not a conventional significance cutoff: positive scores with p-values above 0.99 still reach ordinary search. Numeric/categorical roles, missingness, unequal weights, constants, row order, interaction switches, competing predictors and pure interactions receive dedicated controls. The change leaves importance, other families, QR and pruning paths intact.

The new suite verifies selection outcomes, residual diagnostics, fitted predictions and serialization. Independent R checks all root sign tables/log tails and weighted fits on the frozen terminal partitions. The earlier categorical suite now compares all 41 cases structurally without a selection exception. An actual saved reference8 two-leaf model retains its old predictions when loaded; refitting now produces one leaf.

Local checks pass: **76 regular Rust tests, 14 R comparisons, 2 release statistical checks and 39 Python checks**, plus formatting and Clippy with warnings denied. Broader all-zero multi-predictor/zero-gain behavior remains unverified. The new missing-category weak control also records complementary child orientations, so its unseen-level behavior is not certified by selection/prediction agreement. See [selection validation](selection-validation.md) for exact scope, evidence and next work. Stepwise parity remains deferred and R prediction export remains excluded.

## Categorical multiple-linear fitting and subset search, 2026-10-24

New fits/reports record **`guide-rs-0.1-experimental-reference8`**; saved-model format remains 1. Forty-one cases per executable add **82 captures**, bringing the fixed corpus to **326 captures across 163 distinct cases**. Both native versions agree on all new parsed results; previous captures remain unchanged.

Multiple-linear SSE fitting now uses the verified global category list, local reference/alias convention and mixed-design nominal sample guard. Checks include missing/excluded/zero-weight levels, numeric-looking and case-sensitive labels, weighted/duplicate factors, local references with numeric missingness, four truncation modes, LOOCV and saved categorical model compatibility. Future prediction calls retain the fitted codebook. Model-based categorical search enumerates subsets through 11 levels, correcting a case where response-mean ordering missed the best slope partition.

At this milestone, an explicit selection compatibility gap remained: with four categories and perfectly balanced residual signs, native returned one leaf while Rust searched a beneficial two-leaf split. The shared root has SSE 28750; the two-leaf fit has essentially zero SSE. Reference9 above resolves that difference using the verified singleton zero-score rule. Correct QR and correct pruning remain unchanged. See [categorical validation](categorical-linear-validation.md) for conventions, limitations and reproduction.

Local checks pass: **74 regular Rust tests, 13 R comparisons, 2 release-mode statistical checks and 35 Python checks**, plus formatting and Clippy with warnings denied. The new R check verifies all 41 categorical datasets, candidate partitions and both LOOCV candidates without prediction-code export. Remote CI remains unobserved. Next resolve model-based selection gating with categorical/numeric controls, then broader defaults. Stepwise parity remains deferred; R prediction export remains excluded.

## Numeric multiple-linear policy and prediction limits, 2026-10-24

New fits/reports record **`guide-rs-0.1-experimental-reference7`**. Ninety-four new native captures from 47 cases per executable bring the fixed corpus to **244 captures across 122 distinct cases**. Both native versions agree on all new parsed results; earlier captures remain unchanged. Captures skip R export and use genuine missing-response rows to obtain native held-out predictions.

Numeric LS constant-fallback thresholds now match the verified nominal-column/global-missing-flag policy. Boundary checks cover 1/2/3/4/7 predictors, constants, collinearity, weights, zero weights, missing/all-missing columns, excluded rows and global flags affecting complete child nodes. The earlier four-row/two-predictor native discrepancy is resolved. Non-numeric/other-family fallback conventions remain open.

`prediction_truncation` supports `none`, `node_range`, `node_range_expanded` and `global_range` for multiple-linear squared-error leaves. Rust retains `none` as the compatible default; native's default is global range. Global bounds are fixed from full active training data across CV; node bounds are fold-local. Search/weakest-link risks and fitted residual signs remain unbounded, while predictions and held-out losses use the limits. Tests cover training clipping, held-out/missing predictors, zero-weight outliers, CV, unchanged fixed-size alpha paths, extreme inputs, malformed bounds, CLI operation and JSON round trips. Old JSON remains unbounded and is not refitted; saved format remains 1.

One expanded missing-data design exposes a native LS correctness failure: native training SSE 3.25 versus independent R QR/Rust SSE 2.70, with a −0.5 normal-equation residual for `x1.NA`. Rust keeps correct QR fitting, with both outcomes independently recorded and checked. [The evidence](../tests/fixtures/linear-policy/diagnostics/native-qr-difference.json) pins source/input/native hashes and both fitted vectors. See [linear validation](linear-validation.md) for complete policies, tolerances, scope and remaining differences. The independent R test verifies estimation/limits and both CV conventions without generating prediction-code exports.

Local checks pass: **70 regular Rust tests, 12 independent R comparisons, 2 release-mode statistical checks and 31 Python checks**, plus formatting and Clippy with warnings denied. The fixed-seed null-selection counts and the 1/50 null-importance result below remain unchanged. Remote CI results remain unobserved. Next validate categorical fitting/dummy conventions and their fallback policies, then broader family/default combinations. Stepwise parity remains deferred and R prediction export remains excluded from product scope.

## Multilevel linear trees and numerical rank bounds, 2026-10-24

The fixed corpus now contains **150 native captures across 75 distinct cases**, including 24 new captures from twelve multilevel multiple-linear cases per executable. Both native versions agree on all new parsed results. The preceding 126 captures are unchanged. New cases cover depth-2/3 growth, split-and-fit roles, weights, missing fitting/split values, fixed-size paths and weighted/unweighted LOOCV. See [linear validation](linear-validation.md) for exact settings and limits.

The missing-split extension first reproduced an invalid Rust cut that used missing rows to satisfy a numerical observed-rank constraint. Candidate generation now matches the native observed-value bounds, including the asymmetric upper endpoint: at least the minimum observed count left, strictly more right. Additional local controls under both pinned versions and portable endpoint regressions validate the convention. Two four-row CLI/export smoke cases explicitly use minimum one to retain their intended two-leaf setup. New fits/reports record **`guide-rs-0.1-experimental-reference6`**; saved-model format remains 1 and old stored cuts/routes remain unchanged on load.

Two new CV counterexamples demonstrate a different final size: correct Rust CV selects four leaves while both native versions select three. An independent weighted-QR/exhaustive-fold oracle verifies every correct CV stage, while simulation of the known node-order scan reproduces all native means/SEs and selections. Regular tests check the correct four-leaf result and separately reconstruct/check the native final three-leaf tree with fixed-size pruning. Full fixed-size paths also receive independent per-stage route/prediction checks. Script/input hashes and every held-out prediction are frozen; Python independently recomputes means/SEs, and an R test regenerates the evidence. No unexplained CV difference is suppressed. See [the native CV difference](native-cv-difference.md).

Local checks pass: **64 regular Rust tests, 11 independent R comparisons, 2 release-mode statistical checks and 26 Python checks**, plus formatting and Clippy with warnings denied. Null-selection counts and the 1/50 null-importance result below remain unchanged after the rank correction. Remote CI results remain unobserved. Next address multiple-linear small-sample fallbacks and truncation, then categorical fitting and broader pruning/default combinations. Stepwise parity remains deferred.

## Stepwise deferred; multiple-linear tree acceptance, 2026-10-23

The user chose to retain stepwise as an **experimental, opt-in compatibility feature** and defer further native stepwise parity work. The decision is recorded in the README, API documentation, algorithm specification, coverage inventory and enhancement roadmap. Existing stepwise tests and explicit differences remain in place.

Ten subsequent native captures cover five multiple-linear tree cases under each executable, bringing the corpus to **126 captures across 63 distinct cases**. Both native versions agree on all new parsed results. Rust agrees on numerical split-only/split-and-fit stump growth, unequal fitting weights, original-unit leaf coefficients, predictions and LOOCV retaining a stump or selecting the root. Both CV candidate means and naive SEs are checked. An independent R test verifies leaf fits and independently searches all valid cuts in each held-out fold. See [linear validation](linear-validation.md) for settings and limitations.

No runtime algorithm change was needed; behavior revision remains **`guide-rs-0.1-experimental-reference5`** and saved-model format remains 1. Local checks pass: **61 regular Rust tests, 10 independent R comparisons and 23 Python checks**, plus formatting and Clippy with warnings denied. The two statistical smoke checks last passed in the preceding milestone, with no runtime fitting changes since. Multilevel multiple-linear growth/pruning and fitting fallbacks remain priorities; further stepwise parity is deferred. Remote CI results remain unobserved.

## Linear/stepwise leaves and F-tail correction, 2026-10-23

New fits and importance reports record **`guide-rs-0.1-experimental-reference5`**. Thirty new native captures bring the corpus to **116 captures across 58 distinct cases**: 70 constant fitting/pruning captures, 16 importance reports and 30 root-only linear/stepwise fits. All earlier captures remain unchanged. Both native versions agree on parsed results in the new family after removing version/build metadata.

Twelve of the 15 new cases per executable agree with Rust on identifiable coefficients and training predictions. The other three preserve specific unresolved differences: small-sample multiple-linear fallback, collinear stepwise fallback and missing-data stepwise selection. Independent R `lm` checks verify Rust estimation given its selected columns on all 15 datasets. These checks do not establish native stepwise selection or model-based tree equivalence. See [linear validation](linear-validation.md) for the cases, tolerances, exact differences and reproduction commands.

A new regression test first reproduced stepwise selection of a weaker first column when both F probabilities underflow. Log-space F tails, risk ratios and Bonferroni comparisons now select the stronger predictor in either column order, matching both native executables. A 54-value independent R comparison covers F log tails up to `f64::MAX`, with degrees of freedom up to 100,000. The F(2,2) identity supplies a separate analytic check.

Local checks pass: **60 regular Rust tests, 9 independent R comparisons, 2 release-mode statistical checks and 19 Python checks**, plus formatting and Clippy with warnings denied. The fixed-seed statistical counts below remain unchanged. Saved-model format remains 1; no remote CI result is claimed. The user subsequently chose to retain stepwise as experimental and opt-in, deferring further stepwise parity work. The next native priority is multiple-linear split selection, pruning and fitting fallbacks; see the decision in `linear-validation.md`.

## Association tails and importance acceptance, 2026-10-23

New fits and importance reports record **`guide-rs-0.1-experimental-reference4`**. Sixteen new native importance captures bring the fixed corpus to **86 captures across 43 distinct cases**: 70 fitting/pruning captures plus 16 importance captures. The earlier captures are unchanged.

Tests first reproduced three extreme-tail failures: both ordinary probabilities rounded to zero and selected the weaker first column; transformed importance statistics saturated at the same 1e-300 floor; and zero/subnormal probability quantiles were capped instead of preserving their mathematical meaning. Log-space tails, Bonferroni comparisons and inverse tails correct these failures. Regression/classification checks cover column permutations, mixed degrees of freedom and a subnormal significance level. Four additional local native probes confirm the stronger predictor is also selected by both GUIDE versions on the 6,000-row design.

The new native importance cases cover duplicate predictors, numerical missingness, mixed predictor types and a duplicated interaction variable. They exposed a missing propagation of interaction scores to exact typed predictor copies, a requirement documented in the 42.5 changelog. That failure was reproduced before correction. Rust also reports `indeterminate` rather than arbitrarily assigning different groups across tied 95%/99% score boundaries. Importance validates its grown trees before scoring, returning an error for nonfinite statistics instead of silently reporting zero importance.

The native comparisons check important-variable sets and identifiable rank relations. The native files contain normalized scores; raw Rust scores are checked independently from Pearson statistics and R-validated transforms. Native calibration magnitudes, complete adjusted rankings and fallback normalization remain unverified. Identical predictors receive different normalized native magnitudes but equal Rust raw/null-mean/adjusted scores under shared response permutations; no exact-calibration equivalence is claimed. See [the importance validation details](importance-validation.md).

Local checks pass: **57 regular Rust tests, 7 independent R comparisons, 2 release-mode statistical checks and 15 Python checks**, plus formatting and Clippy with warnings denied. The new R comparison covers 56 forward log tails and 56 inverse tails, including df 100,000 and log probabilities −1,000,000. Fixed-seed null-selection counts and the 1/50 null-importance result below are unchanged.

Saved model format remains 1. Optional diagnostic log fields and an optional importance-report algorithm revision preserve older JSON reads. New `indeterminate` importance groups and the zero-tail inverse's infinity result are documented API changes. Next native priority: existing model-based leaves, followed by remaining importance calibration/default combinations. No remote CI run or full GUIDE equivalence is claimed.

## Pruning and CV acceptance, 2026-10-23

New fits record **`guide-rs-0.1-experimental-reference3`**. The corpus now has **70 native captures across 35 distinct cases**, including 15 new pruning/CV cases per executable. Both versions agree on all parsed native results after excluding their banners/build dates. The original 40 captures remain unchanged; saved model format remains 1.

The new fixtures first reproduced failures in tied fixed-size pruning, classification cutpoint ties, CV priors, prior weighting of held-out costs and the SE formula. Corrections retain separate tied pruning stages, choose balanced numerical classification cuts on equal loss, freeze full-training priors within CV folds, average actual held-out classification costs and use the population naive SE. The unpruned CV candidate uses the full fold tree, and the root candidate uses fold roots. Weighted means/SEs retain weight-unit invariance, with an explicit conversion from native weighted-error units.

An intermediate CV discrepancy led to a read-only debugger inspection of GUIDE 46.3. Its fold-pruning candidate scan ignored a cheaper branch solely because of its smaller node number. Rust deliberately keeps correct weakest-link pruning, with this difference explicitly documented and independently checked. See [the evidence and calculations](native-cv-difference.md). For the two affected cases, tests verify both native error estimates and correct Rust estimates; all other captured stages and every final tree/prediction receive normal native comparisons. Matching final trees in these examples is not a claim that their CV errors agree.

CV fold membership now uses a precomputed index instead of repeated linear searches. Held-out evaluation no longer constructs a full Context for every candidate, also addressing two performance findings in `REVIEW.md`.

The parser also caught a draft batch alignment error caused by an omitted weighted-pruning question. Drafts remain local diagnostics; final captures verify the effective depth, minimum child size and CV folds. New failure tests reject setting drift, incomplete paths, invalid CV values and inconsistent selections.

Local checks pass: **48 regular Rust tests, 6 independent R comparisons, 2 release-mode statistical checks and 11 Python checks**, plus formatting and Clippy with warnings denied. The fixed-seed selection counts and the 1/50 null-importance discovery result below are unchanged. The R one-predictor `rpart` pruning checks still pass.

Next: native importance and existing model-based leaves. Test-sample pruning, broader weighted/multifold CV, model-based pruning and additional native CV variants remain unverified. This is scoped acceptance with an explicit correctness exception, not full GUIDE parity. Remote CI has not been run in this session.

## Constant-node native acceptance, 2026-10-23

Behavior revision: **`guide-rs-0.1-experimental-reference2`**; saved model format remains 1. The replacement manual is verified as GUIDE 46.1, January 24, 2026, 359 pages. Its SHA-256 and source URL are pinned in `reference-upstream.json`. GUIDE 46.3 remains the primary executable target, with 46.2 as a secondary reference.

The expanded corpus has **20 distinct cases under each executable (40 captures)**: 19 fitted trees and one explicit single-class classification rejection per version. Native parsed results agree between versions after excluding version/build fields. Cases cover constant least squares and classification, numerical/categorical missingness, competing mixed predictors, weights, interactions, ties, multiclass predictions, priors and asymmetric binary costs. See [reference provenance and reproduction](reference-upstream.md#constant-node-acceptance-corpus-2026-10-23) for the case inventory, artifact formats and declared tolerances.

The native comparisons initially failed for probability smoothing, categorical orientation/default routing, interaction ties, prior/cost cutpoints, an extra multiclass leaf caused by roundoff, and single-class input acceptance. Focused comparison assertions preceded the fixes. The zero-posterior correction follows manual 46.1 p.27; raw counts are normalized once, and growth/pruning uses unsmoothed posteriors. Split Gini uses observation-count weighting and class cost-sum adjustment. Unknown categories default right. Single-class training is rejected while bootstrap samples retain the global class universe.

Rust comparisons check complete shallow-tree split structure, categorical partitions, numerical thresholds, missing/unseen routes, leaf counts, training-row terminal assignments, predictions and probabilities. Existing six stump captures remain unchanged. Existing serialized models keep their stored routes and probabilities; a regression test explicitly loads empirical probabilities under the older algorithm revision. R export preserves stored single-tree class decisions even when an extreme misclassification cost would change a decision recomputed from corrected probabilities.

Local checks pass:

- **48 regular Rust tests**, including the 40 native captures, conditional probability correction, legacy-model preservation and cost-unit scaling by 1e-200/1/1e200.
- **6 independent R comparisons**, including the extended classification-export check; R 4.5.2, `quantreg` 6.1 and `rpart` 4.1.27.
- **2 release-mode statistical checks**; all root-selection counts/p-values and the 1/50 null importance result below are unchanged.
- **8 Python checks**, covering all file/executable hashes, reproducible catalog inputs, manual fingerprint, output parsing, version mismatch, process failure/timeout, explicit rejection semantics and refusal to overwrite captures.
- `cargo fmt --check` and `cargo clippy --locked --all-targets -- -D warnings`.

Eight additional local numerical/cost probes (four binary and four three-class datasets with estimated or specified priors) matched the analytical observation-count Gini criterion under asymmetric costs. These supplementary exploratory records were removed during the 2026-10-24 cleanup; they were not frozen acceptance cases or validation of arbitrary cost matrices.

This gate covers only the listed constant-node cases. Native pruning/CV, importance calibration, model-based leaves, periodic/discriminant behavior and broader defaults still require acceptance work. The enhancement roadmap now prioritizes that work and completing missing GUIDE families before Python, boosting and other extensions. Cross-platform CI remains configured but has not been run remotely in this session.

## Initial native reference update, 2026-10-23

Apple Silicon GUIDE **46.2 and 46.3 now run successfully**. Six frozen least-squares stump captures compare root feature/cutpoint, leaf count, predictions and the missing-value direction printed by GUIDE. They exposed and corrected Rust's default missing route for numerical thresholds when no training values are missing. At this initial stage newly fitted models recorded revision `guide-rs-0.1-experimental-reference1`; see [reference-upstream.md](reference-upstream.md) for provenance and scope. The site/history end at 46.2 even though the current downloaded binary identifies itself as 46.3. The primary target is now 46.3; 46.2 fixtures provide secondary regression comparisons. These small cases do not establish full-version parity.

After this correction, **45 regular Rust tests, 6 R comparisons, and 3 Python fixture-integrity/failure tests pass**. The two statistical smoke tests from the numerical audit below passed; the routing change affects future missing cases, not those tests' training-data scores. Formatting and Clippy are checked separately. The original 40.4 launch failure remains unresolved but no longer prevents acquiring native reference outputs.

## Numerical audit, 2026-10-23

Behavior revision: `guide-rs-0.1-experimental-audit1`. JSON schema version remains 1. Newly fitted models record this revision; existing serialized coefficients are not refitted automatically.

The audit added 11 regression tests in `tests/audit.rs` and two independent R comparisons. Tests were written to reproduce failures before their fixes; the table also includes boundary checks added during the same audit.

| Area | Finding and correction | Validation |
|---|---|---|
| Split objectives | Fixed absolute epsilons could select the first admissible cut instead of the best one when loss was small. Early stopping also compared unnormalized impurity to a normalized gain threshold. Comparisons now use relative roundoff tolerance, and stopping uses root-weight normalization. | Same exact stump after response scaling by 1e-8/1e100 and weight scaling by 1e-20/1e100, with explicit `min_gain=0`. |
| Pruning and CV | Absolute tie tolerances could collapse meaningful small-loss branches; multiplying alpha endpoints and squaring large losses could overflow, including a CV panic from a NaN cutoff. Alpha interpolation and variance calculations now avoid those overflows. The final CV candidate always evaluates root-only fold models. | Pruning across response scales 1e-8 to 1e100; weighted CV means/SEs across weight scales 1e-200 to 1e200. One-predictor pruning agrees with R `rpart` for ten alpha values, comparing both leaf counts and prediction loss. |
| Quantile optimization | Scaling an uncentered response by a large origin allowed premature convergence. Centering is now applied only when a constant design column permits it. | Adding 1e9 to responses preserves intercept-adjusted coefficients at quantiles 0.1/0.5/0.9 within 5e-6. |
| Poisson exposures | Clipping the log-rate before combining exposure changed fitted means when exposure units changed. Direct sums of extreme exposures could overflow and produce a spurious zero rate. Log-rate and log-exposure are now combined before exponentiation; exposure averages are scaled. | Known log-rate recovery under exposure multipliers 1e-40, 1, 1e40 and 1e307; noisy weighted fits and generated R predictions agree with R `glm` at 1e-40/1/1e40. |
| Poisson categorical search | Sorting by mean counts ignored exposures and missed a zero-deviance partition. Levels are now ordered by exposure-adjusted rate. | A three-category case recovers the exact zero-deviance split. |
| Poisson deviance | Subtracting nearly equal first-order terms lost accuracy near a perfect fit; nonfinite means could also be masked by clamping NaNs. A local series avoids cancellation, and nonfinite means produce nonfinite loss. | Near-perfect mean model agrees with the analytic quadratic expansion to relative tolerance 1e-6; the old calculation missed by about 35%. |
| Stepwise tests | An absolute “perfect fit” cutoff made small-unit responses select an irrelevant variable. Only exact zero residual loss receives that shortcut. | Identical selected groups after response scaling by 1e-14. |
| GLM status | Small objective improvement alone could declare convergence during coefficient divergence. Convergence now requires a small coefficient update; degenerate outcomes retain exact constants. | All-zero Poisson and all-zero/all-one logistic cases predict exactly; a completely separated logistic case reports nonconvergence. This is not a general separation detector. |
| Validation failures | Missing validation exposures were silently treated as one. Nonfinite training statistics could reach pruning before being rejected. | Missing exposures and overflow return errors before pruning. |

Local results: **44 regular tests, 6 optional R comparisons, and 2 release-mode statistical tests pass**. Formatting and Clippy with warnings denied pass. Oracle environment: R 4.5.2, `quantreg` 6.1, `rpart` 4.1.27. Statistical counts below were unchanged by the audit. Tolerances and fixed seeds live in the tests.

```sh
cargo fmt --check
cargo clippy --offline --all-targets -- -D warnings
cargo test --offline --all-targets
cargo test --offline --test interoperability -- --ignored
cargo test --offline --release --test statistical -- --ignored --nocapture
```

CI now includes a separate Linux R-comparison job using the documented [R setup action](https://github.com/r-lib/actions/tree/v2/setup-r) and [dependency setup action](https://github.com/r-lib/actions/tree/v2/setup-r-dependencies), with oracle versions printed in its log. The workflow is configured locally; no remote CI result is claimed.

These checks validate particular numerical objectives, invariants and failure behavior. They do not certify GUIDE parity, every ill-conditioned design, statistical inference, or every combination of priors, costs and missingness. One-predictor `rpart` agreement is a pruning check, not an assertion that GUIDE and CART select variables identically. The pinned 46.3 oracle now supports differential acceptance; expanding its coverage remains necessary.

## Deterministic and independent checks

- Distribution reference values: gamma, chi-square tails/quantiles, incomplete beta and F tails.
- Weighted and rank-deficient least squares; independently compared with R `lm`.
- Weighted linear and quadratic quantile fits at 0.1/0.5/0.9 independently compared with R `quantreg::rq(method="br")`; exported R predictions also agree. Small two-parameter problems at 0.01/0.1/0.25/0.5/0.9/0.99 match exhaustive enumeration of lines through pairs of observations. Tests check dual bounds, stationarity and objective gaps, rank deficiency, zero-weight exclusions, CV pruning and bagging.
- Logistic and weighted Poisson IRLS independently compared with R `glm`; Poisson log-rate recovery on a known model.
- Exact regression stumps, classification priors/costs, XOR requiring lookahead, missingness-only signals, categorical and periodic routing.
- Stepwise removal of irrelevant groups, bivariate discriminant separation, simultaneous quantiles identifying variance changes.
- Cross-validation, external-validation, fixed-size pruning, bagging/forests and OOB predictions; one-predictor cost-complexity paths compared with R `rpart`.
- Saved-model prediction equality, malformed input rejection, CSV quoting/reordered headers, CLI output protection.
- Generated R classification/regression/quantile predictions compared against Rust, including ensembles, missing values, categories and linear splits.
- Bundled Iris smoke test requires at least 140 correct training predictions out of 150 after pruning. This is a regression check, not an out-of-sample accuracy claim.

## Prespecified statistical smoke checks

`cargo test --release --test statistical -- --ignored --nocapture`

The null selection test runs 500 datasets per condition (200 rows each) with independent binary, four-category, and continuous predictors. The missing-data condition makes 25% of the continuous predictor missing. Four goodness-of-fit comparisons use a predeclared rejection level of 0.001/4.

| Response model | Missing continuous predictor | Root selections: binary / categorical / continuous | Goodness-of-fit p |
|---|---|---|---|
| Squared error | No | 179 / 147 / 174 | 0.168976 |
| Classification | No | 168 / 167 / 165 | 0.986098 |
| Squared error | Yes | 171 / 155 / 174 | 0.534726 |
| Classification | Yes | 175 / 149 / 176 | 0.245122 |

The importance check uses 50 independent null datasets, 80 rows, the same mixed predictors with missing values, and 300 permutations per dataset. The nominal 95% threshold selected at least one variable in **1/50** datasets. Its prespecified smoke-test bound rejects 10 or more selections; it is intended to catch gross excess false discoveries, not establish exact 5% calibration. Seeds and rejection bounds are committed in the test, not chosen after seeing results.

These checks do **not** demonstrate unbiasedness in every distribution, adequate power, calibrated uncertainty, or equivalence to GUIDE 46.3. Treatment/survival/longitudinal behavior has not been implemented or tested. The original bundled GUIDE executable cannot currently supply runtime reference results; new 46.x downloads now provide the limited native suite described above.
