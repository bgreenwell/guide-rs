# Working upstream GUIDE references

Retrieved October 23, 2026 from the [official Apple Silicon directory](https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/). Both executables launch and complete model fitting on this Apple Silicon host running macOS 26.5.1. Both Mach-O files declare minimum macOS 26.0 and depend only on `/usr/lib/libSystem.B.dylib`.

| Download | Observed runtime banner | Local executable |
|---|---|---|
| [`guide.zip`](https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip) | **46.3**, build July 11, 2026 | `.reference/upstream/macos-arm64/guide` |
| [`guide462.zip`](https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide462.zip) | **46.2**, build April 27, 2026 | `.reference/upstream/macos-arm64-46.2/guide` |

Archive and executable SHA-256 values, source URLs and runtime metadata are recorded in [reference-upstream.json](reference-upstream.json). Downloads and inspection logs are local-only in the ignored `.reference/` directory. Archives were inspected and extracted with `ditto`, preserving their metadata; no security attributes were removed. The retired wrapper and its old bundled executable were removed during cleanup on 2026-10-24; current pinned references and frozen evidence were preserved. The launch probe now supplies the final newline requested by the disclaimer menu, so successful launches exit normally instead of encountering EOF.

The current archive URL is mutable. Its banner, hash and build date are authoritative for each captured fixture; the filename alone is not a version guarantee.

## History and target boundaries

The [official page](https://pages.stat.wisc.edu/~loh/guide.html) and [revision history](https://pages.stat.wisc.edu/~loh/treeprogs/guide/history.txt) advertised 46.2 at retrieval, although `guide.zip` actually contained 46.3. The downloaded history ends at 46.2 and is retained at `.reference/upstream/history.txt`; its SHA-256 is in the metadata JSON. No 46.3 algorithm changes are inferred from that history.

Documented changes since the originally bundled 44.1 manual (now replaced by 46.1):

| Version | Consequence for validation |
|---|---|
| 45.0 | Defaults and input-file questions changed. Older batch templates cannot be assumed compatible, and defaults must be made explicit in fixtures. |
| 45.1 | Multiple/stepwise linear trees default to four split levels; R export handles spaces in variable names. |
| 45.2 | R export fixes cover all-missing B/C variables; M flags become B variables for multiple/stepwise models; output gains constant-variable annotations. |
| 46.0 | Classification adds LDA/QDA node models and discriminant-coordinate split options. |
| 46.1 | A classification memory-deallocation bug was corrected. |
| 46.2 | The new classification options were improved. |

These are summaries of the upstream changelog, not an exhaustive specification. The project now targets **46.3**, using its pinned executable as the primary behavioral oracle and 46.2 as a secondary regression reference. The bundled manual is now 46.1, January 24, 2026, 359 pages; its source URL and SHA-256 are recorded in the metadata JSON. No 44.1-matched executable is required to advance acceptance. Where older documentation differs from observed 46.3 behavior, record and test the difference; undocumented changes cannot be inferred from the 46.2 history alone.

## First differential fixtures

`tests/fixtures/reference/guide-46.2/` and `guide-46.3/` each contain three real least-squares fits: one numerical predictor, constant nodes, exhaustive cutpoints, depth one, minimum child size two, no pruning. The 20-row datasets have 5, 10 or 15 observations in the lower-mean group. Each capture retains the input batch, DSC/data files, stdout/stderr, full report, split file, predictions and a manifest containing file hashes and the actual runtime version.

The input template was generated interactively by GUIDE 46.3. The capture script makes minimum node size explicit and adapts only the version header for 46.2; successful execution and runtime banners verify both versions.

The initial comparison found:

- Root feature, cutpoint, leaf count and all observed-row predictions agreed with Rust.
- With no training missingness, GUIDE's reported numerical split sends future `NA` values right in all six cases, regardless of child sizes. Rust had sent them to the larger child, choosing left on ties.

Rust now preserves the default right route for numerical thresholds when training has no missing values. With training missingness, both directions are still optimized. The expanded corpus below subsequently established categorical routing as well; linear and periodic defaults remain unverified. Stored models retain their serialized directions; this initial correction used behavior revision `guide-rs-0.1-experimental-reference1`.

`tests/reference.rs` compares the six frozen results, including missing-value direction parsed from the GUIDE branch text. This is a focused regression check, not blanket GUIDE parity. Classification, weighted/missing training cases, categorical routing and interactions are covered by the expanded corpus below. Periodic routing, coefficients and pruning still need native suites.

To reproduce a capture, choose a new output directory:

```sh
python3 tools/capture_reference_stump.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --left-size 10 --output /tmp/guide-463-ls-stump
cargo test --test reference
python3 -m unittest discover -s tools -p 'test_reference_*.py'
```

Captures refuse overwrites. A nonzero process exit, wrong runtime version, incomplete tables or absent missing-route evidence leaves an incomplete diagnostic record rather than a passing fixture. The Python checks verify every retained file hash and exercise version/parse failure handling. Rust fixture tests require neither GUIDE nor Python and run in the normal cross-platform test suite.

## Constant-node acceptance corpus, 2026-10-23

There are **20 distinct cases captured under both versions (40 captures)**. Nineteen cases per version contain fitted trees; the remaining case records GUIDE's explicit rejection of single-class classification input. Both native versions agree on all parsed results after excluding their version/build metadata.

| Family | Cases |
|---|---|
| Least squares | Original numerical stumps with left sizes 5/10/15; numerical missingness; categorical missingness; competing mixed predictors; pure interaction; unequal weights; tied predictors; constant response |
| Classification | Balanced binary; imbalanced binary; multiclass; categorical predictor; numerical missingness; categorical missingness; explicit priors; asymmetric costs; pure interaction; single-class input rejection |

All fitted cases use constant models, exhaustive numerical cutpoints, explicit minimum child size two, depth at most two, no pruning and no linear splits. The prior/cost cases contain both classes in both broad regions; they test changed cutpoints and mixed posteriors as well as class decisions. These small synthetic cases do not validate the complete classification or regression families.

The original six format-1 stump captures are unchanged. New format-2 captures include `dataset.json` and complete `options.json`, the native batch/DSC/data, any prior/cost files, stdout/stderr, report, split encoding, prediction table, file hashes and declared tolerances. The classification batch follows the 46.3 interactive writer; GUIDE indexes costs as [predicted][true], so the harness explicitly transposes Rust's [true][predicted] matrix.

The expanded tests compare every split feature, numerical threshold, categorical left set, missing direction, unseen-category direction, leaf count, training-row terminal-node assignment, prediction, class probability and model round-trip. Native probabilities are printed with five significant digits; absolute probability tolerance is 5e-6. New numerical predictions use absolute tolerance 5e-5 plus relative tolerance 5e-6, and cutpoints use 1e-9 absolute tolerance. The original stumps retain their tighter existing prediction tolerance. These limits were declared in the capture harness before Rust comparisons.

Corrections in behavior revision **`guide-rs-0.1-experimental-reference2`**:

- Zero class posteriors receive the correction specified on manual 46.1 p.27: (N p_j + prior_j)/(N+1), using the full training sample size. Nodes with all positive posteriors are unchanged. Correction happens after search/pruning, preserving pure-node stopping and unsmoothed class decisions. R exports preserve the same stored class decisions under extreme costs.
- Class counts are accumulated before normalization. This avoids probabilities slightly above one and negative pure-node Gini from repeated normalized additions; the multiclass fixture previously grew an extra leaf.
- Child Gini is weighted by raw observation count, rather than prior-adjusted mass (Loh 2009 equation 2.2). Splitting probabilities also incorporate class cost sums. Explicit binary prior/cost fixtures establish this behavior; broader multiclass costs remain unverified.
- Categorical splits send unseen levels right. Without observed missingness they also send future missing cases right. Binary classification uses the descending class-proportion orientation observed in native outputs; regression retains ascending response ordering.
- Equal two-level interaction-search losses choose the second variable in the selected pair. Main-effect ties still use feature order. The pure-interaction captures establish the tied example, not every interaction configuration.
- Single-class classification input is rejected, matching GUIDE. Bootstrap trees can still omit global classes.

Saved model format remains 1. Loading older models preserves serialized probabilities, predicted classes and routes; corrections apply to new fits. Ensemble trees also finalize probabilities after fitting; exact native ensemble probability conventions remain outstanding.

```sh
python3 tools/capture_reference_core.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --case class-priors --output /tmp/guide-463-class-priors
cargo test --locked --test reference
python3 -m unittest discover -s tools -p 'test_reference_*.py'
```

`tools/reference_cases.py` defines deterministic inputs. Integrity tests discover the corpus, require all catalog cases for both versions, verify pinned executable/file hashes and reproducible inputs, reparse all outputs, and reject malformed tables, inconsistent/disconnected trees and wrong versions. The single-class case requires an explicit native diagnostic and absent tree outputs despite GUIDE returning status zero; a failed fit cannot silently become an accepted rejection.

Next acceptance work: pruning/CV, importance and existing model-based leaves, followed by missing GUIDE families. The enhancement roadmap follows this order before Python, boosting and other extensions.

## Pruning and CV corpus, 2026-10-23

A further **15 cases per executable (30 new captures)** bring the total to **70 captures across 35 distinct cases**. The original 40 constant-node captures remain unchanged. Every new capture contains the raw native outputs, explicit options, deterministic input data, a parsed pruning result and hashes.

| Cases | Checks |
|---|---|
| Least-squares LOOCV at 0.25, 4.6 and 5 SE | All stage means/naive SEs; the 4.6-SE boundary retains two leaves, while 5 SE selects the root |
| Weighted least-squares LOOCV stump | Explicit weighted-error input, verified effective depth/minimum size; native error units converted by mean weight |
| Two four-region least-squares LOOCV paths | Every stage and final tree; the three-leaf intermediate means/SEs are independently verified as intentional native differences |
| Balanced/imbalanced classification LOOCV | Full-sample priors frozen in CV folds; empirical held-out costs and naive SE |
| Classification LOOCV with explicit priors or asymmetric costs | Distinguish prior-adjusted fitting from actual held-out cost averaging |
| Four-class LOOCV | Balanced numerical cutpoint ties, full tree/root stages, predictions and probabilities |
| Fixed-size regression with tied sibling alphas | Retain a three-leaf stage by pruning tied branches individually |
| Fixed-size classification with a root tie | A parent can be the weakest link and collapse directly to one leaf |
| Fixed-size regression/classification pure interactions | Whole-branch pruning can undershoot the requested size |

LOOCV makes held-out partitions independent of the different Rust/Fortran random streams. CV table tolerances are half the final printed unit for each statistic, derived before comparison. Fixed-size reports supply the full native alpha path; Rust tests fit every reported size, checking its leaf count and alpha. They also compare all final splits/routes, row assignments, predictions and probabilities.

New fits use **`guide-rs-0.1-experimental-reference3`**. Corrections include one-at-a-time tied pruning, balanced numerical classification split ties, frozen full-training CV priors, empirical classification held-out losses, and population-based naive SEs. The initial nontrivial candidate evaluates the full fold tree; the final candidate evaluates fold roots. Weight/loss scaling and normalized weight units are preserved. Stored models and their original reports are not refitted on load.

The native CV routine has a demonstrated node-number selection flaw. Rust keeps correct weakest-link selection, as explicitly chosen for this reimplementation. [The evidence and independent counterexamples](native-cv-difference.md) explain the mismatch; the retained [debugger observation](../tests/fixtures/reference/diagnostics/cv-node-order-46.3.json) pins the executable and input hashes. Tests verify the explained native results separately from the correct Rust results. They do not suppress arbitrary CV discrepancies.

The capture parser now rejects effective depth/minimum-size drift, wrong CV fold counts, invalid or incomplete pruning tables, inconsistent selected rows and malformed fixed-size paths. Early draft captures were initially retained locally (removed in the 2026-10-24 cleanup); a missing weighted-error answer had shifted one draft's effective depth to 15. Corrected captures were regenerated into new destinations after adding the settings checks.

```sh
python3 tools/capture_reference_core.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --case ls-loocv-se-boundary --output /tmp/guide-463-loocv-se-boundary
cargo test --locked --test reference
python3 -m unittest discover -s tools -p 'test_reference_*.py'
```

This validates the listed constant-leaf fixed-size and LOOCV cases. Random multi-fold equivalence, native test-sample pruning, bootstrap/median CV variants, model-based pruning and broader weighted multilevel behavior remain unverified. Importance and existing model-based leaves are the next native acceptance priorities.

## Association and importance update, 2026-10-23

The fixed corpus now contains **86 captures**: the preceding 70 fitting/pruning captures and **16 importance reports** from eight cases under each executable. Classification and least squares each cover duplicate predictors, numerical missingness, competing continuous/categorical predictors, and a pure interaction with an identical predictor copy. All use 80 rows, one split level, minimum child size two, exhaustive cuts and 300 native permutations.

`tools/capture_reference_importance.py` preserves version/hash provenance, full reports and normalized score tables, explicit inputs/options, and stderr. The parser rejects incomplete runs, incorrect settings, malformed or unordered scores, unknown/duplicate names, mismatched file/report rows, and invalid or missing normalized thresholds. Integrity tests regenerate every input and verify all artifact/executable hashes. Failed captures are diagnostics and existing paths cannot be overwritten.

The comparison is deliberately scoped: Rust's 95%/99% important-variable sets and identifiable rank relations agree on these strong-signal examples. Raw scores receive independent Pearson/R checks, because native output supplies normalized scores only. Neither exact Monte Carlo-adjusted magnitudes nor fallback normalization is certified. The duplicate design preserves an observed normalization difference (native scores about 12.58/12.22; Rust midpoint-normalized scores 2/2). Full details, interfaces and reproduction commands are in [importance validation](importance-validation.md).

The interaction-copy cases exposed a Rust omission relative to the 42.5 changelog: only the selected pair received the winning interaction score. Exact typed copies now receive it too. Native reports mark all three relevant predictors important, and independent raw-score checks ensure the copy is no longer given zero.

Separately, new stable log tails correct a 6,000-row selection/importance underflow failure. Local exploratory captures, made before cleanup, confirmed both executables select the stronger variable for both regression and classification. They supplement, rather than enlarge, the fixed CI corpus. Newly generated Rust models and importance reports record revision `guide-rs-0.1-experimental-reference4`.

The next native acceptance priority is existing model-based leaves. Importance calibration, native normalization/boundary conventions, 50% thresholds and broader missing/periodic/specialized-response behavior remain open.


## Linear/stepwise leaf acceptance, 2026-10-23

Thirty new captures cover 15 root-only numerical least-squares cases under each executable. The fixed corpus now contains **116 captures across 58 distinct cases**. Both native versions agree on all new parsed results, excluding banner/build metadata; the earlier 86 captures are unchanged.

Twelve cases per version agree with Rust on identifiable coefficients and training predictions. Three explicitly preserve differences in small-sample multiple-linear fallback, collinear stepwise fallback and missing-data stepwise selection. Generated native coefficient/R exports are cross-checked against separate native predictions; independent R `lm` checks validate Rust estimation given its selected columns. No full stepwise, model-based splitting/pruning or truncation equivalence is claimed.

The extreme-tail case first reproduced a Rust error: a one-variable stepwise fit selected the weaker first predictor after both ordinary F probabilities rounded to zero. Log F tails and log Bonferroni comparisons correct the failure in either column order, matching both native builds. New models record `guide-rs-0.1-experimental-reference5`. Full scope, remaining differences and reproduction commands are in [linear leaf validation](linear-validation.md). The user subsequently chose to retain experimental opt-in stepwise and defer further stepwise parity work. The next priority is multiple-linear split selection, pruning and fitting fallbacks; the decision is recorded in `linear-validation.md`.


## Multiple-linear growth/LOOCV after stepwise deferral, 2026-10-23

By user decision, stepwise remains an experimental, opt-in compatibility option; further stepwise parity work is deferred. Existing checks and known differences are retained. Work continues on multiple-linear models.

Five new cases per executable cover numerical split-only and split-and-fit roles, unequal fitting weights, and unweighted LOOCV selecting a stump or the root. Both native versions agree on all parsed results. Rust checks splits/routes, leaf assignments, original-unit coefficients, predictions, complete CV means/SEs and selected trees. An independent R check reproduces leaf estimation and exhaustively enumerates valid cuts in every held-out fold.

The corpus now contains **126 captures across 63 distinct cases**. Earlier captures are unchanged; no runtime behavior change was needed, so the algorithm revision remains reference5. See [the linear validation record](linear-validation.md#multiple-linear-tree-growth-and-loocv-2026-10-23) for settings, tolerances, capture commands and remaining scope. Multilevel, weighted/multifold pruning, categorical fits, truncation and multiple-linear fallback rules remain open.


## Multilevel linear acceptance and observed-rank correction, 2026-10-24

Twelve new cases per executable add **24 captures**, bringing the fixed corpus to **150 captures across 75 distinct cases**. Both native versions agree on every new parsed result. Earlier captures are unchanged. Cases cover depth-2/3 growth, split-and-fit roles, unequal weights, missing fitting/split values, fixed-size pruning paths and weighted/unweighted LOOCV.

The missing-split cases reproduced a Rust candidate-constraint bug: routed missing rows were counted toward numerical cut rank bounds. Numerical candidates now require observed counts at least the configured minimum on the left and strictly greater on the right, matching both native versions' asymmetric endpoint convention. Independent objective enumeration and additional local forward/reversed boundary probes establish the constraint; separate missingness-only candidates remain available. New fits/reports use **`guide-rs-0.1-experimental-reference6`**; saved-model format remains 1.

Two multilevel linear CV cases also demonstrate an actual final-size difference under the known native scan flaw. Independent weighted QR/fold enumeration supports Rust's four-leaf selection; native's faulty scan chooses three. A simulated scan reproduces both native CV tables within printed precision. Tests separately check Rust's correct CV result and the native final three-leaf tree, retaining all structural/coefficient/prediction checks. Independent R evidence pins script/input hashes and every held-out prediction; regular tests also check all fixed-size pruning stages against independent R routes/predictions.

See [linear validation](linear-validation.md#multilevel-multiple-linear-growth-and-pruning-2026-10-24) and [the CV difference](native-cv-difference.md#multiple-linear-counterexamples-2026-10-24) for settings, tolerances and reproduction. Next priorities are multiple-linear small-sample fallbacks and prediction truncation, followed by categorical fitting and broader pruning/default combinations. Stepwise parity remains deferred.


## Numeric multiple-linear policy and prediction limits, 2026-10-24

Forty-seven new cases under each executable add **94 captures**, bringing the fixed corpus to **244 captures across 122 distinct cases**. Both native versions agree on all new parsed results. The preceding 150 captures are unchanged. New captures generate no R prediction export; missing-response rows provide actual native out-of-sample predictions.

For numeric multiple-linear LS, the verified constant fallback is `n < p+7` without global missing flags, or `n < p+m+3` with `m>0`. Declared columns, including constant/collinear ones, determine `p`; global flags including excluded rows determine `m`; active node row count determines `n`. Boundary cases span 1/2/3/4/7 predictors, weights, missingness, all-missing columns and exclusions. The earlier four-row/two-predictor difference is now resolved; its historical fixture label remains unchanged.

Prediction limits support `none`, `node_range`, `node_range_expanded` and `global_range`. Rust retains `none` as its compatible default; native defaults to global range. The native protocol fixes global limits from full active training data across CV, while node limits are fold-local. Training residual signs/search/weakest-link risks use unbounded fits, and returned predictions/held-out losses use the limits. Saved bounds are optional, preserving old JSON and its unbounded predictions. New fits/reports record **`guide-rs-0.1-experimental-reference7`**; saved format remains 1.

The higher-dimensional extension exposed a distinct native LS failure: with nine rows, four predictors and two missing flags, native SSE is 3.25 versus independent QR SSE 2.70. Its `x1.NA` normal equation has residual sum −0.5. Rust keeps the QR optimum. [Pinned independent evidence](../tests/fixtures/linear-policy/diagnostics/native-qr-difference.json) preserves both native manifests/predictions and the independently recomputed solution and normal equations. This is an explicit correctness exception, not a relaxed numeric tolerance.

[Linear validation](linear-validation.md#numeric-fallback-policy-and-prediction-truncation-2026-10-24) documents settings, tolerances, capture commands, default/CV conventions and compatibility. The next priority is categorical fitting/dummy conventions and their fallback policies, then broader model-based pruning/default combinations. Stepwise remains deferred; R export remains outside product scope.

## Categorical multiple-linear acceptance, 2026-10-24

Forty-one new cases under each pinned executable add **82 captures**, bringing the corpus to **326 captures across 163 distinct cases**. Both versions agree on all parsed results in this family, excluding version/build metadata. The preceding 244 captures remain unchanged.

The new `tests/fixtures/categorical-linear` family covers global dummy construction, references absent locally, missing/excluded/zero-weight levels, text ordering, mixed numeric missingness, weighted/duplicate factors, sample guards, prediction limits and LOOCV. It also captures a categorical split-and-fit case exposing the response-mean-order search error and its exhaustive-subset correction. One separate balanced-residual-sign case remains an explicit selection compatibility gap: native one leaf versus Rust two leaves. It is not classified as a native bug. Native root predictions are compared to a separately fitted Rust root; the exact tree/risk difference is independently checked.

`tools/capture_reference_categorical_linear.py` captures reports, coefficients, predictions, split structure and full hashed inputs without generating R prediction code. The parser checks row membership, settings and complete connected trees. `tools/categorical_linear_oracle.R` independently validates weighted estimation on fixed partitions, four-category subset risks and LOOCV. Existing output cannot be overwritten. Old categorical JSON preserves its stored design and predictions; new fits record **`guide-rs-0.1-experimental-reference8`**.

See [categorical validation](categorical-linear-validation.md) for exact conventions, declared tolerances, reproduction and limitations. Next resolve model-based selection gating, then extend defaults/family combinations. Stepwise parity stays deferred and R prediction export stays excluded.

## Singleton residual-sign selection acceptance, 2026-10-24

Thirty-six new cases per executable add **72 captures**, bringing the fixed acceptance corpus to **398 captures across 199 distinct cases**. Both native versions agree on all new parsed results, excluding version/build metadata. The preceding 326 captures remain unchanged. New fits/reports record **`guide-rs-0.1-experimental-reference9`**.

The original categorical selection difference is resolved: a sole eligible nonconstant split predictor with exactly zero residual-sign association does not trigger search in multiple-linear SSE fitting. Positive scores with p-values above 0.99 still trigger ordinary search, ruling out a conventional significance cutoff. Controls cover numeric/categorical split/both roles, cardinality, weak/strong signals, weights, missingness, constant columns, row order, interaction switches, competing predictors and pure interactions. Earlier categorical structural tests now pass without an exception.

`tools/capture_reference_selection.py` retains the usual complete hashed inputs and native outputs without prediction-code export, additionally validating the interaction switch against the report. `tools/selection_oracle.R` independently verifies root sign tables/log tails and weighted terminal fits. A saved pre-change reference8 model, pinned with its input hashes, confirms that loading old trees preserves their predictions.

The new selection suite compares decisions, diagnostic tables, fitted predictions and round trips; it does not certify every route or coefficient basis. Broader multi-predictor zero-score/zero-gain behavior and missing-category child orientation/defaults remain open, with the latter difference explicitly asserted in the new suite. See [selection validation](selection-validation.md) for tolerances and reproduction. Correct pruning/QR remain unchanged; stepwise parity stays deferred and R prediction export stays excluded.

## Categorical orientation and unseen-route acceptance, 2026-10-24

Thirty-nine new cases per executable add **78 captures**, bringing the fixed acceptance corpus to **476 captures across 238 distinct cases**. Both versions agree on all new parsed results. The preceding 398 captures remain unchanged. New fits/reports record **`guide-rs-0.1-experimental-reference10`**; stored old trees retain their saved routes and predictions.

Bounded exhaustive multiple-linear SSE categorical search now anchors the first local lexical level on the left, including missingness sorted as `NA`. Complete/missing complementary partitions and missing-only splits now match native orientation and actual held-out unknown-level predictions. Checks cover split/both roles, label ordering, weights, row order and child-local level sets. Full structural/terminal/prediction comparisons pass for all new cases, with a named independently verified CV-table difference.

The new 96-fold categorical CV control has native mean 94.72 versus independent Rust/R 40.2845495213 for the two-leaf candidate. Every direct leave-one-out refit under both native versions agrees with the independent prediction. The native root score and selected final tree agree. Correct CV is retained; the mechanism of this additional discrepancy is unproved. A hashed diagnostic record retains all independent values and all 192 auxiliary native refit outcomes separately from the full-artifact acceptance captures.

See [orientation validation](orientation-validation.md) for the generating scripts, exact scope, tolerances and reproduction. Next investigate multi-predictor zero-score/zero-gain selection. Stepwise parity remains deferred and R prediction export remains excluded.

## Zero-gain splits and same-class leaf merging, 2026-10-25

Six new constant-node cases per executable add **12 captures**, bringing the fixed acceptance corpus to **488 captures across 244 distinct cases**. Both versions agree on every parsed result; the preceding 476 captures are unchanged. New fits record **`guide-rs-0.1-experimental-reference11`**.

The cases join the core catalog in `tools/reference_cases.py` and are captured with `tools/capture_reference_core.py`, so the existing integrity tests and `guide_46x_constant_core_corpus` cover them. `ls-zero-gain-{numeric,categorical}` verify that GUIDE splits a selected variable whose best cut leaves both child means equal. `class-same-class-{stump,cascade}` verify that sibling leaves predicting the same class merge bottom-up even under strong association; `class-same-class-{kept,costs}` are controls where a lower split, or unequal costs, change a prediction and the tree survives. Eight of the twelve captures fail on the previous revision; see [selection validation](selection-validation.md#multi-predictor-zero-scores-zero-gain-and-ties-2026-10-25).

## Missing-value flags and categorical orientation, 2026-10-25

Twenty missing-flag cases (`tests/fixtures/missing-flag`, captured by `tools/capture_reference_missing_flag.py`) and eight categorical orientation cases in the core catalog add **56 captures** across both executables, bringing the fixed acceptance corpus to **544 captures across 272 distinct cases**. Both versions agree on every parsed result; earlier captures are unchanged. New fits record **`guide-rs-0.1-experimental-reference12`**. See [flag validation](missing-flag-validation.md) for semantics, the documented native difference and the held-out cases.
