# Categorical multiple-linear validation

The 2026-10-24 milestone adds **41 cases under each pinned executable (82 captures)**, bringing the fixed corpus to **326 captures across 163 distinct cases**. GUIDE 46.3 and 46.2 agree on every parsed result in this family after removing version/build metadata. Earlier captures remain unchanged. New fits record **`guide-rs-0.1-experimental-reference8`**; saved-model format remains 1.

This covers categorical and mixed numeric/categorical **multiple-linear squared-error** designs. It does not establish the same conventions for stepwise, quantile, Poisson, logistic, periodic or associated M-flag models. Stepwise parity remains deferred; R prediction export remains outside product scope.

## Global levels and local references

Native dummy construction uses the entire input file, including rows excluded because their response is missing or their weight is zero. Levels sort lexically, and categorical missingness participates as the canonical DSC token `NA`. The first global level is omitted. Thus missingness sorts before lowercase `a`, but after uppercase `A` and numeric-looking strings such as `01`. Numeric-looking labels retain their text identity.

When the global reference occurs in a leaf, it supplies the intercept's categorical baseline. When it is absent, the observed native coefficients retain the earlier local dummies and set the last local dummy to zero. This is consistent with sequential alias removal against the intercept; it is an inference from coefficient/prediction probes, not a debugger observation. Rust represents the same convention by explicitly rebasing that feature to the last locally observed level. Local reference selection uses active rows, while the global level list remains fixed across leaves and CV folds.

For example, active categories `a,b,c` with responses `2,8,14` normally give unseen-category prediction 2. Adding an excluded row with missing category introduces a global `NA` reference that is absent among active rows. The local reference becomes `c`, and unseen-category prediction becomes 14. If missing-category rows actually occur with response 20, the reference is present and unseen-category prediction becomes 20. Future calls to `predict` never expand the stored design.

Missing and unseen categories activate no ordinary category dummy. A missing-indicator term is created when missingness is a locally observed nonreference level. This preserves GUIDE's observed predictions without redundant columns. Two child-reference cases check the convention with and without numeric missingness. The typed Rust API also preserves a distinction between literal text `NA` and missingness; the native fixtures cannot represent both using their chosen DSC missing token. Alternate missing tokens and multiple missing codes are unverified.

## Sample guard and rank

The numeric fallback policy extends to mixed designs. Define `p` as the number of declared numeric fitting columns **plus** the sum of `global_level_count - 1` for categorical fitting columns. Include categorical missingness in that level count. Define `m` as the number of numeric fitting columns with missing values anywhere in the raw input. Below `p+7` active rows when `m=0`, or below `p+m+3` when `m>0`, fit a weighted constant.

Categorical missingness adds a dummy; it does not invoke the numeric-missing formula. Excluded levels and zero-weight levels affect `p`. Weight magnitudes do not multiply the active-row count. Constant and collinear declared columns still affect the nominal guard; weighted pivoted QR handles estimable rank after it. Entirely missing or constant categorical predictors can still yield a constant when no term is estimable.

Twenty-two boundary cases check both sides of the guard for two, three and four categories, categorical missingness, excluded/zero-weight levels and missingness, mixed numeric designs and numeric missing flags. Other cases cover ordering, weighted two-factor designs, duplicate factors, constants, entirely missing factors, local references, four prediction-limit modes and LOOCV. Duplicate-factor agreement concerns the observed design; arbitrary inconsistent combinations of aliased predictors remain unverified.

## Categorical split search correction

Ordering categories by response mean can omit the best partition when child linear slopes differ. Multiple-linear squared-error searches now enumerate all distinct nonempty categorical partitions through **11 observed nonmissing levels**, retaining one orientation per complementary pair. Since reference10, the [orientation follow-up](orientation-validation.md) includes missingness as the canonical `NA` level in this bounded enumeration, fixing the first locally observed level on the left. Larger designs retain the existing ordered heuristic; native high-cardinality model-based search remains unverified. Constant regression and other model families retain their existing search rules.

The strong-signal four-category fixture exposes the error: response-mean ordering interleaves the two slope groups, while the best split is `{a,b}` versus `{c,d}`. Both native versions and corrected Rust select that partition. Independent R weighted least squares enumerates all seven partitions and verifies the minimum. This establishes the captured case, not every native categorical search/tie convention.

## Selection difference resolved in reference9

`categorical-linear-split-and-fit` has four equally sized categories and opposite slopes, with exactly 12 positive and 12 negative root residuals per category. At the reference8 milestone, GUIDE returned one leaf while Rust searched a beneficial two-leaf split. The common root fit has SSE **28750**, or mean squared error **299.4791666667**; the hypothetical two-leaf fit is essentially perfect.

The subsequent [selection milestone](selection-validation.md) resolves this difference: a sole eligible predictor with exactly zero residual-sign association does not authorize search. Positive scores with p-values above 0.99 still do. Current Rust matches the native one-leaf result directly, without a root-only refit or structural exception. This was a selection compatibility gap, not a native numerical/pruning defect. The subsequent [orientation milestone](orientation-validation.md) resolves complementary child directions and missing/unseen routes, with an additional independently checked native categorical CV discrepancy. Multi-predictor zero-score fallback remains open.

## Evidence and reproduction

`tests/fixtures/categorical-linear/guide-{46.3,46.2}` retain input, DSC/data/CSV, dataset/options, reports, splits, coefficients, predictions, stdout/stderr, executable/source metadata and artifact hashes. Captures skip native R export, refuse overwrite and retain failures as diagnostics. The parser verifies settings, row membership, missing responses, active counts and connected terminal assignments. It accepts categorical split encodings that retain `b` as well as the captured `c`, preserving the raw fields.

Regular Rust tests now compare full splits, routes, predictions and JSON round trips for all 41 cases per version, with no selection exception. Native prediction tolerance is half the final printed unit plus `1e-9`; LOOCV tables use their printed tolerances. Independent R `lm.wfit` checks all 41 cases at relative tolerance `2e-10`, plus all seven partitions in each four-category control and both LOOCV candidates. It uses frozen partitions to isolate estimation; it does not independently certify the variable-selection algorithm.

Old categorical JSON retains its saved terms and unseen/missing predictions without refitting. This revision changes newly fitted designs only. Correct QR and correct weakest-link pruning remain in place with their earlier native counterexamples.

```sh
python3 tools/capture_reference_categorical_linear.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --case categorical-linear-local-reference --output /tmp/guide-463-categorical
cargo test --locked --test categorical_linear
cargo test --locked --test reference guide_46x_categorical
python3 -m unittest discover -s tools -p 'test_reference_categorical_linear.py'
cargo test --locked --test interoperability categorical_linear_predictions -- --ignored
```
