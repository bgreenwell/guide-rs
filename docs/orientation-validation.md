# Categorical orientation and unseen-level validation

The 2026-10-24 milestone adds **39 cases under each pinned GUIDE executable (78 captures)**, bringing the fixed acceptance corpus to **476 captures across 238 distinct cases**. Both native versions agree on every parsed result in the new family. Earlier captures remain unchanged. New fits record **`guide-rs-0.1-experimental-reference10`**; saved-model format remains 1.

## Verified orientation

In the bounded exhaustive search for **multiple-linear squared-error categorical splits**, the left child contains the **first locally observed category in lexical order**. Missingness participates as the canonical token `NA`. Only active rows in that node define the ordering; excluded responses, zero-weight rows and levels observed elsewhere do not anchor the local split. Such rows can still affect the global fitting design as documented in the earlier categorical milestone.

Complementary partitions have the same fitting objective, but their orientations affect predictions because unknown levels go right. The previous enumeration omitted the last category from the left set. That could reverse the native orientation even for complete data, such as `{a,d}` versus `{b,c}`. With lowercase categories and missingness, native can put `{NA,c}` on the left while Rust previously put `{a,b}` there.

The corrected enumeration fixes the first local level on the left and enumerates every assignment of the remaining levels, excluding the full set. Missingness is an ordinary level in this enumeration. Missing-only partitions use categorical split metadata too, so their orientation and unknown route follow the same rule. For example, with only `a` and missingness, `NA` sorts first and goes left; with `A` and missingness, `A` goes left instead. Unknown levels go right in both cases.

The bounded path still applies through **11 observed nonmissing levels**, with a possible additional missing level. Larger designs retain the existing ordered heuristic. Constant regression, classification, stepwise and other losses retain their existing search paths. These checks do not establish all high-cardinality searches, alternate missing tokens, arbitrary ties or other-family orientation rules.

## Actual held-out predictions

The new cases cover all three complementary two-versus-two partitions of four categories, split-only and split-and-fit roles, lowercase/uppercase and numeric-looking labels, reordered labels, missing-only splits, weights, reversed row order and depth-two local reference changes. Prediction rows have missing responses and include an excluded label that sorts before all active categories (`!future`), a later unseen label, missingness and known levels. Their native training/exclusion flags are checked.

For the earlier `selection-missing-k11` example, future category `future` at `x=23` changes from **−17.16** under the old orientation to **+17.16**, matching native GUIDE. Predictions on the observed training rows stay equal because the two partitions are complements. All 39 cases now compare exact split orientation, missing/unknown routes, terminal assignments, predictions and serialized round trips. The one CV-table discrepancy below has a named independent check rather than a relaxed tolerance.

An actual pre-change model is frozen in `diagnostics/legacy-reference9.json`, with its model and input hashes. Loading it retains its stored orientation and −17.16 prediction; refitting uses the corrected orientation. Old models are not silently rewritten.

## Categorical LOOCV discrepancy

`orientation-loocv` uses 96 observations, four categories including missingness, a fitting numeric predictor, depth one, no truncation and 96-fold CV. Both native versions and Rust select the same two-leaf final tree. Their reported CV tables differ:

| Candidate | Independent exhaustive R / Rust mean | Independent naive SE | Native mean | Native naive SE |
|---|---:|---:|---:|---:|
| Two leaves | 40.2845495213 | 4.3957332178 | 94.72 | 6.616 |
| Root | 159.2407942530 | 2.7667939725 | 159.2 | 2.767 |

The root values agree within native printed precision. The two-leaf difference is too large to be rounding. The independent R calculation enumerates all seven categorical partitions separately in every training fold and fits them by weighted least squares.

As an additional check, **all 96 direct leave-one-out refits were executed under each native version** with the held row's response marked missing and pruning disabled. Every native direct refit produces the same two-leaf partition and agrees with the independent held-out prediction within its printed tolerance. Averaging the rounded native direct-refit losses gives **40.2845117202**, close to the full-precision independent value and inconsistent with the native CV table's 94.72.

Rust retains the independently verified CV calculation, as requested. This is a new categorical CV discrepancy; no native source-level mechanism is established, and it is not attributed to the earlier debugger-observed multilevel node-order flaw. The final selected size happens to agree here. Both the native table and the correct predictions/scores remain explicitly checked.

`diagnostics/categorical-cv-difference.json` pins the independent script, original native manifests/binaries and every auxiliary refit's input hashes. It retains every independent held-out prediction, raw native prediction token, printed tolerance, leaf count and parsed split. The 192 auxiliary diagnostic runs are separate from the 476 full-artifact acceptance captures; their temporary full reports are not included in the acceptance corpus. Python independently recomputes the means/SEs and checks every native direct-refit prediction. The R interoperability test regenerates every independent value.

## Reproduction and scope

`tools/capture_reference_orientation.py` retains complete hashed input/data/options, reports, split/coefficients files, predictions, stdout/stderr and pinned source/binary metadata. Captures skip prediction-code export, reject overwrites and preserve incomplete attempts. Native prediction comparisons use half the last printed unit plus `1e-9`. Independent R comparisons use relative tolerance `2e-10`. The ordinary test suite requires no GUIDE executable or R installation.

```sh
python3 tools/capture_reference_orientation.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --case orientation-missing-lower-p1-both --output /tmp/guide-463-orientation
python3 tools/capture_orientation_cv_difference.py --output /tmp/guide-categorical-cv.json
cargo test --locked --test orientation
cargo test --locked --test reference guide_46x_categorical_orientation
python3 -m unittest discover -s tools -p 'test_reference_orientation.py'
cargo test --locked --test interoperability categorical_linear_predictions -- --ignored
```

The independent categorical R check now covers the prior 41 datasets plus these 39, including fixed terminal estimation and the exhaustive LOOCV calculation. Multi-predictor zero-score/zero-gain behavior remains the next selection priority. Stepwise parity remains deferred and R prediction export remains excluded.
