# Association tails and importance validation

New fits and importance results record numerical behavior revision **`guide-rs-0.1-experimental-reference4`**. This milestone fixes chi-square tail underflow, propagates interaction scores to identical predictor copies, and makes ambiguous importance groups explicit. It adds 16 native importance captures to the existing 70 fitting/pruning captures. Importance trees are checked for nonfinite statistics before raw scores or permutations are accepted.

## Extreme-tail correctness

Two 6,000-row predictors with Pearson statistics 4,000 and 6,000 both have ordinary p-values equal to zero in double precision. Previously, feature-index tie-breaking selected the weaker predictor when it appeared first. The old chi-square inverse also clamped probabilities to 1e-300, giving both predictors the same transformed statistic, about 1,373.8726.

Selection now compares log survival probabilities, including Bonferroni thresholds calculated directly in log space. The incomplete-gamma upper-tail fraction is combined with its prefactor in log space. Importance converts log probabilities directly to one-degree-of-freedom chi-square quantiles. Ordinary probabilities are retained for display and may correctly round to zero.

Tests verify that both regression and classification select the stronger predictor in either column order, including at a subnormal significance level. Their transformed importance statistics remain 4,000 and 6,000. A second case compares a perfect 1,000-level categorical predictor (statistic 6,000, df 999) with a binary predictor (statistic 4,000, df 1). The binary predictor has the smaller tail probability despite its smaller statistic. This prevents replacing probability comparison with an inappropriate comparison of raw chi-square values. It is a numerical comparison of the implemented tails, not validation of the high-cardinality small-cell approximation.

Additional local captures made before repository cleanup verified that both pinned native versions also choose the stronger predictor for regression and classification on the 6,000-row design. Those four supplementary exploratory captures were removed in the 2026-10-24 cleanup; fixed CI evidence and regression tests remain.

Independent numerical checks include the exact df-2 identity `log P(Chi²₂ > x) = −x/2`, fixed R reference values, and an optional R test with **56 forward log tails and 56 inverse log tails**, covering df 1–100,000 and inverse log probabilities down to −1,000,000. Forward comparisons use tolerance `2e-9 × (1 + |reference|)`; inverses use relative tolerance 3e-8. The inverse at log probability −1,000,000 and df 1 is approximately 1,999,985.039766, beyond the former search cap.

## Native importance captures

`tests/fixtures/importance/guide-46.2/` and `guide-46.3/` each contain eight cases: classification and least squares crossed with four designs.

| Design | Checked behavior |
|---|---|
| Strong binary signal and an identical copy, independent binary noise, constant predictor | Both signal copies are important; noise/constant have zero raw score |
| Numerical signal with missing observations and independent noise | Missingness is a separate association category; signal remains important |
| Competing continuous/categorical predictors | Signal/noise rank relations and raw Pearson statistics 80 and 0.8 with df 3 |
| Pure interaction, an identical copy of one interacting predictor, independent noise | All three relevant predictors receive the winning interaction score |

Every case uses 80 rows, constant nodes, one split level, exhaustive numerical cuts, minimum child size two and 300 native permutations. The capture parser validates the runtime version, effective settings, score-file/report agreement, rank ordering and all five reported normalized thresholds. Omitted zero-score variables are recorded explicitly. Raw reports, score tables, stderr, batch/DSC/data and hashes are retained. The prior 70 captures were not modified.

**Comparison scope:** native files expose normalized scores, not unadjusted scores. Rust tests compare the reported 95%/99% important-variable sets and identifiable rank relations. Raw scores are checked independently from the designs' Pearson statistics and the R-validated tail/quantile transform. Neither this raw check nor matching important sets certifies equality of native raw scores, all adjusted rankings, or Monte Carlo calibration magnitudes.

The native and Rust random streams differ. Native duplicates also received different normalized magnitudes, while Rust's shared response permutations give identical copies identical raw/null-mean/adjusted scores. In the duplicate design, native normalized values are about 12.58 and 12.22, whereas Rust's midpoint normalization gives 2 for both. The native calibration/normalization details responsible remain unverified. These observations are preserved as differences, not labeled a demonstrated native bug.

The changelog's version 42.5 entry explicitly requires extending interaction importance updates to identical predictors. Previously, Rust updated only the selected pair, giving the third copy raw score zero. It now updates exact typed copies within the node as well. The ordinary regression test reproduced this failure before the correction; the frozen native reports identify all three predictors as important.

```sh
python3 tools/capture_reference_importance.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --case ls-importance-interaction-copy --output /tmp/guide-463-importance-copy
cargo test --locked --test importance_reference
cargo test --locked --test interoperability extreme_chi -- --ignored
python3 -m unittest discover -s tools -p 'test_reference_*.py'
```

Captures refuse overwrites; failed/version-mismatched/incomplete runs are diagnostics. Integrity tests verify pinned binary/file hashes, regenerate inputs and reparse outputs. No GUIDE executable is needed to run the frozen comparisons in CI. The rejected draft was removed with other temporary probes during repository cleanup; accepted captures remain frozen.

## Interfaces and interpretation

- Public numerical additions: `log_gamma_q`, `chi_square_log_survival`, and `chi_square_log_upper_quantile`. The ordinary inverse no longer floors small probabilities; zero upper-tail probability returns infinity for positive df, and invalid probabilities return NaN.
- `SelectionDiagnostic` adds optional `log_p_value` and `importance_log_p_value`. Older model JSON can omit them; new models retain finite negative logs when displayed p-values underflow. Saved model format stays 1 and older models are not refitted on load.
- `ImportanceResult` adds optional `algorithm_version`; older reports deserialize with no revision. The fitting options and CLI commands are unchanged.
- Importance groups now include `indeterminate` when a 95% or 99% rank boundary crosses equal adjusted scores. Equal scores are not assigned different groups because of feature-name ordering. `important_count` remains the rank-based target count, not a count of resolved memberships in an ambiguous block.
- A positive raw score with zero estimated null mean also has an indeterminate group and no normalized score. The existing scalar adjusted score remains a zero compatibility placeholder with an explicit warning; its value must not be interpreted as a calibrated zero.

Rust still uses the documented rank/midpoint approximation at 80/90/95/99%, with a missing numerical cutoff at tied/all/none boundaries. Native 50% reporting, fallback normalization, all boundary conventions, broader calibration, missing flags, periodic predictors and specialized responses remain incomplete or unverified. The strong-signal fixtures do not establish selection unbiasedness, power, exact familywise calibration, or full GUIDE importance equivalence.
