# Residual-sign selection validation

The 2026-10-24 follow-up resolves the one-leaf versus two-leaf difference in `categorical-linear-split-and-fit`. New fits record **`guide-rs-0.1-experimental-reference9`**. Thirty-six cases under each pinned GUIDE executable add **72 captures**, bringing the fixed acceptance corpus to **398 captures across 199 distinct cases**. Both native versions agree on every parsed result in this family. Earlier captures remain unchanged.

## Verified rule

For **multiple-linear squared-error fitting**, when there is exactly one eligible nonconstant split predictor and its residual-sign Pearson statistic is **exactly zero**, Rust now stops before searching its cuts. The diagnostic remains available with statistic zero and p-value one. Eligibility is determined after constant discretized predictors are removed, so adding a constant split-only column does not bypass the rule.

This is a narrowly verified singleton rule, not a significance-level stopping criterion. Positive scores can authorize ordinary search even when their p-values greatly exceed the configured Bonferroni alpha. In four-category controls, each category contains the same number of rows, and the two sign distributions are reversed in the two slope groups:

| Rows per category | Positive/negative counts in opposite groups | Pearson statistic, 3 df | Native result |
|---|---|---:|---|
| 24 | 12/12 versus 12/12 | 0 | One leaf |
| 24 | 11/13 versus 13/11 | 0.6666667 | Two leaves, p approximately 0.88 |
| 240 | 119/121 versus 121/119 | 0.0666667 | Two leaves, p greater than 0.99 |

The code tests the statistic directly, without a p-value cutoff, rounding or epsilon. The existing significance level still controls escalation to interaction/projection tests; this change does not turn it into a universal growth threshold.

For the original case, both native versions and current Rust retain the common root fit with SSE **28750**. The earlier regression test no longer substitutes a separately fitted root or accepts a structure exception. Its full split/route/prediction and JSON round-trip comparisons now pass normally. Independent R still demonstrates that a hypothetical two-leaf partition can fit essentially perfectly; lower training error alone does not authorize that search under the verified rule.

## Controls and limits

The 36 cases cover categorical and numeric predictors with `split`/`both` roles, two and four observed levels, exactly zero/weak/strong associations, missing categorical values, unequal weights, reversed row order, irrelevant constant columns, interaction tests enabled/disabled and a zero-score rival to a positive-score predictor. Two pure interaction cases have zero main-effect scores but a strong joint score and retain their four-leaf trees. The singleton guard runs only in fitting mode; importance scoring, stepwise and other response/model families retain their existing behavior. Correct QR and weakest-link pruning are unchanged.

Do not generalize this to “all main scores zero means stop.” Exploratory multi-predictor controls show native continuation even with interaction testing disabled, including zero-gain first splits. Native multi-predictor zero-score fallback/tie rules and zero-gain growth remain unverified. The implementation deliberately leaves that path unchanged rather than imposing an unsupported global gate. No native source-level explanation is claimed from the black-box observations.

The new selection suite compares decision outcomes (leaf counts), residual diagnostics, fitted predictions on the recorded rows and JSON round trips. It does not certify every categorical partition orientation, unseen-level route or native coefficient basis. At this milestone, the weak missing-category control exposed complementary native/Rust child orientations. The subsequent [orientation milestone](orientation-validation.md) resolves that difference and checks actual unknown-level predictions; the selection test now requires the corrected direction. The prior 41-case categorical family continues to require its full structural comparisons, now with no selection exception.

## Evidence and independent checks

`tests/fixtures/selection/guide-{46.3,46.2}` retain complete inputs, dataset/options, DSC/data/CSV, native reports, coefficients, splits, predictions, stdout/stderr, executable/source metadata and file hashes. Captures explicitly request an intercept, no truncation/pruning, exhaustive numerical cuts, minimum node size two and depth one (two for the interaction controls). The parser verifies the interaction switch in the native report as well as the existing fitting/settings/row-membership/completion checks. Captures refuse overwrite and preserve failed attempts as incomplete diagnostics. No prediction-code export is generated.

`tools/selection_oracle.R` independently fits the root with base R `lm`, constructs unweighted residual-sign tables, computes Pearson statistics/degrees of freedom/log tails, checks joint interaction tables and fits the frozen terminal partitions by weighted least squares. Rust comparisons use relative tolerance `2e-10`. These designs have at most four distinct values per split feature, so numeric grouping produces the same active table columns as their distinct-value groups. Python separately verifies zero versus positive tables and evaluates the chi-square(3) tail analytically for the near-one controls. Native prediction comparisons retain half the final printed unit plus `1e-9`.

`diagnostics/legacy-reference8.json` is the actual two-leaf Rust model fitted before this change. Its provenance pins the model and original input hashes. A regression test loads it without refitting and verifies its original near-perfect predictions while a new fit produces one leaf. Saved-model format remains 1; loading old stored trees does not apply the new selection rule.

```sh
python3 tools/capture_reference_selection.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --case selection-categorical-near-one --output /tmp/guide-463-selection
cargo test --locked --test selection
cargo test --locked --test reference guide_46x_categorical
python3 -m unittest discover -s tools -p 'test_reference_selection.py'
cargo test --locked --test interoperability selection_sign_tables -- --ignored
```

The [orientation follow-up](orientation-validation.md) validates missing/unseen routes and records a separate categorical CV discrepancy. Next validate multi-predictor zero-score/zero-gain behavior. Stepwise parity remains deferred; R prediction export remains excluded.
