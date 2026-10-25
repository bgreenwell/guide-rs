# Root selection bias study

This study checks GUIDE's defining property, unbiased split-variable selection, at the root node. The protocol, seeds and pass criteria below were fixed on 2026-10-25, before any results were collected. Results are appended under [Results](#results) and must not motivate changes to this protocol. Any later revision of the criteria needs a new dated protocol section, with the original kept.

The harness is `tests/selection_bias.rs`:

```sh
cargo test --release --test selection_bias -- --ignored --nocapture
```

## Design

Each dataset has n = 400 rows and five split-only predictors, drawn independently:

| Name | Type | Distribution |
|---|---|---|
| `binary` | numeric | 0/1 with probability 1/2 |
| `cat4` | categorical | 4 equiprobable levels |
| `cat50` | categorical | 50 equiprobable levels |
| `cont` | numeric | standard normal |
| `cont_missing` | numeric | standard normal, missing completely at random with probability 0.3 |

Every configuration fits R = 1000 datasets with `max_depth = 1`, `pruning: none` and otherwise default options. Dataset `r` of configuration `k` uses ChaCha8 seed `20261025 + 1_000_000 k + r`, and that value is also the fit seed.

| k | Configuration | Response |
|---|---|---|
| 1 | Constant least squares | y ~ N(0, 1) |
| 2 | Binary classification | y ~ Bernoulli(1/2) |
| 3 | Three-class classification | y uniform on {0, 1, 2} |
| 4 | Multiple-linear least squares, fit-only covariate z ~ N(0, 1) | y = 1 + 2z + N(0, 1) |
| 5 | Constant least squares, interaction tests off | y ~ N(0, 1) |
| 6 | Constant least squares, weak signal | y = 0.25 `binary` + N(0, 1) |
| 7 | Binary classification, weak signal | P(y = 1) = 0.42 + 0.16 `binary` |

**Counted selection.** For each fit, the counted variable is the root split variable. If the root has no split (no admissible split, or same-class leaves merged), it is the root diagnostic with the smallest log p-value, with ties to the lowest index. Fits with neither are counted as "none".

**Exhaustive baseline.** For configurations 1, 2, 6 and 7, the same datasets are also scored by a CART-style exhaustive search implemented in the test: each variable's best binary split by impurity reduction (sum of squares for regression, Gini for classification). Numeric splits scan all cuts between distinct observed values. Categorical splits order levels by response mean (regression) or class-1 proportion (classification), which is optimal for these two cases. Missing values are sent to whichever side gives the larger reduction. The variable with the largest reduction is selected, with ties to the lowest index.

## Pass criteria

Null configurations (k = 1 to 5):

1. **Primary.** The Pearson goodness-of-fit test of the five selection counts against uniform (4 degrees of freedom) has p > 0.002. This is a 0.01 familywise level, Bonferroni-adjusted over five configurations.
2. **Secondary.** Every predictor's selection frequency lies in [0.14, 0.26], and at most 1% of fits are counted as "none".
3. **Design check.** In configurations 1 and 2, the exhaustive baseline's goodness-of-fit p-value is below 1e-6. This shows the design can detect the bias GUIDE is meant to avoid. If it fails, the study is underpowered and its null results are inconclusive, not a pass.

Weak-signal configurations (k = 6, 7):

4. **Power.** guide-rs selects `binary` in more than 20% of fits, and more often than the exhaustive baseline on the same datasets.

A failed criterion is reported as a failure with its numbers. Do not relax a bound to make it pass; investigate selection instead.

## Results

### First run, 2026-10-25 (algorithm revision `guide-rs-0.1-experimental-reference10`)

The harness gained one change before this run: a configuration whose fits raise an error is reported as not evaluable and counted as a failure, instead of aborting the study. No design, seed or criterion changed.

| k | Configuration | guide-rs frequencies (binary, cat4, cat50, cont, cont_missing) | GOF p | Exhaustive baseline | Verdict |
|---|---|---|---|---|---|
| 1 | Constant least squares | 0.194, 0.207, 0.186, 0.207, 0.206 | 0.767 | cat50 in 100% of fits, p < 1e-300 | Pass |
| 2 | Binary classification | 0.201, 0.192, 0.205, 0.180, 0.222 | 0.301 | cat50 in 100% of fits, p < 1e-300 | Pass |
| 3 | Three-class classification | not evaluable | | | **Fail** |
| 4 | Multiple-linear least squares | 0.219, 0.206, 0.176, 0.204, 0.195 | 0.280 | | Pass |
| 5 | Constant least squares, no interactions | 0.181, 0.186, 0.225, 0.223, 0.185 | 0.046 | | Pass |
| 6 | Least squares, weak binary signal | binary 0.679 | | binary 0.003 | Pass |
| 7 | Classification, weak binary signal | binary 0.927 | | binary 0.012 | Pass |

No fit was counted as "none". Every evaluable null configuration meets the primary and secondary criteria, and the design check passes decisively: exhaustive search selects the 50-level noise categorical in essentially every null and weak-signal fit, while guide-rs selects each predictor about one fifth of the time and finds the weak binary signal in 68% (regression) and 93% (classification) of fits.

Configuration 3 fails as not evaluable. guide-rs rejects multiclass classification with a categorical predictor of more than 11 levels ("GUIDE's LDA/grouped-category search is not implemented"), a capability already listed as missing in [the coverage checklist](coverage.md). The study therefore says nothing yet about multiclass selection bias with high-cardinality predictors. It should be rerun unchanged once that search exists.

This run predates the zero-gain and same-class merging change (`reference11`, PR #9). That change can remove a root split, in which case the counting rule falls back to the smallest p-value, so the study is rerun unchanged after it merges.
