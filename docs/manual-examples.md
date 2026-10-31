# GUIDE manual examples

This record compares guide-rs with native GUIDE 46.3 on the examples in the GUIDE 46.1 manual that guide-rs can express. Unlike the acceptance corpus, these examples use the manual's real datasets: larger samples, many missing values, flags and wide feature sets. They are a broad check rather than an acceptance gate, and the mismatches below are open work, not accepted differences.

## Method

`tools/compare_manual_examples.py` runs each batch input in `tests/fixtures/manual-examples/` (see its README for provenance) under native GUIDE, translates it into a guide-rs configuration, fits and predicts on the same data, and counts the training rows whose predictions agree (class labels exactly; numbers to a relative 1e-4).

- **Unpruned** mode rewrites each batch file to grow the tree without pruning. Growth is deterministic, so this mode is an exact comparison.
- **Pruned** mode keeps the manual's cross-validation. GUIDE draws its folds from its own random stream, so these results show broad agreement only (see [native CV differences](native-cv-difference.md)).

Translation details:

- GUIDE's defaults for depth and minimum node size come from the native report.
- `min_gain` is 0.
- Interaction tests are off when the report says native skipped them.
- Poisson fitted values are compared as rates (counts divided by exposure).
- Prediction truncation is off for stepwise and polynomial leaves, since guide-rs truncates only multiple-linear predictions.

## Results

Recorded 2026-10-31 on `main` at revision `guide-rs-0.1-experimental-reference24` (PR #61). Each row shows training rows agreeing, then terminal nodes as native / guide-rs.

| Example | Manual | Unpruned rows | Unpruned leaves | Pruned rows | Pruned leaves |
|---|---|---|---|---|---|
| `fish-constant` | 4.1 | 154/159 | 10 / 15 | 154/159 | 8 / 11 |
| `fish-lda` | 4.2 | 159/159 | 1 / 1 | 159/159 | 1 / 1 |
| `ce-classification` | 5.1 | 3137/3965 | 63 / 63 | 3219/3965 | 20 / 46 |
| `ce-constant` | 6.1 | 437/2487 | 65 / 56 | 2396/2487 | 9 / 8 |
| `ce-polynomial` | 6.2 | 8/2487 | 37 / 16 | 0/2487 | 1 / 7 |
| `ce-stepwise` | 6.3 | 325/2487 | 1 / 1 | not completed | 1 / - |
| `ce-quantile` | 7.1 | 53/2487 | 49 / 62 | 65/2487 | 24 / 18 |
| `ce-quantile-linear` | 7.2 | 79/2487 | 33 / 17 | 32/2487 | 10 / 9 |
| `ce-two-quantiles` | 7.3 | 11/2487 | 72 / 88 | 0/2487 | 4 / 8 |
| `nhtsa-polynomial` | 8.1 | 1/3276 | 73 / 46 | 0/3276 | 10 / 1 |
| `solder-poisson` | 9.1 | 157/720 | 116 / 228 | 471/720 | 28 / 39 |
| `solder-poisson-linear` | 9.2 | 12/720 | 56 / 60 | 720/720 | 3 / 3 |
| `lungcancer-poisson-offset` | 9.3 | 27/920 | 56 / 93 | 920/920 | 3 / 3 |

The pruned `ce-stepwise` fit was stopped after 39 minutes. The full tree is a single leaf, but each of the ten CV folds grows a stepwise tree, and that is slow on 2487 rows with 383 numeric, 48 categorical and 116 flag variables.

## Known causes

These are tracked in the [coverage checklist](coverage.md):

- **Sparse numeric variables.** Native groups the observed values of mostly missing variables differently in its selection tests. This is the main source of unpruned differences on the CE examples. Native's chi-square statistics match contingency tables whose columns are contiguous groups of observed values plus a missing column, with degrees of freedom one less than the number of nonempty columns. Some variables instead match tables where the missing rows join an observed group, and a few match no grouping tried. The grouping rule is not yet known.
- **Polynomial and linear splits with missing regressors** (`ce-polynomial`, `nhtsa-polynomial`, the linear quantile and Poisson examples).
- **Stepwise leaves.** Further stepwise parity is deferred: the root model already differs (325/2487 rows agree with a single leaf on both sides).

Examples omitted as unsupported: censored response, multiresponse, longitudinal, treatment and propensity models, DIF scoring, and logistic regression with an E variable.

## Reproducing

```sh
cargo build --locked --release
python3 tools/compare_manual_examples.py --binary .reference/upstream/macos-arm64/guide --mode both
```

The first run downloads `datafiles.zip` (SHA-256 `8e734235...4e72`) into `.reference/manual-data/`. `--timeout` limits each native or guide-rs run (default 1800 seconds), and `--json` writes the results to a file.
