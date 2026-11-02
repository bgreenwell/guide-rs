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

Recorded 2026-11-01 at revision `guide-rs-0.1-experimental-reference26`. Each row shows training rows agreeing, then terminal nodes as native / guide-rs; parenthesized values are from reference25 where they changed.

| Example | Manual | Unpruned rows | Unpruned leaves | Pruned rows | Pruned leaves |
|---|---|---|---|---|---|
| `fish-constant` | 4.1 | 153/159 (154) | 10 / 17 (15) | 153/159 (154) | 8 / 11 |
| `fish-lda` | 4.2 | 159/159 | 1 / 1 | 159/159 | 1 / 1 |
| `ce-classification` | 5.1 | 3179/3965 (3137) | 63 / 57 (63) | 3292/3965 (3219) | 20 / 17 (46) |
| `ce-constant` | 6.1 | 1572/2487 (626) | 65 / 62 (59) | 2487/2487 (2396) | 9 / 9 (8) |
| `ce-polynomial` | 6.2 | 8/2487 | 37 / 19 | 0/2487 | 1 / 7 |
| `ce-stepwise` | 6.3 | 325/2487 | 1 / 1 | not completed | 1 / - |
| `ce-quantile` | 7.1 | 48/2487 (50) | 49 / 49 (51) | 94/2487 (222) | 24 / 15 (24) |
| `ce-quantile-linear` | 7.2 | 54/2487 (64) | 33 / 18 | 32/2487 | 10 / 9 |
| `ce-two-quantiles` | 7.3 | 5/2487 (7) | 72 / 97 (90) | 0/2487 | 4 / 10 (8) |
| `nhtsa-polynomial` | 8.1 | 0/3276 (1) | 73 / 53 (54) | 35/3276 (0) | 10 / 5 (1) |
| `solder-poisson` | 9.1 | 277/720 | 116 / 172 | 342/720 (471) | 28 / 48 (39) |
| `solder-poisson-linear` | 9.2 | 12/720 | 56 / 60 | 720/720 | 3 / 3 |
| `lungcancer-poisson-offset` | 9.3 | 82/920 | 56 / 16 | 920/920 | 3 / 3 |

The pruned `ce-stepwise` fit does not finish within the 900-second timeout (an earlier attempt was stopped after 39 minutes). The full tree is a single leaf, but each of the ten CV folds grows a stepwise tree, and that is slow on 2487 rows with 383 numeric, 48 categorical and 116 flag variables.

The importance example (`rhc-importance`, manual section 16.1) compares the variables above GUIDE's 95% and 99% thresholds (its groups A and B) and the rank correlation of the scores. Its permutations come from different random streams, so the sets are not expected to match exactly; see [importance validation](importance-validation.md#manual-example-rhc-2026-11-01).

| Example | Manual | Above 95%, native / guide-rs | Same side of 95% | Above 99%, native / guide-rs | Rank correlation |
|---|---|---|---|---|---|
| `rhc-importance` | 16.1 | 37 / 43 (44) | 47/53 (46) | 32 / 41 (42) | 0.986 (0.974) |

## Known causes

These are tracked in the [coverage checklist](coverage.md):

- **Classification selection binning.** Probes of GUIDE 46.3 show classification fits binning numeric variables at quantiles (four groups from 30 rows per class, three below), where guide-rs uses mean-based cuts; the frozen classification captures use binary variables and could not tell them apart. This is the main remaining source of differences on `fish-constant` and `ce-classification`.
- **Selection tests of sparse variables.** Reference25 matched native's grouping of categorical and few-valued numeric variables, and reference26 its 1-df ranking scale, tied quartile cuts and one-row groups (see the [validation record](validation.md#selection-scale-tied-quantiles-and-importance-trees-2026-11-01)); together they bring the pruned CE least-squares tree into full agreement. Few-valued variables in classification and interaction tests are still grouped differently.
- **Polynomial and linear splits with missing regressors** (`ce-polynomial`, `nhtsa-polynomial`, the linear quantile and Poisson examples).
- **Linear splits in classification importance trees**, which guide-rs does not make (`rhc-importance`).
- **Stepwise leaves.** Further stepwise parity is deferred: the root model already differs (325/2487 rows agree with a single leaf on both sides).

Examples omitted as unsupported: censored response (including importance scoring with one, section 16.2), multiresponse, longitudinal, treatment and propensity models, DIF scoring, and logistic regression with an E variable.

## Reproducing

```sh
cargo build --locked --release
python3 tools/compare_manual_examples.py --binary .reference/upstream/macos-arm64/guide --mode both
```

The first run downloads `datafiles.zip` (SHA-256 `8e734235...4e72`) into `.reference/manual-data/`. `--timeout` limits each native or guide-rs run (default 1800 seconds), and `--json` writes the results to a file.
