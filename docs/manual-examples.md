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

Recorded 2026-11-02 at revision `guide-rs-0.1-experimental-reference28`. Each row shows training rows agreeing, then terminal nodes as native / guide-rs; parenthesized values are from reference27 where they changed.

| Example | Manual | Unpruned rows | Unpruned leaves | Pruned rows | Pruned leaves |
|---|---|---|---|---|---|
| `fish-constant` | 4.1 | 152/159 | 10 / 12 | 154/159 | 8 / 9 |
| `fish-lda` | 4.2 | 159/159 | 1 / 1 | 159/159 | 1 / 1 |
| `ce-classification` | 5.1 | 3267/3965 | 63 / 39 | 3341/3965 | 20 / 36 |
| `ce-constant` | 6.1 | 1572/2487 | 65 / 62 | 2487/2487 | 9 / 9 |
| `ce-polynomial` | 6.2 | 25/2487 (8) | 37 / 25 (19) | 0/2487 | 1 / 4 (7) |
| `ce-stepwise` | 6.3 | 325/2487 | 1 / 1 | not completed | 1 / - |
| `ce-quantile` | 7.1 | 1397/2487 (48) | 49 / 52 (49) | 1306/2487 (94) | 24 / 24 (15) |
| `ce-quantile-linear` | 7.2 | 65/2487 (54) | 33 / 18 | not completed (32) | 10 / - (9) |
| `ce-two-quantiles` | 7.3 | 2/2487 (5) | 72 / 86 (97) | 0/2487 | 4 / 20 (10) |
| `nhtsa-polynomial` | 8.1 | 1/3276 (0) | 73 / 60 (51) | 0/3276 (35) | 10 / 1 (5) |
| `solder-poisson` | 9.1 | 277/720 | 116 / 172 | 342/720 | 28 / 48 |
| `solder-poisson-linear` | 9.2 | 12/720 | 56 / 60 | 720/720 | 3 / 3 |
| `lungcancer-poisson-offset` | 9.3 | 82/920 | 56 / 16 | 920/920 | 3 / 3 |

Row counts are low wherever the trees first differ near the root, however close the rest is, and pruned counts also depend on GUIDE's random folds. The `nhtsa-polynomial` tree differs from native at its root both before and after reference28, so its lower pruned count does not come from the numeric cuts.

The pruned `ce-stepwise` fit does not finish within the 900-second timeout (an earlier attempt was stopped after 39 minutes). The full tree is a single leaf, but each of the ten CV folds grows a stepwise tree, and that is slow on 2487 rows with 383 numeric, 48 categorical and 116 flag variables. Since reference28 the pruned `ce-quantile-linear` fit does not finish within 900 seconds either: the unpruned fit went from 85 to 246 seconds, because the scan evaluates as many cuts as native does (100 against 40 at the root, 99 against 23 at the next node), each with linear programs for both children.

The importance example (`rhc-importance`, manual section 16.1) compares the variables above GUIDE's 95% and 99% thresholds (its groups A and B) and the rank correlation of the scores. Its permutations come from different random streams, so the sets are not expected to match exactly; see [importance validation](importance-validation.md#manual-example-rhc-2026-11-01).

| Example | Manual | Above 95%, native / guide-rs | Same side of 95% | Above 99%, native / guide-rs | Rank correlation |
|---|---|---|---|---|---|
| `rhc-importance` | 16.1 | 37 / 43 | 47/53 | 32 / 41 | 0.986 |

## Known causes

These are tracked in the [coverage checklist](coverage.md):

- **Linear splits.** Native's classification fits (`fish-constant`, `ce-classification`, both with linear splits after univariate ones) and its classification importance trees (`rhc-importance`) make linear splits that guide-rs reproduces differently or not at all. The fish root ranking now matches native's exactly (reference27).
- **Selection tests of sparse variables.** Reference25 matched native's grouping of categorical and few-valued numeric variables, and reference26 its 1-df ranking scale, tied quartile cuts and one-row groups (see the [validation record](validation.md#selection-scale-tied-quantiles-and-importance-trees-2026-11-01)); together they bring the pruned CE least-squares tree into full agreement. Few-valued numeric variables in interaction tests are still binned as other numeric variables, unverified.
- **Categorical subset search.** Quantile, two-quantile, Poisson and polynomial trees choose different category subsets from native. This is the first difference at the root of `ce-two-quantiles`, `ce-polynomial` and `nhtsa-polynomial`, at node 2 of `lungcancer-poisson-offset`, and at the `STATE` nodes of `ce-quantile`; `solder-poisson`, whose predictors are all categorical, also differs this way.
- **Selection ties between monotone variables.** Node 2 of `ce-quantile` splits on `FINCBTAX` where native splits on `INC_RANK`. Income and its rank fall into the same quartile groups, so their test statistics tie, and which one native picks changes with the set of columns in the file.
- **Variable selection in linear quantile trees** (`ce-quantile-linear`, from node 2).
- **Stepwise leaves.** Further stepwise parity is deferred: the root model already differs (325/2487 rows agree with a single leaf on both sides).

Examples omitted as unsupported: censored response (including importance scoring with one, section 16.2), multiresponse, longitudinal, treatment and propensity models, DIF scoring, and logistic regression with an E variable.

## Reproducing

```sh
cargo build --locked --release
python3 tools/compare_manual_examples.py --binary .reference/upstream/macos-arm64/guide --mode both
```

The first run downloads `datafiles.zip` (SHA-256 `8e734235...4e72`) into `.reference/manual-data/`. `--timeout` limits each native or guide-rs run (default 1800 seconds), and `--json` writes the results to a file.
