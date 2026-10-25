# Intentional difference from GUIDE's CV pruning

Rust retains the mathematically correct weakest-link search. The pinned GUIDE 46.2 and 46.3 builds produce incorrect intermediate CV estimates in four captured multilevel regression cases, including two with multiple-linear leaves. These differences are preserved and tested explicitly; they are not numerical tolerances or unexplained test exclusions. They can affect tree size on other data or with different SE allowances.

## Counterexample and evidence

`ls-loocv-path` contains 24 observations with x = 0,…,23 and six observations at each response level 0, 1, 7 and 10. A depth-two tree has four pure leaves. The main-tree weakest-link alphas are 0, 0.125, 1.125 and 16; the three-leaf evaluation alpha is sqrt(0.125 × 1.125) = 0.375.

A read-only LLDB inspection of the pinned 46.3 binary stopped in `cv_test_` at address `0x100246278`, after its candidate scan. ASLR was disabled for stable addresses; the executable was not patched. The observed fold had these nonterminal nodes:

| Node number | Alpha |
|---|---|
| 1 | 5.6937618147448008 |
| 3 | 1.173913043478261 |
| 2 | 0.11857707509881422 |

The scan selected **node 3**, although node 2 has the smallest alpha. It visited nodes in the order 1, 3, 7, 6, 2, 5, 4. The candidate update requires a larger node number even when the candidate has a strictly lower cost. Consequently, it ignores node 2 after selecting node 3. The 0.375 threshold then leaves the fold unpruned because the incorrectly selected cost exceeds the threshold.

The full node arrays, breakpoint, pinned binary hash and input-file hashes are retained in `tests/fixtures/reference/diagnostics/cv-node-order-46.3.json`. The debugger observation concerns 46.3; the frozen 46.2 outputs exhibit the same error estimates. No source or runtime code was translated into the Rust implementation.

## Independent calculation

The fixtures use leave-one-out CV, so their held-out partitions do not depend on either implementation's RNG. `tests/reference.rs` enumerates held-out predictions directly from the four explicit regions and their training means. It does not call the Rust pruning code to generate its expected results.

| Three-leaf CV candidate | Correct mean loss | Native mean loss | Correct naive SE | Native naive SE |
|---|---|---|---|---|
| Response levels 0, 1, 7, 10 | 2.284177 | 1.916667 | 1.739006 | 1.496427 |
| Response levels 0, 2, 8, 10 | 2.640496 | 2.212121 | 1.419875 | 1.449459 |

For the first case, the correct prediction in the collapsed left branch is its leave-one-out mean. The squared-error sum is `12 × (6/11)^2 + (7 − 0.5)^2 + (10 − 7)^2`, divided by 24. The native scan leaves the fold fully grown instead, giving `(1 + 36 + 9)/24`.

For the second case, correct pruning collapses the branch whose training SSE decreased when an observation was held out. Its mean loss is `(23 × (12/11)^2 + 36)/24`. The native scan fails to collapse the cheaper left branch, giving `(4 + 36 + 11 × (12/11)^2)/24`. SE uses the population variance of held-out losses divided by the number of held-out observations.

Tests assert both independently calculated results: Rust must match the correct result, and the preserved native table must match the explained flawed result within its printed precision. Every other CV stage and every final split, route, probability and prediction in the two earlier constant-node fixtures still receives the ordinary native comparison. Their final selected trees happen to agree at the captured 0.25-SE setting; this does not imply the intermediate errors are interchangeable.

## Other explicit conventions

Native weighted pruning reports `sum(w × loss)/N`. Rust reports `sum(w × loss)/sum(w)` to preserve weight-unit invariance. The capture parser retains native values and divides both their means and SEs by the full-sample mean weight for comparison. This is an explicit change of units; it does not justify changing SE formulas or relaxing numerical checks.

Classification priors affect fitted probabilities, decisions and training pruning risks. Held-out misclassification costs are averaged over actual validation observations without prior reweighting. Estimated priors are frozen from the full training data while fitting CV folds. The native prior/cost fixtures check these distinctions.


## Multiple-linear counterexamples, 2026-10-24

`linear-multilevel-loocv` and `linear-multilevel-loocv-weighted` each contain 64 observations in four regions with means 0, 3, 15 and 21 plus a linear `z` contribution and small deterministic noise. Both pinned native builds report identical four-leaf and three-leaf CV estimates. Independent weighted QR fits and exhaustive fold searches give these three-leaf estimates instead:

| Case | Correct mean | Correct naive SE | Native printed mean in Rust weight units |
|---|---:|---:|---:|
| Unweighted | 5.807620586189 | 3.258289911152 | 3.987 |
| Weighted | 5.987631175959 | 3.379119363133 | 4.156632 |

The four-leaf means are 3.987390019776 and 4.158233620094. Correct 0.25-SE CV therefore selects **four leaves**; native selects **three**. Unlike the earlier constant-node cases, these fixtures demonstrate an actual final-tree size difference.

`tools/multilevel_linear_oracle.R` also simulates the previously observed faulty scan: visit root/right/left and update the candidate only when both its alpha is lower and its node number is larger. That simulation reproduces every native CV mean and naive SE within the printed precision, and both native selected sizes. This supports attributing the new discrepancy to the known scan flaw; the debugger observation itself remains limited to the earlier pinned 46.3 case.

[The frozen independent evidence](../tests/fixtures/linear-tree/diagnostics/multilevel-cv-oracle.json) records script/input hashes, every held-out prediction, the four-/three-/two-/one-leaf CV summaries, and full-tree pruning-stage predictions. The R interoperability suite regenerates it; Python checks hashes and independently reconstructs weighted means/SEs. The fixed-size eight-region case in the same evidence also checks every full-tree pruning stage.

Regular Rust tests compare the correct CV result and its four-leaf predictions/routes with that independent evidence. They separately reconstruct the native final three-leaf tree using fixed-size pruning and compare all its splits, coefficients, row assignments and predictions with the native fixture. The native and correct results remain explicitly distinct. No fitting/pruning code was changed to reproduce the native scan defect.

## Additional categorical LOOCV discrepancy, 2026-10-24

The later `orientation-loocv` control exposes a separate two-leaf CV-table difference: native 46.2/46.3 mean **94.72**, versus independent exhaustive R/Rust **40.2845495213**. All 96 direct leave-one-out refits under each native version agree with the independent held-out predictions within their printed precision. The root score and selected two-leaf final tree agree. Correct CV remains in place.

No source-level cause is established for this categorical case. It is not attributed to the multilevel node-order flaw described in this document. See [the orientation validation record](orientation-validation.md#categorical-loocv-discrepancy) for the separately pinned evidence, all direct-refit outcomes, tolerances and reproduction.

