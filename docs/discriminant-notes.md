# LDA/QDA node models: native probe findings

Working notes from black-box probes of the pinned GUIDE 46.3 binary (and 46.2 where noted), 2026-10-27, in preparation for classification LDA/QDA node models. Nothing here is implemented yet, and no captures are frozen; treat it as the starting point for that item in [the coverage checklist](coverage.md). Data were iris (`tests/fixtures/datasets/iris.csv`) and seeded synthetic Gaussian designs with 2 or 3 classes, 2 to 4 signal and 1 to 8 noise variables, rounded to four decimals before being written for GUIDE.

## Running GUIDE

The interactive writer asks "1 for basic tree, 2 for trees with higher accuracy" before the node model; only the second shows the model menu, and the answer is not written to the batch file. The node-model codes differ by version:

| Version | Codes |
|---|---|
| 46.2 | 1 simple, 2 nearest-neighbor, 3 kernel, 4 LDA, 5 QDA, 6 linear splits with constant nodes |
| 46.3 | 1 simple, 2 nearest-neighbor, 3 kernel, 4 LDA, 5 QDA, 6 bivariate QDA, 7 linear splits with constant nodes |

LDA is the menu default. With codes 4 to 6 there are no interaction or linear-split prompts. Defaults in this mode are 5 split levels and minimum node size 10, independent of n (checked for n = 40 to 3000). A no-pruning run with explicit depth and minimum node size is:

```
 1  (model fitting)    "o.out"    1  (one tree)    1  (classification)
 4  (LDA; 5 QDA, 6 bivariate QDA in 46.3)    3  (no pruning)    "data.dsc"
 1  (estimated priors; 3 then a file)    1  (unit costs)    1  (default split count)
 2  then depth    2  then minimum node size
 0  (no LaTeX)    2  "o.coef"    2  "o.fit"    1  (no R)    2  (no plot code)
```

Batch mode still validates its inputs: depth must exceed 1 and minimum node size cannot exceed n/2, and fixed-size pruning requires at least 3 leaves, so a single-node fit is only obtainable from data the model classifies perfectly. The `.coef` file lists, per node, each variable used with its node mean and both coordinate vectors. The `.fit` file gives each case's terminal node, its two coordinate values (raw `x·a`, not centered by the printed means) and the observed and predicted classes. Both are better acceptance evidence than the report.

## Established

- **Discriminant coordinates.** Coefficients are the generalized eigenvectors of the between-class scatter B against the total scatter T (unweighted sums of squares about the node means), normalized so `a'Ta = 1`. Recomputing them reproduces native's to 8 printed digits at every iris node, up to sign. Because of that normalization, a coordinate's total variance is `1/(n - 1)`; a coordinate with no between-class separation has pooled variance `1/(n - g)` (the `0.100000` printed for a 12-case, 2-class node is this, not a floor).
- **Splits.** Every split is `D <= c` on the node's first coordinate. Cuts are midpoints of adjacent sorted coordinate values (507 of 507 synthetic splits). Rows with missing values are routed by their imputed coordinate value, not by a separate learned direction (29 of 29 checked rows).
- **Variable subset.** Each node fits its own subset of the candidate columns: numeric variables, k-1 dummies for each categorical (first level as reference; C variables become B), and a `.NA` indicator for each variable with missing values, each competing on its own (a node can use `sl.NA` without `sl`). The kept variables are always the top of the marginal one-way ANOVA F ranking within the node (160 of 160 synthetic roots), with at least two kept. GUIDE's own chi-square ranking agrees in only 66 of 160, and forward Wilks stepwise selection disagrees on iris.
- **Missing values.** Missing numeric values are imputed by the node mean of the observed values; this reproduces native coefficients at every node, where root-mean or zero imputation fails.
- **LDA leaves.** A case is classified by Gaussian densities on the first two coordinates (one with two classes) treated as independent: class means, a pooled variance per coordinate with denominator `n - g`, and the leaf priors below. This reproduces all 150 native iris predictions at depths 1, 2 and 5.
- **QDA leaves (46.3 code 5).** As LDA but with a per-class variance per coordinate (denominator `n_c - 1`) and the `log variance` term, still with independent coordinates, and always two coordinates, even in a two-class node. Reproduces all 150 predictions at depths 1, 2 and 5.
- **Bivariate QDA leaves (code 6).** A full per-class 2×2 covariance. Printed correlations match the sample correlation except for classes with two cases (below), and with native's printed correlation the rule reproduces every prediction in nodes whose class covariances are nonsingular.
- **Leaf priors.** The printed prior is the node posterior `π_c n_c / N_c` (normalized; `π` the training priors), corrected as `(N p + π_c) / (N + 1)` with N the root size, exactly `Context::correct_zero_posteriors`. Classes absent from the node keep `π_c / (N + 1)`. Explicit priors change only these; coordinates and cuts are identical to the estimated-priors fit.
- **Stopping.** A node splits only if `n >= ceil(2.5 × min_node)` (exact for minimum sizes 3, 5, 8, 10, 15 and 20), and children are at least `min_node`. Nodes the leaf model classifies without error do not split (52 mixed-class zero-error leaves above the size threshold); single-class nodes print "All cases classified as".
- **46.2.** LDA trees, coefficients and predictions on iris are identical to 46.3. QDA differs: 46.2 writes `NA` for the second coordinate of two-class nodes, chooses a different root cut (-0.0875 against -0.0840) and grows a different tree. 46.2 is not a regression reference for QDA.
- **Not observed.** The "crimcoords switched in node" messages of manual section 5.2 (46.1) appeared in no 46.3 or 46.2 run.

## Open questions

- **How many variables a node keeps.** Ranking is settled, the count is not. It is not a fixed F p-value threshold (kept p up to 0.18 beyond the first two, dropped p as low as 0.051), a ratio to the largest or mean F, a gap rule, a sequential Wilks partial-F threshold (best 127 of 160), a training or leave-one-out LDA error minimum (73 of 160), or a chi-square, Bonferroni or F-or-Levene significance count (best 147 of 160). A raw F p < 0.05 count with a minimum of two fits 143 of 160; every misfit keeps more variables than it predicts.
- **The cut criterion.** Native's cut is not the constant-node class impurity minimum (misclassification 144, Gini 39, entropy 36 of 507), not the one-dimensional LDA or QDA boundary between two classes, not the minimum total training error of LDA/QDA leaves refitted in both children on the parent's variables (27 of 106), and not guide-rs's constant-node classification search applied to native's coordinate (4 of 52 LDA and 4 of 54 QDA splits). Some cuts are the node median (the iris LDA root, roughly one synthetic split in six). Candidate count limits (default 100 splits) are untested.
- **The second coordinate in two-class nodes.** Its direction is any null vector of B, and 46.3 QDA depends on it. Cholesky or symmetric whitening of T or W does not reproduce native's choice. Without it, two-class QDA leaves cannot match native exactly; a documented deterministic choice is the alternative.
- **Coordinate sign.** No simple convention (first, last or largest coefficient positive, class mean order, sign of `a·mean`) matches more than 218 of 341 nodes, as for the high-cardinality categorical coordinates. The sign only exchanges which child is left, so predictions are unaffected.
- **Degenerate classes.** For a class with two cases native prints correlations of ±0.50 or 1.0 where the sample value is ±1, and its predictions in such nodes follow its printed value. Whether this is a guard or numerical noise is unknown; it should be a documented guard in guide-rs rather than reproduced.
- **Not yet probed.** Weights (the manual reports "All positive weights treated as 1"), P variables (manual: excluded from LDA/QDA models with a warning), pruning and CV with these models, ensembles (LDA has been off in forests since version 24), and multiclass nodes with fewer cases than variables.

## Consequences for implementation

Leaf fitting, coordinates, imputation, routing, priors and the size and zero-error stopping rules are specified well enough to implement and test against frozen `.coef` and `.fit` captures. The variable count and the cut criterion decide whether native trees can be matched at all; until they are found, a guide-rs implementation can match native leaf models only on native's own structure, and tree-level parity would be reported as a known difference.
