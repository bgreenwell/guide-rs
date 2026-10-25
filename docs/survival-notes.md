# Proportional hazards: native probe findings

Working notes from black-box probes of the pinned GUIDE 46.3 binary, 2026-10-25, in preparation for censored-response support. Nothing here is implemented yet, and no captures are frozen; treat it as a starting point for the survival milestone in [the coverage checklist](coverage.md). By user decision (2026-10-25), survival was resumed after the Python wrapper landed and then parked again pending the native Poisson split-point rule below.

## Running GUIDE

The censored-response path is regression model type 4, then 1 for proportional hazards. With constant nodes, interaction tests, no pruning, an explicit depth and minimum node size, the generated input file is:

```
 1  (model fitting)            "out.txt"         1  (one tree)
 2  (regression)               4  (censored)     1  (proportional hazards)
 3  (constant)                 1  (interaction tests)
 2  (no pruning)               "data.dsc"
 2  then depth                 2  then minimum node size
 20 (iterations for maximum likelihood estimation)
 0  (no LaTeX)                 3  "splits.txt"   2  "pred.txt"   1  (no R)   1  (root rank)
```

The DSC file marks the time as `t` and the 0/1 event indicator as `d`. The fit file has columns `train node observed event logbasecumhaz survivalprob mediansurvtime`; `mediansurvtime` is the Kaplan-Meier median within the terminal node. The report prints each node's log-relative-hazard constant and, usefully, the root's per-variable selection statistics as 1-df chi-square equivalents ("Ranks of variables and their 1-df chi-squared values at root node").

## Established

- **Exclusions.** Censored observations with times below the smallest event time are dropped (199 of 200 rows in the first probe).
- **Root-only trees.** The baseline cumulative hazard is the Nelson-Aalen estimate, to printed precision (largest log error 4.9e-6). Its scale satisfies Σ Λ0(t_i) = Σ δ_i, which is why the root constant prints as 0.
- **Depth 1.** The tree, its split and the root chi-square ranks are identical to a GUIDE Poisson tree with response δ and offset `log Λ_NA(t)`; the Poisson node constants are the first step of the PH estimates. Native then re-estimates node constants and a Breslow baseline jointly, normalized as above. Its final values are within about 2e-4 (log hazard) of the exact Cox partial-likelihood fixed point; native's baseline increments imply node risks that do not satisfy its own node equations, so it stops before convergence.
- **Deeper trees.** The baseline is updated during growth. In a depth-3 probe, the native split of the second node is reproduced by a Poisson search using the baseline re-estimated after the root split, but not by the original Nelson-Aalen baseline. The final tree is not a Poisson tree grown on native's final baseline either (the third node differs), so updates are progressive; their schedule (per split, per level, iteration count) is not yet known.

## Native Poisson split points (2026-10-25, second session)

PH splits follow native Poisson splits, so the PH work depends on native Poisson growth, which the corpus does not cover. Probes with one numeric predictor (120 to 300 rows, unit exposures, random exposures, and 0/1 counts):

- **Partly deviance-optimal.** The native split is the minimum-deviance cut in 77 of 90 cases at 120 rows. In every miss with admissible children, it is the second-best cut by deviance, sometimes by a wide margin (1.4 on a total of 131). guide-rs, which minimizes deviance, matches native exactly when native is optimal.
- **Depends on node size, not on particular rows.** Mismatches rise with n: 0 of 15 at 60 to 115 rows, 2 or 3 of 15 at 118 to 120, 5 of 15 at 150, 11 of 15 at 200. Dropping any one row from a 120-row mismatch makes native optimal again.
- **Deterministic and order-invariant.** Permuting rows does not change the split.
- **Not explained by:** a fixed cap on candidate cuts (grids of 100 to 102 points by several index formulas; native's cut is not always on any of them), exact or limited-iteration child fits (one to five Newton or IRLS steps from the parent rate or from y + c), Pearson, raw or Anscombe residual criteria, node-size normalizations, residual-sign or t statistics, or restricting cuts to event rows or response changes.
- **Reported deviance** is the node deviance divided by n − 1; node rates match the exact MLE.

The search space for a deviance-like criterion with an n-dependent candidate rule is not exhausted, but black-box probing has stopped converging.

## PH oracle results

- **Final constants.** With native's final leaves, 5 or more joint Breslow iterations reproduce the native leaf constants to the printed four decimals, and the baseline to within 6e-5 in log.
- **Root selection.** Root chi-square ranks differ between native PH and a native Poisson tree with offset `log Λ_NA` (for example 22.9 against 25.0), so PH variable selection does not use exactly those Poisson residuals; the survey's mention of adjusted Anscombe residuals for Poisson trees is a candidate.

## Next steps

1. Determine the growth-time baseline schedule with designs where per-split and per-level updates disagree. This needs native Poisson split points first (above).
2. Implement constant-node PH trees on the existing Poisson-with-exposure machinery: exposure `Λ0(t)`, response δ, then the joint re-estimation.
3. Freeze captures that compare root chi-square ranks, splits, node constants and fit-file columns, with declared tolerances for the documented non-convergence.
4. Then restricted mean event time, pruning and linear node models.
