# Proportional hazards: native probe findings

Working notes from black-box probes of the pinned GUIDE 46.3 binary, 2026-10-25, in preparation for censored-response support. Nothing here is implemented yet, and no captures are frozen; treat it as a starting point for the survival milestone in [the coverage checklist](coverage.md).

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

## Next steps

1. Determine the growth-time baseline schedule with designs where per-split and per-level updates disagree.
2. Implement constant-node PH trees on the existing Poisson-with-exposure machinery: exposure `Λ0(t)`, response δ, then the joint re-estimation.
3. Freeze captures that compare root chi-square ranks, splits, node constants and fit-file columns, with declared tolerances for the documented non-convergence.
4. Then restricted mean event time, pruning and linear node models.
