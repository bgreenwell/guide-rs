# guide-rs

scikit-learn estimators for GUIDE classification and regression trees, backed by a pure Rust implementation of GUIDE 46.3.

GUIDE chooses each split variable with chi-square tests of association, including pairwise interaction tests, before searching for the split point. This avoids CART's bias toward variables with many distinct values, and leaves can hold linear, polynomial or quantile models instead of constants.

The package is experimental. See the [coverage checklist](https://github.com/bgreenwell/guide-rs/blob/main/docs/coverage.md) for what is validated against native GUIDE.

```{toctree}
:maxdepth: 2

getting-started
api
examples
```
