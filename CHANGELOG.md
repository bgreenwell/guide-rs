# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- JSS manuscript skeleton, official style files, and a writing and submission guide in `manuscript/`

## [0.1.0] - 2026-10-27

### Added

- Experimental GUIDE classification and regression trees targeting GUIDE 46.3, with a CLI and Rust API
- `--threads` CLI option and `n_jobs` estimator parameter set the threads used for parallel fits
- Python package (`import guide`) with scikit-learn tree and forest estimators, feature roles and importance scores
- Python estimators accept periodic features, missing-value flags, logistic and dual-quantile losses, stepwise leaves, prediction truncation and validation pruning
- `guide.plot_tree` draws fitted trees with matplotlib
- Example notebooks comparing GUIDE with scikit-learn decision trees on its bundled datasets
- Example notebook comparing GUIDE bagging, forests and boosting with scikit-learn's ensembles
- Sphinx documentation for the Python package: user guide, API reference and the example notebooks
- Python wheels for Linux, macOS and Windows built and tested in CI, published to PyPI on version tags
- `pruning_loss` prunes classification trees and measures out-of-bag loss by log loss or Brier score instead of misclassification cost
- Experimental gradient boosting with GUIDE variable selection (`fit_boosting`, a `boosting` CLI section, `GuideBoostingRegressor` and `GuideBoostingClassifier`) for squared-error, Poisson and logistic regression and classification, with early stopping
- Multiclass trees split categorical predictors with more than 11 levels instead of rejecting them
- Missing-value flag variables (GUIDE M variables) for constant node models
- `exposures_from_csv` reads prediction exposures with line-numbered errors
- Per-subcommand `--help` output

### Changed

- Classification trees and forests that look ahead on interactions fit 3 to 7 times faster, with unchanged models
- Ensemble trees, cross-validation folds and importance permutations are fitted in parallel, with results independent of the thread count
- Pruning large trees is faster: the pruning sequence no longer copies the tree per stage, and held-out rows are routed once
- Variable selection and split search are faster and run in parallel at large nodes, with unchanged models
- Linear, polynomial and quantile-regression leaves fit faster, with unchanged models
- Squared-error regression trees with constant leaves fit about 2.5 times faster, with unchanged models
- Categorical splits put the side with fewer categories on the left, matching GUIDE; unseen categories may reach a different leaf than before
- Classification trees merge sibling leaves that predict the same class, matching GUIDE
- `min_gain = 0` accepts zero-gain splits on the selected variable, matching GUIDE
- With `min_gain = 0`, classification keeps splits that raise prior- or cost-adjusted impurity, matching GUIDE
- Linear quantile leaves need `p + 4` rows for a linear fit, matching GUIDE; smaller leaves are constant
- Poisson and quantile trees search GUIDE's fixed grid of cutpoints, Poisson children need a positive count, and quantile leaves average at exact ties, matching GUIDE
- Constant-leaf trees use the two-level lookahead for interaction pairs at every depth, with parent cuts on GUIDE's fixed grid, matching GUIDE more often
- Multiple-linear least-squares trees with numeric regressors use GUIDE's separate row minimums for split search and leaves, and keep the best split whatever its gain
- Poisson trees select split variables from Anscombe residual groups and quantile trees place zero residuals as GUIDE does; constant-response Poisson nodes are not split
- CSV parse errors report the line number and column name
- File I/O errors name the file that could not be read or written

### Removed

- R prediction-function export (`export --format r` and `FittedModel::to_r`)

### Fixed

- Blank lines in CSV input no longer cause a field-count error
- Bagged ensembles reject cross-validation pruning on bootstrap samples instead of under-pruning
