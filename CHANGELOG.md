# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `exposures_from_csv` reads prediction exposures with line-numbered errors
- Per-subcommand `--help` output

### Changed

- Classification trees merge sibling leaves that predict the same class, matching GUIDE
- `min_gain = 0` accepts zero-gain splits on the selected variable, matching GUIDE
- CSV parse errors report the line number and column name
- File I/O errors name the file that could not be read or written

### Removed

- R prediction-function export (`export --format r` and `FittedModel::to_r`)

### Fixed

- Blank lines in CSV input no longer cause a field-count error
- Bagged ensembles reject cross-validation pruning on bootstrap samples instead of under-pruning

## [0.1.0] - 2026-10-24

### Added

- Initial experimental implementation of GUIDE-style classification and regression trees targeting GUIDE 46.3
