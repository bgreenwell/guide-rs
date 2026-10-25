# guide-rs

A pure Rust implementation of GUIDE-style classification and regression trees, targeting the behavior of GUIDE 46.3.

**Status: experimental and incomplete.** The core works, but this is not yet a full or numerically validated replacement for GUIDE. Survival, longitudinal, treatment-effect and several advanced node models are not implemented. Do not use it to reproduce published GUIDE analyses without independent validation; see [coverage and remaining work](docs/coverage.md).

## Features

- Separates association-based split-variable selection from split-point search, with interaction tests
- Constant, multiple-linear, polynomial, quantile, Poisson and logistic node models
- Numeric, categorical and periodic predictors; weights, exposures and missing values
- Cost-complexity, fixed-size, validation-sample and cross-validation pruning
- Bagged ensembles, forests and GUIDE-style importance scores
- Seeded, deterministic fits saved as JSON; text, Graphviz DOT and LaTeX tree output
- No GUIDE, Fortran, BLAS or LAPACK dependency

## Installation

```sh
cargo install --git https://github.com/bgreenwell/guide-rs
```

Or build from a clone with `cargo build --release`.

## Quick start

Using the bundled Iris data:

```sh
guide-rs fit --data tests/fixtures/datasets/iris.csv --config examples/iris.json --model iris.json
guide-rs predict --model iris.json --data tests/fixtures/datasets/iris.csv
guide-rs inspect --model iris.json
```

From Rust:

```rust
use guide_rs::{CsvOptions, FitOptions, FittedModel, Pruning, Task, dataset_from_csv};

let data = dataset_from_csv(
    "x,target\n1,0\n2,0\n3,1\n4,1\n",
    &CsvOptions::default(),
    &Task::default(),
)?;
let options = FitOptions { pruning: Pruning::None, ..Default::default() };
let model = FittedModel::fit(&data, &options)?;
let predictions = model.predict(&data.rows)?;
# Ok::<(), guide_rs::Error>(())
```

## Documentation

- [Usage and configuration](docs/usage.md)
- [Algorithm specification](docs/algorithms.md)
- [Coverage and remaining work](docs/coverage.md)
- [Validation record](docs/validation.md) and [upstream reference provenance](docs/reference-upstream.md)
- [Roadmap](docs/roadmap.md)
- API reference: `cargo doc --open`

## Development

Branch, commit and PR conventions are in [AGENTS.md](AGENTS.md). Work happens on short-lived branches merged into `main` through pull requests once CI passes.

Quality gate:

```sh
cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && cargo test
```

Additional checks:

```sh
python3 -m unittest discover -s tools -p 'test_reference_*.py'   # fixture integrity
cargo test --test interoperability -- --ignored                  # needs Rscript, quantreg, rpart
cargo test --release --test statistical -- --ignored             # statistical smoke checks
```

Ordinary tests compare against frozen GUIDE 46.2/46.3 outputs in `tests/fixtures/` and never run GUIDE. The GUIDE manual, papers and pinned native executables are optional local references kept in the ignored `assets/` and `.reference/` directories. No GUIDE Fortran source or runtime code is included.

## License

MIT
