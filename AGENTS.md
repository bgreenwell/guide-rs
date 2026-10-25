# Repository Guidelines

## Project Structure & Module Organization

- `src/lib.rs` exposes the Rust API; `src/main.rs` implements the CLI.
- Core modules separate data/configuration, node fitting, variable selection, split search, pruning, ensembles, importance, and numerical routines.
- `tests/` contains integration, numerical, CLI, and reference comparisons. `tests/fixtures/` preserves native outputs, diagnostics, and example datasets.
- `tools/` contains Python capture/integrity utilities and independent R oracles.
- `examples/` holds JSON configurations; `docs/coverage.md` tracks implemented capabilities and remaining work.
- Ignored `assets/` and `.reference/upstream/` hold optional local papers and pinned executables. Ordinary tests must remain independent of them.

## Build, Test, and Development Commands

- `cargo build --locked --release`: build the optimized CLI/library.
- `cargo run -- --help`: inspect CLI commands; `cargo run -- config` prints defaults.
- `cargo fmt --check`: verify formatting.
- `cargo clippy --locked --all-targets -- -D warnings`: enforce warning-free linting.
- `cargo test --locked`: run ordinary Rust tests.
- `python3 -m unittest discover -s tools -p 'test_reference_*.py'`: verify capture tooling and fixture integrity.
- `cargo test --locked --test interoperability -- --ignored`: run independent R checks; requires Rscript, quantreg, and rpart.
- `cargo test --locked --release --test statistical -- --ignored`: run statistical smoke checks.

## Coding Style & Naming Conventions

Use Rust 2024, rustfmt's four-space indentation, `snake_case` functions/modules, `PascalCase` types, and `SCREAMING_SNAKE_CASE` constants. Keep the runtime pure Rust. Preserve strict configuration/model validation and deterministic seeded behavior.

## Testing Guidelines

Use descriptive `#[test]` names stating the behavior, such as `saved_models_preserve_predictions`. Add focused regressions for statistical or compatibility changes. Pin native versions, inputs, hashes, and tolerances; never overwrite accepted captures or relax unexplained mismatches. Preserve saved-model compatibility and update `ALGORITHM_VERSION` when numerical behavior changes. Record evidence in the relevant validation document and coverage checklist.

## Commit & Pull Request Guidelines

Git currently has no commits, so no established message convention exists. Use concise imperative subjects, such as `Fix categorical missing-value routing`. PRs should explain the problem, resulting behavior, validation commands/results, and remaining limitations; link relevant issues. Keep unrelated changes separate.

## Scope & Reference Hygiene

Preserve independently verified QR and pruning correctness when native GUIDE disagrees. Further stepwise parity is deferred; R prediction export is excluded. Keep local binaries, papers, caches, and private metadata out of commits while retaining frozen acceptance evidence.
