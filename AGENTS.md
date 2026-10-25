# Repository Guidelines

## Project Structure & Module Organization

- `src/lib.rs` exposes the Rust API; `src/main.rs` implements the CLI.
- Core modules separate data/configuration, node fitting, variable selection, split search, pruning, ensembles, importance, and numerical routines.
- `tests/` contains integration, numerical, CLI, and reference comparisons. `tests/fixtures/` preserves native outputs, diagnostics, and example datasets.
- `tools/` contains Python capture/integrity utilities and independent R oracles.
- `python/` is a separate Cargo workspace: PyO3 bindings (`src/lib.rs`, module `guide_rs._core`) and the scikit-learn estimators in `guide_rs/`. Options and schemas cross the boundary as core JSON, so validation stays in Rust.
- `examples/` holds JSON configurations; `docs/coverage.md` tracks implemented capabilities and remaining work.
- Ignored `assets/` and `.reference/upstream/` hold optional local papers and pinned executables. Ordinary tests must remain independent of them.

## Build, Test, and Development Commands

- `cargo build --locked --release`: build the optimized CLI/library.
- `cargo run -- --help`: inspect CLI commands; `cargo run -- config` prints defaults.
- `cargo fmt --all -- --check`: verify formatting.
- `cargo clippy --locked --all-targets -- -D warnings`: enforce warning-free linting.
- `cargo test --locked`: run ordinary Rust tests.
- `python3 -m unittest discover -s tools -p 'test_reference_*.py'`: verify capture tooling and fixture integrity.
- `cargo test --locked --test interoperability -- --ignored`: run independent R checks; requires Rscript, quantreg, and rpart.
- `cargo test --locked --release --test statistical -- --ignored`: run statistical smoke checks.
- In `python/`: `uv sync` builds the extension in editable mode (rebuilt automatically when Rust sources change); `cargo clippy --locked --all-targets -- -D warnings` lints the binding crate. `tests/test_estimators.py` requires the estimators to reproduce the core model on every reference case they can express; extend `estimator_for` when exposing new options.

## Coding Style & Naming Conventions

Use Rust 2024, rustfmt's four-space indentation, `snake_case` functions/modules, `PascalCase` types, and `SCREAMING_SNAKE_CASE` constants. Keep the runtime pure Rust. Preserve strict configuration/model validation and deterministic seeded behavior.

## Testing Guidelines

Use descriptive `#[test]` names stating the behavior, such as `saved_models_preserve_predictions`. Add focused regressions for statistical or compatibility changes. Pin native versions, inputs, hashes, and tolerances; never overwrite accepted captures or relax unexplained mismatches. Preserve saved-model compatibility and update `ALGORITHM_VERSION` when numerical behavior changes. Record evidence in the relevant validation document and coverage checklist.

## Git Workflow

- `main` is the only long-lived branch and must always pass CI. Never commit or push directly to it.
- Start each change from an up-to-date `main` on a short-lived branch named by type and topic: `feat/survival-data`, `fix/bagging-cv-pruning`, `docs/readme-shape`, `ci/...`, `test/...`, `refactor/...`, `chore/...`.
- Keep one logical change per branch. Rebase onto `main` before opening a PR rather than merging `main` into the branch.
- Open a PR for every branch and merge only after all CI jobs pass (Linux/macOS/Windows tests and the independent R job). Merge with a merge commit; do not squash. Delete the branch after merging.
- Commit messages follow Conventional Commits (`fix(ensemble): ...`) in the imperative mood, first line under 72 characters, with a body explaining why when the subject alone is not enough. Reference issues or PRs in parentheses at the end of the subject, e.g. `(#12)`. Mark breaking changes to the CLI, Rust API, or saved-model format with `!` and a `BREAKING CHANGE:` footer.
- Each commit should build and pass the quality gate on its own so `git bisect` stays useful.
- Put native recaptures or new frozen fixtures in their own commit, separate from the code that consumes them, and name the GUIDE version and case family in the message.
- Commits that change numerical behavior must also bump `ALGORITHM_VERSION` and update `docs/coverage.md` and the relevant validation record in the same PR.
- Add a `CHANGELOG.md` entry under `Unreleased` for any user-visible change.
- Before pushing, check `git status` for stray files; `assets/`, `.reference/`, `target/`, and local scratch output must never be committed.

PRs should explain the problem, resulting behavior, validation commands/results, and remaining limitations; link relevant issues.

## Scope & Reference Hygiene

Preserve independently verified QR and pruning correctness when native GUIDE disagrees. Further stepwise parity is deferred; R prediction export was removed and stays out of scope. Keep local binaries, papers, caches, and private metadata out of commits while retaining frozen acceptance evidence.
