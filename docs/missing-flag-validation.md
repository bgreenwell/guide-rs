# Missing-value flag validation

The 2026-10-25 milestone adds GUIDE missing-value flag (M) variables for constant node models, and corrects categorical split orientation. New fits record **`guide-rs-0.1-experimental-reference12`**. Twenty flag cases and eight orientation cases under each pinned executable add **56 captures**, bringing the fixed acceptance corpus to **544 captures across 272 distinct cases**. Both native versions agree on every parsed result. Earlier captures are unchanged.

## Verified semantics

The GUIDE 46.1 manual (chapter 5) describes flags as categorical columns that give reasons for missing values in an associated B, C, N, P or S variable. Black-box probes and the frozen captures establish the behavior for numeric and categorical parents:

- **Codes only matter for missing values.** A flag's levels are the codes recorded on rows where its parent is missing. Codes on observed rows are ignored.
- **Selection.** Each code forms its own missing group in the residual-sign and class tables, so a variable can be selected because of how its missing values differ by reason (`missing-flag-selection`, where the flag makes `x` outrank a competitor that wins when codes are ignored).
- **Split forms.** A split can send any nonempty subset of codes left: alone (`x = NA & x_ = "A"`), with a numerical cut (`x <= c or x = NA & x_ = "A", "B"`) or with a category set (`x in S or (x = NA & x_ in S*)`).
- **Routing.** A missing value goes left exactly when its code is in the left set. Missing values with no code, or with a code unseen in training, go right. When the left set is every code the data file records for the variable, GUIDE writes `or NA`, and then all missing values go left.
- **Ties.** On equal loss, cut splits prefer named codes over `or NA`, while missing-only splits prefer the plain `x = NA` form.
- **Categorical orientation.** The side with fewer categories goes left, counting missing values as one category (or each code, and uncoded values, separately). Ties keep the previous orientation. This rule also holds without flags; see below.

Flags are declared as a categorical feature with role `flag` naming its parent (see [usage](usage.md#missing-value-flags)). Linear and polynomial node models reject flags: GUIDE converts them to indicator (B) variables for those models, and that conversion is not implemented. Periodic parents are rejected because the manual's two descriptions of them conflict. At most 10 missing codes per flag are supported, since code subsets are enumerated exhaustively.

## Documented native difference

GUIDE only considers flag code subsets that contain the node's first (lexically smallest) code, for cut splits and for missing-only splits alike. That anchoring is harmless for categorical partitions, where a subset and its complement describe the same split, but not here: observed values are fixed on one side, so a subset and its complement are different splits. When the best split leaves the first code out, native chooses a worse one:

| Case | Best split | Native split |
|---|---|---|
| `numeric-left-none` | `x <= 25`, all missing right (exact fit) | `x <= 25 or x = NA & x_ = "A"` |
| `numeric-left-B` | `x <= 25 or x_ = "B"` | `x <= 25 or x_ = "A", "B"` |
| `numeric-left-BC` | `x <= 25 or x_ = "B", "C"` | `x <= 25 or NA` |
| `missing-only-C` | `x = NA & x_ = "C"` | `x = NA` |
| `selection` | `x = NA & x_ = "C"` | `x = NA & x_ = "A"` |

Without a flag, native finds the corresponding best splits. Rust searches every subset. For these five cases the Rust test requires a strictly lower training loss than native, and an independent Python brute force over every depth-1 split confirms both halves of the explanation: the best split over all subsets beats native, and the best split over subsets containing the first code reproduces native's loss exactly.

## Categorical orientation

Unbalanced designs show that GUIDE orients constant-node categorical splits so that the side with fewer categories is on the left, regardless of row counts or responses (`ls-categorical-orientation-{1..4}`, `class-categorical-orientation-{1..4}`). Rust previously oriented by response order, which matched only the balanced fixtures. Orientation does not change the partition of training rows, but it decides which child receives unseen categories, which always go right. Eight of the sixteen new captures fail on the previous revision. Multiple-linear SSE categorical splits keep their separately verified rule (first observed level on the left; see [orientation validation](orientation-validation.md)).

## Evidence

`tools/reference_missing_flag_cases.py` defines the flag designs and `tools/capture_reference_missing_flag.py` captures them, declaring the flag column as `m` in the DSC file. GUIDE's split file does not record flag codes, and writes categorical flag splits inconsistently, so the parser checks the flag declaration and retains per-row terminal nodes, predictions and the report's tree text. Every design appends prediction-only rows (missing response) with an unseen code, no code, and each training code, so routing is checked directly.

`tests/missing_flag.rs` fits each case and requires the same leaf count, a one-to-one correspondence between native and Rust terminal nodes over all rows, matching predictions within half the printed precision, and unchanged predictions after a JSON round trip. `tools/test_reference_missing_flag.py` verifies hashes, regenerated inputs, re-parsed results, version agreement and parser rejections.

Four held-out designs (`missing-flag-holdout-*`) were captured after the routing and orientation rules were fixed. Three passed unchanged. `holdout-categorical-uncoded` exposed a gap: with uncoded missing values in a node, the optimal partition can require the complement of a mean-ordered level prefix on the left, because uncoded values cannot go left. Rust now adds those complements, and all held-out cases pass.

```sh
python3 tools/capture_reference_missing_flag.py \
  --binary .reference/upstream/macos-arm64/guide --version 46.3 \
  --source-url https://pages.stat.wisc.edu/~loh/treeprogs/guide/mac/m1/guide.zip \
  --case missing-flag-numeric-left-AB --output /tmp/guide-463-flag
cargo test --locked --test missing_flag
python3 -m unittest discover -s tools -p 'test_reference_missing_flag.py'
```

A fifth check came from portability: the original `selection` design drew its noise with `random.gauss`, whose last digits differ between platform math libraries, so CI could not regenerate the captured inputs. Its replacement uses exact decimal noise; recapturing it exposed the anchoring difference above, which the earlier capture had not hit.

## Remaining

- Flags with linear, polynomial and stepwise node models (GUIDE's M-to-B conversion).
- Flags on periodic parents, flags in classification with priors or costs, and importance scoring with flags have no native captures yet.
- Interaction tests use the same per-code missing groups; their native construction is unverified.
