# Historical executable analysis

**Update, 2026-10-23:** Native Apple Silicon GUIDE 46.2 and 46.3 downloads now run successfully on this host. The original bundled 40.4 launch failure below is historical and remains unresolved. See [the working upstream reference](reference-upstream.md) for hashes, history boundaries, captured fixtures and the first compatibility correction.

Former path: `assets/guidevi/guidevi/resources/macos/guide`. The retired wrapper and this binary were removed during repository cleanup on 2026-10-24. The evidence below is a historical record, not a dependency of the Rust project.

| Property | Evidence |
|---|---|
| SHA-256 | `429a30d819226d6e3d9c5d4ecee6a7b01b5073da6e3e28e17ecdd5d7e4492565` |
| Length | 4,184,816 bytes |
| Architecture | x86-64 Mach-O executable |
| Version | 40.4, confirmed in `_main` initialization: immediate `0x342e3034` writes ASCII `40.4`; floating immediate `0x4221999a` is approximately 40.4 |
| Build date | December 10, 2022, from embedded banner |
| Compiler | NAG Fortran 6.2 per bundled resources README, supported by embedded NAG runtime symbols |
| Deployment target | macOS 11.1 in `LC_BUILD_VERSION` |
| Dependency | `/usr/lib/libSystem.B.dylib`; numerical/Fortran routines are linked into the executable |
| Symbols | 3,021 total symbol-table entries; `nm` reports 1,041 text symbols |
| Runtime status | Initial probe timed out after 8 seconds; subsequent sandboxed and approved unsandboxed probes terminated with signal 9 without stdout/stderr. Cause unconfirmed; sandboxing alone does not explain it. Quarantine/provenance attributes exist; they were not removed. |

## Recovered structure

- `_grow_tree_` calls model-specific routines: `_lsq_fit_`, `_classify_`, `_poisson_fit_`, `_quantile_fit_`, `_logistic_fit_`, `_treat_`, `_lon_`, and `_calibrate_`.
- `_select_interact_` calls curvature and interaction subroutines, contingency-table and distribution routines, category search, discriminant/PCA routines and model-specific split searches.
- `_cv_prune_` stratifies data, selects fold training data, grows trees, evaluates validation loss and restores/removes tree nodes.
- `_classify_` is recursive and references nearest-neighbor and kernel routines.
- `_luxury_MP_ranlux` and NAG random-number routines are present. Their presence does not prove which stream every operation uses.
- Numerical symbols include QR/least-squares routines, SVD/LAPACK routines, chi-square transforms, incomplete beta functions and smoothing splines.

Embedded source names include `main.f90`, `init.f90`, `read.f90`, `tree.f90`, `select.f90`, `prune.f90`, `class.f90`, `lsq.f90`, `lsmod.f90`, `poisson.f90`, `quantile.f90`, `logistic.f90`, `lon.f90`, `ransrt.f90`, `drawtree.f90`, `statpk.f90`, `nswc.f90`, and `nrmod.f90`. These names and symbols establish structure, not recovered Fortran source or full algorithm semantics.

## Version boundaries

The Python wrapper templates also specify 40.4 and implement only variable-importance workflows. Their tests check shapes/file bookkeeping, not numerical values. The originally bundled 2025 manual was version 44.1; it has since been replaced by the 46.1 manual dated January 24, 2026. Upstream revision history records changes after 40.4 affecting periodic variables, missing handling, propensity trees, forest defaults, stepwise selection and duplicate-predictor importance. Do not use the older binary's output as the expected result for those changed behaviors.

`tools/inspect_reference.py` reads the Mach-O headers, symbols and strings without executing or altering the binary. `--probe` is an explicit, timed, temporary-directory launch probe. The script does not modify permissions or security attributes. Its output can be retained as provenance when establishing numerical reference fixtures.

The obsolete 40.4-template prototype `tools/reference_fixture.py` was removed with the retired wrapper. Current fixture capture uses the version-checked 46.x tools described in [reference provenance](reference-upstream.md). To inspect the current locally cached primary binary:

```sh
python3 tools/inspect_reference.py .reference/upstream/macos-arm64/guide
```
