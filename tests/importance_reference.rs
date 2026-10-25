//! Frozen native importance reports. Compare reported sets/rank relations, not
//! Monte Carlo magnitudes from the unrelated Fortran and ChaCha RNG streams.
use guide_rs::*;
use std::{collections::BTreeSet, fs, path::Path};

#[test]
fn native_importance_sets_and_independently_checked_raw_scores() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/importance");
    for version in ["46.3", "46.2"] {
        let mut paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|p| p.unwrap().path())
            .collect();
        paths.sort();
        assert!(!paths.is_empty());
        for path in paths {
            let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
            let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
            assert_eq!(manifest["completed"], true);
            assert_eq!(manifest["returncode"], 0);
            assert_eq!(manifest["result"]["version"], version);
            assert_eq!(manifest["result"]["score_kind"], "normalized");
            let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
            let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
            let io: ImportanceOptions = serde_json::from_str(&read("importance.json")).unwrap();
            assert_eq!(
                manifest["settings"],
                serde_json::to_value(&options).unwrap()
            );
            assert_eq!(
                manifest["importance_options"],
                serde_json::to_value(&io).unwrap()
            );
            let actual = importance(&data, &options, &io).unwrap();
            let set = |values: &serde_json::Value| -> BTreeSet<String> {
                values
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_owned())
                    .collect()
            };
            let expected = set(&manifest["expected_important"]);
            for (native_group, rust_groups) in [
                ("A", vec!["highly_important"]),
                ("AB", vec!["highly_important", "likely_important"]),
            ] {
                let native: BTreeSet<_> = manifest["result"]["scores"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|s| native_group.contains(s["group"].as_str().unwrap()))
                    .map(|s| s["feature"].as_str().unwrap().to_owned())
                    .collect();
                let rust: BTreeSet<_> = actual
                    .scores
                    .iter()
                    .filter(|s| rust_groups.contains(&s.group.as_str()))
                    .map(|s| s.feature.clone())
                    .collect();
                assert_eq!(native, expected, "native {version} {}", manifest["case"]);
                assert_eq!(
                    rust, native,
                    "Rust vs native {version} {}",
                    manifest["case"]
                );
            }
            let zero = set(&manifest["expected_zero"]);
            assert_eq!(set(&manifest["result"]["omitted_features"]), zero);
            for s in &actual.scores {
                if zero.contains(&s.feature) {
                    assert_eq!(s.raw_score, 0.0);
                }
                if expected.contains(&s.feature) {
                    for z in actual.scores.iter().filter(|s| zero.contains(&s.feature)) {
                        assert!(s.adjusted_score > z.adjusted_score);
                    }
                }
            }
            // The native file supplies normalized scores only. These expected
            // root Pearson statistics are calculated from the balanced designs;
            // their tail/quantile transform is independently verified against R.
            for (feature, statistic) in manifest["independent_raw_statistics"].as_object().unwrap()
            {
                let raw = actual
                    .scores
                    .iter()
                    .find(|s| &s.feature == feature)
                    .unwrap()
                    .raw_score;
                let expected = (data.rows.len() as f64).sqrt()
                    * stats::chi_square_log_upper_quantile(
                        stats::chi_square_log_survival(
                            statistic["chi_square"].as_f64().unwrap(),
                            statistic["degrees_of_freedom"].as_u64().unwrap() as usize,
                        ),
                        1,
                    );
                assert!(
                    (raw - expected).abs() < 1e-9,
                    "{feature}: {raw} != {expected}"
                );
            }
            if data.schema.iter().any(|f| f.name == "copy") {
                let name = if data.schema.iter().any(|f| f.name == "a") {
                    "a"
                } else {
                    "signal"
                };
                let a = actual.scores.iter().find(|s| s.feature == name).unwrap();
                let copy = actual.scores.iter().find(|s| s.feature == "copy").unwrap();
                assert_eq!(a.raw_score, copy.raw_score);
                assert_eq!(a.null_mean, copy.null_mean);
                assert_eq!(a.adjusted_score, copy.adjusted_score);
            }
        }
    }
}
