use guide_rs::*;
use std::{fs, path::Path};

#[test]
fn native_singleton_zero_positive_and_interaction_selection_boundaries() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/selection");
    for version in ["46.3", "46.2"] {
        let paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(paths.len(), 36);
        for path in paths {
            let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
            let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
            let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
            let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
            let model = FittedModel::fit(&data, &options).unwrap();
            let result = &manifest["result"];
            if manifest["case"] == "selection-missing-k11" {
                // The earlier complementary child-orientation gap is resolved.
                let Some(Split::Categorical {
                    left_categories,
                    missing_left,
                    ..
                }) = &model.trees[0].root.split
                else {
                    panic!()
                };
                assert_eq!(left_categories, &["c"]);
                assert!(missing_left);
                assert_eq!(
                    result["splits"][0]["left_categories"],
                    serde_json::json!(["c"])
                );
                assert_eq!(result["splits"][0]["missing_left"], true);
            }
            assert_eq!(
                model.trees[0].root.leaves() as u64,
                result["leaf_count"].as_u64().unwrap(),
                "{}",
                path.display()
            );
            let diagnostics = &model.trees[0].root.diagnostics;
            assert!(!diagnostics.is_empty());
            match result["expected_root_association"].as_str().unwrap() {
                "zero" => assert!(
                    diagnostics
                        .iter()
                        .all(|d| d.chi_square == 0.0 && d.p_value == 1.0)
                ),
                "positive" => {
                    assert!(diagnostics.iter().any(|d| d.chi_square > 0.0));
                    if path
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .contains("near-one")
                    {
                        assert!(diagnostics[0].p_value > 0.99);
                    }
                }
                "interaction" => {
                    assert!(
                        diagnostics
                            .iter()
                            .all(|d| d.chi_square == 0.0 && d.p_value == 1.0)
                    );
                    assert!(diagnostics.iter().all(|d| d.interaction_with.is_some()));
                }
                _ => panic!(),
            }
            let actual = model.predict(&data.rows).unwrap();
            for (i, p) in actual.iter().enumerate() {
                let Prediction::Regression { value } = p else {
                    panic!()
                };
                assert!(
                    (value - result["predictions"][i].as_f64().unwrap()).abs()
                        <= result["prediction_tolerances"][i].as_f64().unwrap(),
                    "{} row {i}",
                    path.display()
                );
            }
            let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(restored.predict(&data.rows).unwrap()).unwrap()
            );
        }
    }
}

#[test]
fn saved_reference8_trees_keep_their_pre_gate_predictions() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let saved =
        fs::read_to_string(base.join("selection/diagnostics/legacy-reference8.json")).unwrap();
    let old = FittedModel::from_json(&saved).unwrap();
    let path = base.join("categorical-linear/guide-46.3/categorical-linear-split-and-fit");
    let data: Dataset =
        serde_json::from_str(&fs::read_to_string(path.join("dataset.json")).unwrap()).unwrap();
    let options: FitOptions =
        serde_json::from_str(&fs::read_to_string(path.join("options.json")).unwrap()).unwrap();
    let current = FittedModel::fit(&data, &options).unwrap();
    assert_eq!(
        old.algorithm_version,
        "guide-rs-0.1-experimental-reference8"
    );
    assert_eq!(old.trees[0].root.leaves(), 2);
    assert_eq!(current.trees[0].root.leaves(), 1);
    for (prediction, y) in old.predict(&data.rows).unwrap().iter().zip(&data.targets) {
        let Prediction::Regression { value } = prediction else {
            panic!()
        };
        assert!((value - y.number().unwrap()).abs() < 1e-10);
    }
}
