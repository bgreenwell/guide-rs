use guide_rs::*;
use std::{fs, path::Path};

fn fixture(name: &str) -> (Dataset, FitOptions) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/linear-policy/guide-46.3")
        .join(name);
    (
        serde_json::from_str(&fs::read_to_string(path.join("dataset.json")).unwrap()).unwrap(),
        serde_json::from_str(&fs::read_to_string(path.join("options.json")).unwrap()).unwrap(),
    )
}
fn values(model: &FittedModel, rows: &[Vec<Value>]) -> Vec<f64> {
    model
        .predict(rows)
        .unwrap()
        .into_iter()
        .map(|p| match p {
            Prediction::Regression { value } => value,
            _ => panic!(),
        })
        .collect()
}

#[test]
fn limits_are_leaf_local_serialized_and_bound_extreme_inputs() {
    for (mode, expected) in [
        ("none", [203.0, -97.0, 26.0, 126.0]),
        ("node_range", [49.0, 103.0, 26.0, 126.0]),
        ("node_range_expanded", [53.6, 98.4, 26.0, 126.0]),
        ("global_range", [149.0, 3.0, 26.0, 126.0]),
    ] {
        let (data, options) = fixture(&format!("linear-limit-tree-{mode}"));
        let model = FittedModel::fit(&data, &options).unwrap();
        let predictions = values(&model, &data.rows[48..]);
        for (actual, expected) in predictions.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-9);
        }
        let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
        assert_eq!(predictions, values(&restored, &data.rows[48..]));
        if mode != "none" {
            let expected = if mode == "global_range" {
                149.0
            } else if mode == "node_range_expanded" {
                53.6
            } else {
                49.0
            };
            assert_eq!(
                values(&model, &[vec![Value::Number(0.0), Value::Number(f64::MAX)]])[0],
                expected
            );
        }
    }
}

#[test]
fn least_squares_search_risks_remain_untruncated() {
    let (data, mut options) = fixture("linear-limit-training-quadratic");
    let bounded = FittedModel::fit(&data, &options).unwrap();
    options.prediction_truncation = PredictionTruncation::None;
    let raw = FittedModel::fit(&data, &options).unwrap();
    assert!((raw.trees[0].root.risk - 877.8).abs() < 1e-9);
    assert!((bounded.trees[0].root.risk - raw.trees[0].root.risk).abs() < 1e-9);
    assert!((values(&raw, &data.rows[..1])[0] + 57.0).abs() < 1e-9);
    assert_eq!(values(&bounded, &data.rows[..1])[0], 0.0);
    for mode in ["none", "node_range"] {
        let (data, options) = fixture(&format!("linear-limit-fixed-{mode}"));
        let model = FittedModel::fit(&data, &options).unwrap();
        assert_eq!(model.trees[0].root.leaves(), 3);
        assert!((model.report.selected_alpha - 1250.0).abs() < 1e-7);
    }
}

#[test]
fn legacy_small_linear_models_load_without_refitting_or_limits() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/linear/guide-46.3/ls-linear-small");
    let data: Dataset =
        serde_json::from_str(&fs::read_to_string(path.join("dataset.json")).unwrap()).unwrap();
    let options: FitOptions =
        serde_json::from_str(&fs::read_to_string(path.join("options.json")).unwrap()).unwrap();
    let current = FittedModel::fit(&data, &options).unwrap();
    assert!(matches!(
        current.trees[0].root.model,
        LeafModel::Constant { .. }
    ));
    let mut saved = serde_json::to_value(&current).unwrap();
    saved["algorithm_version"] = serde_json::json!("guide-rs-0.1-experimental-reference6");
    saved["trees"][0]["root"]["model"] = serde_json::json!({
        "kind":"linear","intercept":2.8,"coefficients":[2.1,-4.0],
        "terms":[{"feature":0,"power":1,"category":null,"center":0.0,"scale":1.0,"impute":1.5,"missing_indicator":false},
                 {"feature":1,"power":1,"category":null,"center":0.0,"scale":1.0,"impute":0.75,"missing_indicator":false}],
        "loss":{"kind":"squared_error"},"converged":true});
    assert!(saved["options"].get("prediction_truncation").is_none());
    let restored = FittedModel::from_json(&saved.to_string()).unwrap();
    for (prediction, target) in values(&restored, &data.rows).iter().zip(&data.targets) {
        assert!((prediction - target.number().unwrap()).abs() < 1e-9);
    }
    assert!(
        (values(&restored, &[vec![Value::Number(100.0), Value::Number(0.0)]])[0] - 212.8).abs()
            < 1e-9
    );
}

#[test]
fn invalid_bounds_and_unsupported_modes_are_rejected() {
    assert!(
        FitOptions {
            prediction_truncation: PredictionTruncation::NodeRange,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    let (data, options) = fixture("linear-limit-root-global_range");
    for (lower, upper) in [(10.0, 1.0), (0.0, f64::INFINITY)] {
        let mut model = FittedModel::fit(&data, &options).unwrap();
        let LeafModel::Linear {
            prediction_bounds, ..
        } = &mut model.trees[0].root.model
        else {
            panic!()
        };
        *prediction_bounds = Some(PredictionBounds { lower, upper });
        assert!(model.validate().is_err());
    }
}
