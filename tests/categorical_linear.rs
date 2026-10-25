use guide_rs::*;
use std::{fs, path::Path};

fn fixture(name: &str) -> (Dataset, FitOptions) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/categorical-linear/guide-46.3")
        .join(format!("categorical-linear-{name}"));
    (
        serde_json::from_str(&fs::read_to_string(path.join("dataset.json")).unwrap()).unwrap(),
        serde_json::from_str(&fs::read_to_string(path.join("options.json")).unwrap()).unwrap(),
    )
}
fn value(model: &FittedModel, row: Vec<Value>) -> f64 {
    let Prediction::Regression { value } = model.predict(&[row]).unwrap()[0] else {
        panic!()
    };
    value
}

#[test]
fn future_levels_use_the_fitted_reference_without_changing_the_codebook() {
    for (name, expected) in [
        ("reference", 2.0),
        ("inactive-reference", 14.0),
        ("observed-missing", 20.0),
    ] {
        let (data, options) = fixture(name);
        let model = FittedModel::fit(&data, &options).unwrap();
        let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
        for row in [vec![Value::Text("future".into())], vec![Value::Missing]] {
            assert!((value(&model, row.clone()) - expected).abs() < 1e-9);
            assert_eq!(value(&model, row.clone()), value(&restored, row));
        }
    }
}

#[test]
fn legacy_categorical_terms_preserve_their_original_predictions() {
    let (data, options) = fixture("inactive-reference");
    let current = FittedModel::fit(&data, &options).unwrap();
    assert!((value(&current, vec![Value::Text("future".into())]) - 14.0).abs() < 1e-9);
    let mut saved = serde_json::to_value(current).unwrap();
    saved["algorithm_version"] = serde_json::json!("guide-rs-0.1-experimental-reference7");
    saved["trees"][0]["root"]["model"] = serde_json::json!({
        "kind":"linear","intercept":2.0,"coefficients":[6.0,12.0],
        "terms":[{"feature":0,"power":1,"category":"b","center":0.0,"scale":1.0,"impute":0.0,"missing_indicator":false},
                 {"feature":0,"power":1,"category":"c","center":0.0,"scale":1.0,"impute":0.0,"missing_indicator":false}],
        "loss":{"kind":"squared_error"},"converged":true});
    let restored = FittedModel::from_json(&saved.to_string()).unwrap();
    assert_eq!(value(&restored, vec![Value::Text("future".into())]), 2.0);
    assert_eq!(value(&restored, vec![Value::Missing]), 2.0);
    assert_eq!(value(&restored, vec![Value::Text("c".into())]), 14.0);
}

#[test]
fn literal_na_and_missing_remain_distinct_in_the_typed_api() {
    let (mut data, options) = fixture("observed-missing");
    for row in &mut data.rows {
        if row[0] == Value::Text("a".into()) {
            row[0] = Value::Text("NA".into());
        }
    }
    let model = FittedModel::fit(&data, &options).unwrap();
    assert!((value(&model, vec![Value::Text("NA".into())]) - 2.0).abs() < 1e-9);
    assert!((value(&model, vec![Value::Missing]) - 20.0).abs() < 1e-9);
}
