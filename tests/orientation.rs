use guide_rs::*;
use std::{fs, path::Path};

#[test]
fn saved_reference9_orientation_and_future_predictions_are_preserved() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let old = FittedModel::from_json(
        &fs::read_to_string(
            root.join("tests/fixtures/orientation/diagnostics/legacy-reference9.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        old.algorithm_version,
        "guide-rs-0.1-experimental-reference9"
    );
    let base = root.join("tests/fixtures/selection/guide-46.3/selection-missing-k11");
    let data: Dataset =
        serde_json::from_str(&fs::read_to_string(base.join("dataset.json")).unwrap()).unwrap();
    let options: FitOptions =
        serde_json::from_str(&fs::read_to_string(base.join("options.json")).unwrap()).unwrap();
    let current = FittedModel::fit(&data, &options).unwrap();
    let future = vec![vec![Value::Text("future".into()), Value::Number(23.0)]];
    let prediction = |model: &FittedModel| {
        let Prediction::Regression { value } = model.predict(&future).unwrap()[0] else {
            panic!()
        };
        value
    };
    assert!((prediction(&old) + 17.16).abs() < 1e-9);
    assert!((prediction(&current) - 17.16).abs() < 1e-9);
    let restored = FittedModel::from_json(&old.to_json().unwrap()).unwrap();
    assert_eq!(prediction(&old), prediction(&restored));
    for (a, b) in old
        .predict(&data.rows)
        .unwrap()
        .iter()
        .zip(current.predict(&data.rows).unwrap())
    {
        let (Prediction::Regression { value: a }, Prediction::Regression { value: b }) = (a, b)
        else {
            panic!()
        };
        assert!((a - b).abs() < 1e-9);
    }
}
