//! Models saved by published releases must keep loading and predicting the same.
//! The fixtures come from `tools/freeze_saved_models.py`, run under each release.

use guide_rs::*;
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Deserialize)]
struct Manifest {
    release: String,
    algorithm_version: String,
    families: BTreeMap<String, Family>,
}

#[derive(Deserialize)]
struct Family {
    rows: Vec<Vec<Value>>,
    exposure: Option<Vec<f64>>,
    predictions: Vec<Prediction>,
}

// Not exact equality: the log and exponential links can differ by a rounding
// error between platform math libraries.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

fn same(a: &Prediction, b: &Prediction) -> bool {
    match (a, b) {
        (Prediction::Regression { value: a }, Prediction::Regression { value: b }) => close(*a, *b),
        (
            Prediction::Quantiles { lower: a, upper: c },
            Prediction::Quantiles { lower: b, upper: d },
        ) => close(*a, *b) && close(*c, *d),
        (
            Prediction::Classification {
                label: a,
                probabilities: c,
            },
            Prediction::Classification {
                label: b,
                probabilities: d,
            },
        ) => {
            a == b
                && c.len() == d.len()
                && c.iter().zip(d).all(|(c, d)| c.0 == d.0 && close(c.1, d.1))
        }
        _ => false,
    }
}

fn assert_predictions(model: &FittedModel, family: &Family, context: &str) {
    let predicted = model
        .predict_with_offsets(&family.rows, family.exposure.as_deref())
        .unwrap_or_else(|e| panic!("{context}: {e}"));
    assert_eq!(predicted.len(), family.predictions.len(), "{context}");
    for (i, (a, b)) in predicted.iter().zip(&family.predictions).enumerate() {
        assert!(same(a, b), "{context}, row {i}: {a:?} against {b:?}");
    }
}

#[test]
fn saved_models_from_published_releases_keep_their_predictions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/saved-models");
    let mut models = 0;
    for release in fs::read_dir(&root).unwrap() {
        let release = release.unwrap().path();
        let manifest: Manifest =
            serde_json::from_str(&fs::read_to_string(release.join("manifest.json")).unwrap())
                .unwrap();
        for (name, family) in &manifest.families {
            let context = format!("{} {name}", manifest.release);
            let path = release.join("models").join(format!("{name}.json"));
            let model = FittedModel::from_json(&fs::read_to_string(path).unwrap())
                .unwrap_or_else(|e| panic!("{context}: {e}"));
            assert_eq!(model.algorithm_version, manifest.algorithm_version);
            assert_predictions(&model, family, &context);
            let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
            assert_predictions(&restored, family, &format!("{context}, resaved"));
            models += 1;
        }
    }
    assert!(models >= 19, "found only {models} frozen models");
}
