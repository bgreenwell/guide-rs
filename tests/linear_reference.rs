//! Native root-only least-squares fits, with explicit fallback counterexamples.
use guide_rs::*;
use std::{collections::BTreeMap, fs, path::Path};

fn numeric(model: &FittedModel, data: &Dataset) -> Vec<f64> {
    model
        .predict(&data.rows)
        .unwrap()
        .into_iter()
        .map(|p| match p {
            Prediction::Regression { value } => value,
            _ => panic!("expected regression"),
        })
        .collect()
}

// Undo the serialized centering/scaling to compare coefficients in native units.
fn coefficients(model: &LeafModel, data: &Dataset) -> (f64, BTreeMap<String, f64>) {
    match model {
        LeafModel::Constant { value } => (*value, BTreeMap::new()),
        LeafModel::Linear {
            intercept,
            coefficients,
            terms,
            converged,
            ..
        } => {
            assert!(*converged);
            let mut intercept = *intercept;
            let mut result = BTreeMap::new();
            for (term, beta) in terms.iter().zip(coefficients) {
                assert_eq!(term.power, 1);
                assert!(term.category.is_none());
                let name = format!(
                    "{}{}",
                    data.schema[term.feature].name,
                    if term.missing_indicator { ".NA" } else { "" }
                );
                let value = beta / term.scale;
                intercept -= value * term.center;
                result.insert(name, value);
            }
            (intercept, result)
        }
        _ => panic!("unexpected leaf model"),
    }
}

#[test]
fn native_linear_coefficients_predictions_and_documented_fallback_differences() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linear");
    for version in ["46.3", "46.2"] {
        let mut paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|p| p.unwrap().path())
            .collect();
        paths.sort();
        assert_eq!(paths.len(), 15);
        let mut exceptions = 0;
        for path in paths {
            let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
            let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
            let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
            let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
            let name = manifest["case"].as_str().unwrap();
            assert_eq!(manifest["completed"], true);
            assert_eq!(manifest["returncode"], 0);
            assert_eq!(manifest["result"]["version"], version);
            assert_eq!(
                manifest["settings"],
                serde_json::to_value(&options).unwrap()
            );
            let expected = &manifest["result"];
            let model = FittedModel::fit(&data, &options).unwrap();
            assert!(model.trees[0].root.split.is_none());
            let predictions = numeric(&model, &data);
            let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
            assert_eq!(predictions, numeric(&restored, &data));
            let (intercept, coefficients) = coefficients(&model.trees[0].root.model, &data);
            // Historical metadata is immutable; this formerly divergent case
            // now agrees with the verified numeric multiple-linear fallback.
            if name == "ls-linear-small" {
                assert!(matches!(
                    model.trees[0].root.model,
                    LeafModel::Constant { .. }
                ));
            }
            if let Some(difference) = expected["native_difference"]
                .as_str()
                .filter(|s| *s != "small_multiple_fallback")
            {
                exceptions += 1;
                let mean = data
                    .targets
                    .iter()
                    .map(|v| v.number().unwrap())
                    .sum::<f64>()
                    / data.rows.len() as f64;
                if difference != "grouped_stepwise_selection" {
                    // The native collinear-stepwise fallback remains unverified.
                    // Check Rust's rank-aware fit against analytic OLS.
                    assert!(expected["coefficients"].as_object().unwrap().is_empty());
                    assert!((expected["intercept"].as_f64().unwrap() - mean).abs() < 1e-9);
                }
                match difference {
                    "collinear_stepwise_fallback" => {
                        // Cov(x,y)/Var(x) and mean(y)-slope*mean(x).
                        assert!((intercept + 0.8341463414634146).abs() < 1e-9);
                        let slope = coefficients.get("x").copied().unwrap_or(0.0)
                            + 2.0 * coefficients.get("z").copied().unwrap_or(0.0);
                        assert!((slope - 1.996622889305816).abs() < 1e-9);
                    }
                    "grouped_stepwise_selection" => {
                        // Rust groups x and its flag; native selects x and the
                        // flag without z. This is a selection gap, not a claim
                        // that the native choice is erroneous. The independent
                        // R suite verifies the fit given Rust's selected columns.
                        let native = expected["coefficients"].as_object().unwrap();
                        assert_eq!(
                            native.keys().map(String::as_str).collect::<Vec<_>>(),
                            vec!["x", "x.NA"]
                        );
                        assert!((coefficients["z"] + 2.55464699629).abs() < 1e-9);
                        assert!((coefficients["x"] - 2.00367047737).abs() < 1e-9);
                        assert!((coefficients["x.NA"] - 78.4048327186).abs() < 1e-9);
                    }
                    other => panic!("unrecognized native difference: {other}"),
                }
                assert!(
                    predictions
                        .iter()
                        .zip(expected["predictions"].as_array().unwrap())
                        .any(|(a, b)| (a - b.as_f64().unwrap()).abs() > 1.0)
                );
                continue;
            }
            let expected_intercept = expected["intercept"].as_f64().unwrap();
            assert!(
                (intercept - expected_intercept).abs() < 2e-9 * (1.0 + expected_intercept.abs()),
                "{version} {name}: intercept {intercept} vs {expected_intercept}"
            );
            let native = expected["coefficients"].as_object().unwrap();
            if name == "ls-linear-collinear" {
                // Individual coefficients are not identifiable when z=2*x.
                // QR pivot choices can differ while fitted functions agree on
                // that design. Do not assert prediction parity off its span.
                let slope = coefficients.get("x").copied().unwrap_or(0.0)
                    + 2.0 * coefficients.get("z").copied().unwrap_or(0.0);
                let reference = native.get("x").map(|v| v.as_f64().unwrap()).unwrap_or(0.0)
                    + 2.0 * native.get("z").map(|v| v.as_f64().unwrap()).unwrap_or(0.0);
                assert!((slope - reference).abs() < 1e-9);
            } else {
                for feature in coefficients.keys().chain(native.keys()) {
                    let actual = coefficients.get(feature).copied().unwrap_or(0.0);
                    let reference = native
                        .get(feature)
                        .map(|v| v.as_f64().unwrap())
                        .unwrap_or(0.0);
                    assert!(
                        (actual - reference).abs() < 2e-9 * (1.0 + reference.abs()),
                        "{version} {name}: {feature} = {actual} vs {reference}"
                    );
                }
            }
            for (i, actual) in predictions.iter().enumerate() {
                let reference = expected["predictions"][i].as_f64().unwrap();
                let tolerance = expected["prediction_tolerances"][i].as_f64().unwrap();
                assert!(
                    (actual - reference).abs() <= tolerance,
                    "{version} {name} row {i}"
                );
            }
        }
        assert_eq!(exceptions, 2);
    }
}
