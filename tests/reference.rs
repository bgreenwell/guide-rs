//! Frozen outputs from real GUIDE executions; no binary needed to run the tests.
use guide_rs::*;
use std::{fs, path::Path};

// Independently enumerate held-out predictions for the two shallow, four-region
// counterexamples. This uses the explicit partitions/means, not the pruning
// implementation. See docs/native-cv-difference.md for the native node-order flaw.
fn cv_counterexample(tied: bool, native: bool) -> (f64, f64) {
    let levels: [f64; 4] = if tied {
        [0.0, 2.0, 8.0, 10.0]
    } else {
        [0.0, 1.0, 7.0, 10.0]
    };
    let mut losses = vec![];
    for held in 0..24 {
        let y = levels[held / 6];
        let prediction = if native && (!tied || held < 12) {
            // GUIDE leaves all four leaves when the cheaper left branch is
            // ignored. A held-out boundary belongs to the preceding region.
            if [6, 12, 18].contains(&held) {
                levels[held / 6 - 1]
            } else {
                y
            }
        } else if held == 12 {
            // The root cut becomes 12 when x=12 is held out.
            if tied {
                levels[1]
            } else {
                (levels[0] + levels[1]) / 2.0
            }
        } else if tied || held < 12 {
            let first = if held < 12 { 0 } else { 2 };
            (6.0 * (levels[first] + levels[first + 1]) - y) / 11.0
        } else if held == 18 {
            levels[2]
        } else {
            y
        };
        losses.push((y - prediction).powi(2));
    }
    let mean = losses.iter().sum::<f64>() / 24.0;
    let se = (losses.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (24.0 * 24.0)).sqrt();
    (mean, se)
}

fn compare_node(
    node: &Node,
    native_id: usize,
    result: &serde_json::Value,
    data: &Dataset,
    tolerance: f64,
) -> std::result::Result<(), String> {
    if let Some(expected) = result["leaf_coefficients"][native_id.to_string()].as_object() {
        let mut actual = std::collections::BTreeMap::new();
        match &node.model {
            LeafModel::Constant { value } => {
                actual.insert("Constant".to_owned(), *value);
            }
            LeafModel::Linear {
                intercept,
                coefficients,
                terms,
                converged,
                ..
            } => {
                assert!(*converged);
                let mut constant = *intercept;
                for (term, beta) in terms.iter().zip(coefficients) {
                    assert_eq!(term.power, 1);
                    assert!(term.category.is_none());
                    let name = format!(
                        "{}{}",
                        data.schema[term.feature].name,
                        if term.missing_indicator { ".NA" } else { "" }
                    );
                    actual.insert(name, beta / term.scale);
                    constant -= beta * term.center / term.scale;
                }
                actual.insert("Constant".to_owned(), constant);
            }
            _ => return Err("unexpected multiple-linear leaf".into()),
        }
        if actual.len() != expected.len() {
            return Err(format!(
                "node {native_id}: fitted coefficient count differs"
            ));
        }
        for (name, reference) in expected {
            let coefficient = actual
                .get(name)
                .ok_or_else(|| format!("missing fitted term {name}"))?;
            let wanted = reference["value"].as_f64().unwrap();
            if (coefficient - wanted).abs() > reference["tolerance"].as_f64().unwrap() {
                return Err(format!(
                    "node {native_id}: {name}={coefficient}, expected {wanted}"
                ));
            }
        }
    }
    let expected = result["splits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["node_id"].as_u64() == Some(native_id as u64));
    match (&node.split, expected) {
        (None, None) => Ok(()),
        (Some(actual), Some(expected)) => {
            let feature = &data.schema[actual.feature()].name;
            if expected["feature"].as_str() != Some(feature) {
                return Err(format!(
                    "node {native_id}: feature {feature}, expected {}",
                    expected["feature"]
                ));
            }
            let same = match actual {
                Split::Numeric {
                    threshold,
                    missing_left,
                    ..
                } => {
                    expected["kind"] == "numeric"
                        && (threshold - expected["threshold"].as_f64().unwrap()).abs() <= tolerance
                        && Some(*missing_left) == expected["missing_left"].as_bool()
                }
                Split::Categorical {
                    left_categories,
                    missing_left,
                    unknown_left,
                    ..
                } => {
                    let mut actual = left_categories.clone();
                    actual.sort();
                    let mut wanted: Vec<_> = expected["left_categories"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned())
                        .collect();
                    wanted.sort();
                    expected["kind"] == "categorical"
                        && actual == wanted
                        && Some(*missing_left) == expected["missing_left"].as_bool()
                        && Some(*unknown_left) == expected["unknown_left"].as_bool()
                }
                _ => false,
            };
            if !same {
                return Err(format!(
                    "node {native_id}: split {actual:?}, expected {expected}"
                ));
            }
            compare_node(
                node.left.as_ref().unwrap(),
                native_id * 2,
                result,
                data,
                tolerance,
            )?;
            compare_node(
                node.right.as_ref().unwrap(),
                native_id * 2 + 1,
                result,
                data,
                tolerance,
            )
        }
        _ => Err(format!("node {native_id}: terminal/internal node differs")),
    }
}

fn compare_core_case(path: &Path) -> std::result::Result<(), String> {
    let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
    let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
    let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
    assert_eq!(manifest["completed"], true);
    assert_eq!(manifest["returncode"], 0);
    assert_eq!(
        manifest["settings"],
        serde_json::to_value(&options).unwrap()
    );
    let result = &manifest["result"];
    let fitted = FittedModel::fit(&data, &options);
    if result["outcome"] == "rejected" {
        return match fitted {
            Err(e) if result["error_code"] == "single_class" && e.0.contains("at least two") => {
                Ok(())
            }
            other => Err(format!(
                "expected single-class input rejection, got {other:?}"
            )),
        };
    }
    let mut model = fitted.map_err(|e| e.to_string())?;
    let orientation_cv = if manifest["case"] == "orientation-loocv" {
        let file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/orientation/diagnostics/categorical-cv-difference.json");
        let evidence: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(file).unwrap()).unwrap();
        assert_eq!(evidence["completed"], true);
        Some(evidence)
    } else {
        None
    };
    if manifest["case"] == "categorical-linear-split-and-fit" {
        // The historical selection difference is resolved: a sole zero-score
        // candidate does not authorize a split search. No root-only refit or
        // structure exception is needed to compare this frozen native case.
        assert_eq!(result["leaf_count"], 1);
        assert_eq!(model.trees[0].root.leaves(), 1);
        assert_eq!(model.trees[0].root.diagnostics.len(), 1);
        assert!(model.trees[0].root.diagnostics[0].p_value > 0.99);
        assert!((model.trees[0].root.risk - 28750.0 / 96.0).abs() < 1e-9);
    }
    let qr_evidence = if manifest["case"] == "linear-policy-two-missing-p4-n9" {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/linear-policy/diagnostics/native-qr-difference.json");
        let evidence: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(evidence["completed"], true);
        let actual = model.predict(&data.rows).unwrap();
        let mut native_sse = 0.0;
        let mut correct_sse = 0.0;
        let mut native_flag_residual = 0.0;
        let mut correct_flag_residual = 0.0;
        for (i, prediction) in actual.iter().enumerate() {
            let Prediction::Regression { value } = prediction else {
                panic!()
            };
            let y = data.targets[i].number().unwrap();
            let native = result["predictions"][i].as_f64().unwrap();
            assert_eq!(native, evidence["native_predictions"][i].as_f64().unwrap());
            native_sse += (y - native).powi(2);
            correct_sse += (y - value).powi(2);
            if data.rows[i][1].is_missing() {
                native_flag_residual += y - native;
                correct_flag_residual += y - value;
            }
        }
        assert!((native_sse - 3.25).abs() < 1e-9);
        assert!((correct_sse - 2.70).abs() < 1e-9);
        assert!((native_flag_residual + 0.5).abs() < 1e-9);
        assert!(correct_flag_residual.abs() < 1e-9);
        Some(evidence)
    } else {
        None
    };
    let cv_evidence = match manifest["case"].as_str() {
        Some(name @ ("linear-multilevel-loocv" | "linear-multilevel-loocv-weighted")) => {
            let file = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/linear-tree/diagnostics/multilevel-cv-oracle.json");
            let evidence: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(file).unwrap()).unwrap();
            assert_eq!(evidence["completed"], true);
            Some(evidence["cases"][name].clone())
        }
        _ => None,
    };
    if let Some(scores) = result["pruning_scores"].as_array() {
        if scores.len() != model.report.pruning_scores.len() {
            return Err(format!(
                "{} pruning stages, expected {}",
                model.report.pruning_scores.len(),
                scores.len()
            ));
        }
        for (stage, (actual, expected)) in
            model.report.pruning_scores.iter().zip(scores).enumerate()
        {
            if let Some(evidence) = &orientation_cv {
                assert_eq!(actual.leaves as u64, expected["leaves"].as_u64().unwrap());
                for (value, offset) in [(actual.mean_loss, 0), (actual.standard_error, 2)] {
                    let correct = evidence["correct_scores"][stage + offset].as_f64().unwrap();
                    assert!((value - correct).abs() < 2e-10 * (1.0 + correct.abs()));
                }
                let version = manifest["requested_version"].as_str().unwrap();
                assert_eq!(
                    expected,
                    &evidence["references"][version]["native_cv_scores"][stage]
                );
                if stage == 0 {
                    assert!((actual.mean_loss - 40.28454952128264).abs() < 1e-9);
                    assert!((expected["mean_loss"].as_f64().unwrap() - 94.72).abs() < 1e-9);
                }
                continue;
            }
            if let Some(evidence) = &cv_evidence {
                let independent = &evidence["scores"][stage];
                assert_eq!(
                    actual.leaves as u64,
                    independent["leaves"]
                        .as_u64()
                        .unwrap_or_else(|| independent["leaves"].as_f64().unwrap() as u64)
                );
                for (actual, field, simulated) in [
                    (actual.mean_loss, "mean_loss", "native_scan_mean"),
                    (actual.standard_error, "standard_error", "native_scan_se"),
                ] {
                    let correct = independent[field].as_f64().unwrap();
                    assert!((actual - correct).abs() < 2e-10 * (1.0 + correct.abs()));
                    assert!(
                        (expected[field].as_f64().unwrap()
                            - independent[simulated].as_f64().unwrap())
                        .abs()
                            <= expected[format!("{field}_tolerance")].as_f64().unwrap()
                    );
                }
                if stage == 1 {
                    assert!(
                        (actual.mean_loss - expected["mean_loss"].as_f64().unwrap()).abs() > 1.0
                    );
                }
                continue;
            }
            let difference = match (manifest["case"].as_str(), stage) {
                (Some("ls-loocv-tied-path"), 1) => Some(true),
                (Some("ls-loocv-path"), 1) => Some(false),
                _ => None,
            };
            if let Some(tied) = difference {
                // Deliberate, user-approved divergence: keep correct weakest-link
                // CV, and verify BOTH the native flaw and the correct result.
                let correct = cv_counterexample(tied, false);
                let native = cv_counterexample(tied, true);
                assert_eq!(actual.leaves, 3);
                assert!((actual.mean_loss - correct.0).abs() < 1e-12);
                assert!((actual.standard_error - correct.1).abs() < 1e-12);
                assert!(
                    (expected["mean_loss"].as_f64().unwrap() - native.0).abs()
                        <= expected["mean_loss_tolerance"].as_f64().unwrap()
                );
                assert!(
                    (expected["standard_error"].as_f64().unwrap() - native.1).abs()
                        <= expected["standard_error_tolerance"].as_f64().unwrap()
                );
                assert!((correct.0 - native.0).abs() > 0.1);
                continue;
            }
            if Some(actual.leaves as u64) != expected["leaves"].as_u64()
                || (actual.mean_loss - expected["mean_loss"].as_f64().unwrap()).abs()
                    > expected["mean_loss_tolerance"].as_f64().unwrap()
                || (actual.standard_error - expected["standard_error"].as_f64().unwrap()).abs()
                    > expected["standard_error_tolerance"].as_f64().unwrap()
            {
                return Err(format!("pruning stage {actual:?}, expected {expected}"));
            }
        }
    }
    if let Some(evidence) = &cv_evidence {
        assert_eq!(evidence["correct_selected_leaves"], 4);
        assert_eq!(evidence["native_scan_selected_leaves"], 3);
        assert_eq!(model.trees[0].root.leaves(), 4);
        assert_eq!(result["leaf_count"], 3);
        let predictions = model.predict(&data.rows).unwrap();
        for (i, prediction) in predictions.iter().enumerate() {
            let Prediction::Regression { value } = prediction else {
                panic!()
            };
            let expected = &evidence["trees"]["4"][i];
            assert!((value - expected["prediction"].as_f64().unwrap()).abs() < 1e-9);
            let mut node = &model.trees[0].root;
            let mut id = 1;
            while let Some(split) = &node.split {
                if split.goes_left(&data.rows[i]) {
                    id *= 2;
                    node = node.left.as_ref().unwrap();
                } else {
                    id = 2 * id + 1;
                    node = node.right.as_ref().unwrap();
                }
            }
            assert_eq!(id as f64, expected["node"].as_f64().unwrap());
        }
        let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(predictions).unwrap(),
            serde_json::to_value(restored.predict(&data.rows).unwrap()).unwrap()
        );
        // Compare the preserved native final three-leaf tree separately. Correct
        // Rust CV selects four leaves; fixed-size pruning reconstructs the native
        // final tree without adopting its erroneous CV candidate scan.
        let mut native_size_options = options.clone();
        native_size_options.pruning = Pruning::FixedLeaves { max_leaves: 3 };
        model = FittedModel::fit(&data, &native_size_options).unwrap();
    }
    if let Some(path) = result["pruning_path"].as_array() {
        let independent_path = if matches!(
            manifest["case"].as_str(),
            Some("linear-multilevel-fixed-3" | "linear-multilevel-fixed-5")
        ) {
            let file = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/linear-tree/diagnostics/multilevel-cv-oracle.json");
            let record: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(file).unwrap()).unwrap();
            Some(record["cases"]["linear-multilevel-fixed-3"].clone())
        } else {
            None
        };
        assert_eq!(
            model.report.unpruned_leaves as u64,
            path[0]["leaves"].as_u64().unwrap()
        );
        let Pruning::FixedLeaves { max_leaves } = options.pruning else {
            panic!()
        };
        let selected = path
            .iter()
            .find(|row| row["leaves"].as_u64().unwrap() <= max_leaves as u64)
            .unwrap();
        assert!(
            (model.report.selected_alpha - selected["alpha"].as_f64().unwrap()).abs()
                <= selected["alpha_tolerance"].as_f64().unwrap()
        );
        for row in path {
            let mut size_options = options.clone();
            size_options.pruning = Pruning::FixedLeaves {
                max_leaves: row["leaves"].as_u64().unwrap() as usize,
            };
            let m = FittedModel::fit(&data, &size_options).unwrap();
            assert_eq!(
                m.report.fitted_leaves as u64,
                row["leaves"].as_u64().unwrap()
            );
            assert!(
                (m.report.selected_alpha - row["alpha"].as_f64().unwrap()).abs()
                    <= row["alpha_tolerance"].as_f64().unwrap()
            );
            if let Some(independent) = &independent_path {
                let leaves = row["leaves"].as_u64().unwrap().to_string();
                let reference = &independent["trees"][&leaves];
                for (i, prediction) in m.predict(&data.rows).unwrap().iter().enumerate() {
                    let Prediction::Regression { value } = prediction else {
                        panic!()
                    };
                    assert!((value - reference[i]["prediction"].as_f64().unwrap()).abs() < 1e-9);
                    let mut node = &m.trees[0].root;
                    let mut id = 1;
                    while let Some(split) = &node.split {
                        if split.goes_left(&data.rows[i]) {
                            id *= 2;
                            node = node.left.as_ref().unwrap();
                        } else {
                            id = 2 * id + 1;
                            node = node.right.as_ref().unwrap();
                        }
                    }
                    assert_eq!(id as f64, reference[i]["node"].as_f64().unwrap());
                }
            }
        }
    }
    if model.trees[0].root.leaves() as u64 != result["leaf_count"].as_u64().unwrap() {
        return Err(format!(
            "{} leaves, expected {}",
            model.trees[0].root.leaves(),
            result["leaf_count"]
        ));
    }
    compare_node(
        &model.trees[0].root,
        1,
        result,
        &data,
        manifest["tolerances"]["cutpoint_absolute"]
            .as_f64()
            .unwrap(),
    )?;
    let predictions = model.predict(&data.rows).unwrap();
    if predictions.len() != result["predictions"].as_array().unwrap().len() {
        return Err("prediction count differs".into());
    }
    for (i, prediction) in predictions.iter().enumerate() {
        let mut node = &model.trees[0].root;
        let mut id = 1;
        while let Some(split) = &node.split {
            if split.goes_left(&data.rows[i]) {
                id *= 2;
                node = node.left.as_ref().unwrap();
            } else {
                id = 2 * id + 1;
                node = node.right.as_ref().unwrap();
            }
        }
        if result["terminal_node_ids"][i].as_u64() != Some(id) {
            return Err(format!(
                "row {i}: leaf {id}, expected {}",
                result["terminal_node_ids"][i]
            ));
        }
        let expected = &result["predictions"][i];
        match prediction {
            Prediction::Regression { value } => {
                let wanted = qr_evidence
                    .as_ref()
                    .map(|e| e["correct_predictions"][i].as_f64().unwrap())
                    .unwrap_or_else(|| expected.as_f64().unwrap());
                let atol = manifest["tolerances"]["prediction_absolute"]
                    .as_f64()
                    .unwrap();
                let rtol = manifest["tolerances"]["prediction_relative"]
                    .as_f64()
                    .unwrap();
                let tolerance = if qr_evidence.is_some() {
                    1e-9
                } else {
                    result["prediction_tolerances"][i]
                        .as_f64()
                        .unwrap_or(atol + rtol * wanted.abs())
                };
                if (value - wanted).abs() > tolerance {
                    return Err(format!("row {i}: prediction {value}, expected {wanted}"));
                }
            }
            Prediction::Classification {
                label,
                probabilities,
            } => {
                if expected.as_str() != Some(label) {
                    return Err(format!("row {i}: class {label}, expected {expected}"));
                }
                for (j, (class, p)) in probabilities.iter().enumerate() {
                    let wanted = result["probabilities"][i][j].as_f64().unwrap();
                    if result["classes"][j].as_str() != Some(class)
                        || (p - wanted).abs()
                            > manifest["tolerances"]["probability_absolute"]
                                .as_f64()
                                .unwrap()
                    {
                        return Err(format!("row {i}: P({class})={p}, expected {wanted}"));
                    }
                }
            }
            _ => return Err("unexpected prediction type".into()),
        }
    }
    let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(restored.predict(&data.rows).unwrap()).unwrap(),
        serde_json::to_value(predictions).unwrap()
    );
    Ok(())
}

#[test]
fn guide_46x_categorical_orientation_and_unseen_routes() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/orientation");
    for version in ["46.3", "46.2"] {
        let paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(paths.len(), 39);
        let mut failures = vec![];
        for path in paths {
            if let Err(error) = compare_core_case(&path) {
                failures.push(format!("{}: {error}", path.display()));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

#[test]
fn guide_46x_categorical_multiple_linear_designs() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/categorical-linear");
    for version in ["46.3", "46.2"] {
        let paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(paths.len(), 41);
        let mut failures = vec![];
        for path in paths {
            if let Err(error) = compare_core_case(&path) {
                failures.push(format!("{}: {error}", path.display()));
            }
            let manifest: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(path.join("manifest.json")).unwrap())
                    .unwrap();
            if let Some(minimum) = manifest["result"]["nominal_minimum"].as_u64() {
                let data: Dataset =
                    serde_json::from_str(&fs::read_to_string(path.join("dataset.json")).unwrap())
                        .unwrap();
                let options: FitOptions =
                    serde_json::from_str(&fs::read_to_string(path.join("options.json")).unwrap())
                        .unwrap();
                let model = FittedModel::fit(&data, &options).unwrap();
                assert_eq!(
                    matches!(model.trees[0].root.model, LeafModel::Constant { .. }),
                    data.active_rows().len() < minimum as usize
                );
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

#[test]
fn guide_46x_numeric_linear_fallbacks_and_prediction_truncation() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linear-policy");
    for version in ["46.3", "46.2"] {
        let paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(paths.len(), 47);
        for path in paths {
            if let Err(error) = compare_core_case(&path) {
                panic!("{}: {error}", path.display());
            }
            let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
            let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
            let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
            let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
            let model = FittedModel::fit(&data, &options).unwrap();
            if let Some(minimum) = manifest["result"]["nominal_minimum"].as_u64() {
                let should_fallback = data.active_rows().len() < minimum as usize;
                assert_eq!(
                    matches!(model.trees[0].root.model, LeafModel::Constant { .. }),
                    should_fallback
                );
                assert_eq!(
                    manifest["result"]["expected_fit"] == "constant",
                    should_fallback
                );
            }
        }
    }
}

#[test]
fn guide_46x_multiple_linear_tree_growth_and_loocv() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linear-tree");
    for version in ["46.3", "46.2"] {
        let paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("linear-tree-")
            })
            .collect();
        assert_eq!(paths.len(), 5);
        for path in paths {
            if let Err(error) = compare_core_case(&path) {
                panic!("{}: {error}", path.display());
            }
        }
    }
}

#[test]
fn guide_46x_multilevel_multiple_linear_growth_and_pruning() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linear-tree");
    let mut differences = vec![];
    for version in ["46.3", "46.2"] {
        let paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("linear-multilevel-")
            })
            .collect();
        assert_eq!(paths.len(), 12);
        for path in paths {
            if let Err(error) = compare_core_case(&path) {
                differences.push(format!("{}: {error}", path.display()));
            }
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

#[test]
fn probability_correction_is_conditional_and_saved_models_keep_their_values() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/reference/guide-46.3/class-balanced");
    let data: Dataset =
        serde_json::from_str(&fs::read_to_string(base.join("dataset.json")).unwrap()).unwrap();
    let mut options: FitOptions =
        serde_json::from_str(&fs::read_to_string(base.join("options.json")).unwrap()).unwrap();
    options.max_depth = 0;
    let root = FittedModel::fit(&data, &options).unwrap();
    let LeafModel::Classification { probabilities, .. } = &root.trees[0].root.model else {
        panic!()
    };
    assert_eq!(probabilities, &[0.5, 0.5]);
    options.max_depth = 2;
    let model = FittedModel::fit(&data, &options).unwrap();
    assert_eq!(
        model.trees[0].root.leaves(),
        2,
        "pure nodes must stop before smoothing"
    );
    assert_eq!(model.trees[0].root.left.as_ref().unwrap().impurity, 0.0);
    let LeafModel::Classification { probabilities, .. } =
        &model.trees[0].root.left.as_ref().unwrap().model
    else {
        panic!()
    };
    assert_eq!(probabilities, &[(80.0 + 0.5) / 81.0, 0.5 / 81.0]);

    // A saved reference1 model retains its empirical probabilities on load.
    let mut legacy = serde_json::to_value(&model).unwrap();
    legacy["algorithm_version"] = "guide-rs-0.1-experimental-reference1".into();
    legacy["trees"][0]["root"]["left"]["model"]["probabilities"] = serde_json::json!([1.0, 0.0]);
    legacy["trees"][0]["root"]["right"]["model"]["probabilities"] = serde_json::json!([0.0, 1.0]);
    let restored = FittedModel::from_json(&legacy.to_string()).unwrap();
    let Prediction::Classification { probabilities, .. } =
        &restored.predict(&data.rows).unwrap()[0]
    else {
        panic!()
    };
    assert_eq!(probabilities, &vec![("a".into(), 1.0), ("b".into(), 0.0)]);
}

#[test]
fn guide_46x_constant_core_corpus() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference");
    let mut errors = vec![];
    let mut exercised = 0;
    for version in ["46.3", "46.2"] {
        let mut paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|p| p.unwrap().path())
            .collect();
        paths.sort();
        for path in paths {
            let manifest: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(path.join("manifest.json")).unwrap())
                    .unwrap();
            if manifest["fixture_format"] != 2 {
                continue;
            }
            assert_eq!(manifest["result"]["version"], version);
            exercised += 1;
            if let Err(e) = compare_core_case(&path) {
                errors.push(format!(
                    "{version}/{}: {e}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }
    assert!(exercised > 0);
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn guide_46x_numeric_stumps_match_structure_predictions_and_missing_direction() {
    for version in ["46.2", "46.3"] {
        for size in [5, 10, 15] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "tests/fixtures/reference/guide-{version}/ls-stump-{size}"
            ));
            let manifest: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(path.join("manifest.json")).unwrap())
                    .unwrap();
            assert_eq!(manifest["completed"], true);
            assert_eq!(manifest["returncode"], 0);
            assert_eq!(manifest["result"]["version"], version);
            let csv = fs::read_to_string(path.join("data.csv")).unwrap();
            let d = dataset_from_csv(
                &csv,
                &CsvOptions {
                    target: "y".into(),
                    ..Default::default()
                },
                &Task::default(),
            )
            .unwrap();
            let options = FitOptions {
                max_depth: 1,
                min_samples_leaf: 2,
                max_split_candidates: 0,
                pruning: Pruning::None,
                ..Default::default()
            };
            let model = FittedModel::fit(&d, &options).unwrap();
            let result = &manifest["result"];
            assert_eq!(
                model.trees[0].root.leaves(),
                result["leaf_count"].as_u64().unwrap() as usize
            );
            let Split::Numeric {
                feature,
                threshold,
                missing_left,
            } = model.trees[0].root.split.as_ref().unwrap()
            else {
                panic!("expected numeric root")
            };
            assert_eq!(
                d.schema[*feature].name,
                result["root_feature"].as_str().unwrap()
            );
            assert!(
                (threshold - result["root_threshold"].as_f64().unwrap()).abs()
                    <= manifest["tolerances"]["cutpoint_absolute"]
                        .as_f64()
                        .unwrap()
            );
            assert_eq!(
                *missing_left,
                result["missing_left"].as_bool().unwrap(),
                "GUIDE {version}, left_size={size}"
            );
            for (p, expected) in model
                .predict(&d.rows)
                .unwrap()
                .iter()
                .zip(result["predictions"].as_array().unwrap())
            {
                let Prediction::Regression { value } = p else {
                    panic!()
                };
                assert!(
                    (value - expected.as_f64().unwrap()).abs()
                        <= manifest["tolerances"]["prediction_absolute"]
                            .as_f64()
                            .unwrap()
                );
            }
            let Prediction::Regression { value } =
                model.predict(&[vec![Value::Missing]]).unwrap()[0]
            else {
                panic!()
            };
            assert_eq!(value, 8.0);
        }
    }
}
