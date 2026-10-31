//! Frozen GUIDE 46.2/46.3 trees with missing-value flag (M) variables.
use guide_rs::*;
use std::{collections::BTreeMap, fs, path::Path};

fn compare(path: &Path) -> std::result::Result<(), String> {
    let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
    let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
    let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
    assert_eq!(manifest["completed"], true);
    let native = &manifest["result"];
    let model = FittedModel::fit(&data, &options).map_err(|e| e.to_string())?;
    let tree = &model.trees[0];
    let predictions = model.predict(&data.rows).map_err(|e| e.to_string())?;
    let leaves: Vec<usize> = data.rows.iter().map(|r| tree.root.leaf(r).id).collect();
    if native["native_difference"] == "first-code-anchored-subsets" {
        // GUIDE only tries flag code subsets containing the node's first
        // code, so it misses the best split here. Rust searches every subset;
        // require a strictly lower training loss.
        let sse = |values: &[f64]| -> f64 {
            data.targets
                .iter()
                .zip(values)
                .filter_map(|(t, p)| t.number().map(|y| (y - p).powi(2)))
                .sum()
        };
        let native_predictions: Vec<f64> = native["predictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let ours: Vec<f64> = predictions
            .iter()
            .map(|p| match p {
                Prediction::Regression { value } => *value,
                other => panic!("{other:?}"),
            })
            .collect();
        return if sse(&ours) + 1e-6 < sse(&native_predictions) {
            Ok(())
        } else {
            Err(format!(
                "expected a strictly better split, got SSE {} vs native {}",
                sse(&ours),
                sse(&native_predictions)
            ))
        };
    }
    if tree.root.leaves() as u64 != native["leaf_count"].as_u64().unwrap() {
        return Err(format!(
            "{} leaves, expected {}",
            tree.root.leaves(),
            native["leaf_count"]
        ));
    }
    let mut mapping = BTreeMap::new();
    let mut reverse = BTreeMap::new();
    for (i, node) in native["nodes"].as_array().unwrap().iter().enumerate() {
        let node = node.as_u64().unwrap();
        if *mapping.entry(node).or_insert(leaves[i]) != leaves[i]
            || *reverse.entry(leaves[i]).or_insert(node) != node
        {
            return Err(format!("row {}: terminal partition differs", i + 1));
        }
        let expected = &native["predictions"][i];
        let ok = match &predictions[i] {
            Prediction::Regression { value } => {
                let e = expected.as_f64().unwrap();
                (value - e).abs() <= 5e-5 + 5e-6 * e.abs()
            }
            Prediction::Classification { label, .. } => expected.as_str() == Some(label),
            Prediction::Quantiles { .. } => false,
        };
        if !ok {
            return Err(format!(
                "row {}: prediction {:?}, expected {expected}",
                i + 1,
                predictions[i]
            ));
        }
    }
    let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
    if restored.predict(&data.rows).unwrap() != predictions {
        return Err("saved model changed predictions".into());
    }
    Ok(())
}

#[test]
fn guide_46x_missing_value_flags() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/missing-flag");
    let mut errors = vec![];
    let mut exercised = 0;
    for version in ["46.3", "46.2"] {
        let mut paths: Vec<_> = fs::read_dir(base.join(format!("guide-{version}")))
            .unwrap()
            .map(|p| p.unwrap().path())
            .collect();
        paths.sort();
        for path in paths {
            exercised += 1;
            if let Err(e) = compare(&path) {
                errors.push(format!(
                    "{version}/{}: {e}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }
    assert_eq!(exercised, 40);
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn flags_become_categorical_features_for_linear_node_models() {
    // GUIDE reports "M variables changed to B" for multiple-linear and
    // stepwise node models: flags are split on and fitted as dummies.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/missing-flag/guide-46.3/missing-flag-numeric-left-A");
    let data: Dataset =
        serde_json::from_str(&fs::read_to_string(path.join("dataset.json")).unwrap()).unwrap();
    let mut data = data;
    data.schema[0].role = Role::Both;
    let flag = data
        .schema
        .iter()
        .position(|s| s.role == Role::Flag)
        .unwrap();
    let options = FitOptions {
        node_model: NodeModel::Linear,
        pruning: Pruning::None,
        ..Default::default()
    };
    let model = FittedModel::fit(&data, &options).unwrap();
    assert_eq!(model.schema[flag].role, Role::Both);
    assert_eq!(model.schema[flag].feature_type, FeatureType::Categorical);
    assert!(model.schema[flag].flags.is_none());
    assert!(model.predict(&data.rows).is_ok());
    let polynomial = FitOptions {
        node_model: NodeModel::Polynomial {
            degree: 1,
            alpha: 0.05,
        },
        ..options.clone()
    };
    let model = FittedModel::fit(&data, &polynomial).unwrap();
    assert_eq!(model.schema[flag].role, Role::Flag);
    let lda = FitOptions {
        task: Task::Classification,
        node_model: NodeModel::Lda,
        ..options
    };
    assert!(FittedModel::fit(&data, &lda).is_err());
}
