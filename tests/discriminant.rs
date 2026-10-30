use guide_rs::*;

fn iris() -> Dataset {
    let features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
        .iter()
        .map(|n| FeatureSchema {
            name: n.to_string(),
            feature_type: FeatureType::Numeric,
            role: Role::Both,
            flags: None,
        })
        .collect();
    let cfg = CsvOptions {
        target: "Species".into(),
        features,
        ..CsvOptions::default()
    };
    dataset_from_csv(
        include_str!("fixtures/datasets/iris.csv"),
        &cfg,
        &Task::Classification,
    )
    .unwrap()
}

fn lda() -> FitOptions {
    FitOptions {
        task: Task::Classification,
        node_model: NodeModel::Lda,
        max_depth: 5,
        min_samples_leaf: 10,
        pruning: Pruning::None,
        ..FitOptions::default()
    }
}

fn leaves(n: &Node) -> Vec<&Node> {
    match (&n.left, &n.right) {
        (Some(l), Some(r)) => [leaves(l), leaves(r)].concat(),
        _ => vec![n],
    }
}

#[test]
fn lda_trees_split_on_discriminant_coordinates_and_round_trip() {
    let data = iris();
    let model = FittedModel::fit(&data, &lda()).unwrap();
    let root = &model.trees[0].root;
    assert!(matches!(root.split, Some(Split::Discriminant { .. })));
    assert!(
        leaves(root)
            .iter()
            .any(|l| matches!(l.model, LeafModel::Discriminant(_)))
    );
    // Children have more than the minimum node size, and nodes with fewer
    // than 2.5 times it do not split.
    fn check(n: &Node, s: usize) {
        if let (Some(l), Some(r)) = (&n.left, &n.right) {
            assert!(2 * n.sample_count >= 5 * s);
            assert!(l.sample_count > s && r.sample_count > s);
            check(l, s);
            check(r, s);
        }
    }
    check(root, 10);
    let predictions = model.predict(&data.rows).unwrap();
    let mut correct = 0;
    for (p, y) in predictions.iter().zip(&data.targets) {
        let Prediction::Classification {
            label,
            probabilities,
        } = p
        else {
            panic!("expected a classification")
        };
        let total: f64 = probabilities.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(probabilities.iter().all(|(_, p)| *p > 0.0));
        correct += usize::from(Some(label.clone()) == y.label());
    }
    assert!(correct >= 145, "{correct} of 150 correct");
    let restored = FittedModel::from_json(&model.to_json().unwrap()).unwrap();
    assert_eq!(restored.predict(&data.rows).unwrap(), predictions);
}

#[test]
fn lda_imputes_missing_values_with_node_means() {
    let mut data = iris();
    for (i, row) in data.rows.iter_mut().enumerate() {
        if i % 9 == 0 {
            row[2] = Value::Missing;
        }
    }
    let model = FittedModel::fit(&data, &lda()).unwrap();
    let mut row = data.rows[0].clone();
    row[2] = Value::Missing;
    assert!(model.predict(&[row]).is_ok());
}

#[test]
fn lda_cross_validation_pruning_runs() {
    let options = FitOptions {
        pruning: Pruning::CrossValidation {
            folds: 5,
            standard_errors: 0.25,
        },
        ..lda()
    };
    let model = FittedModel::fit(&iris(), &options).unwrap();
    assert!(model.report.fitted_leaves >= 1);
    assert!(model.report.fitted_leaves <= model.report.unpruned_leaves);
}

#[test]
fn lda_rejects_unsupported_configurations() {
    let data = iris();
    let fails = |options: FitOptions, data: &Dataset, message: &str| {
        let error = FittedModel::fit(data, &options).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
    };
    fails(
        FitOptions {
            task: Task::Regression {
                loss: Loss::SquaredError,
            },
            ..lda()
        },
        &data,
        "require classification",
    );
    fails(
        FitOptions {
            class_costs: Some(vec![
                vec![0.0, 1.0, 2.0],
                vec![1.0, 0.0, 1.0],
                vec![1.0, 1.0, 0.0],
            ]),
            ..lda()
        },
        &data,
        "unequal class costs",
    );
    fails(
        FitOptions {
            mtry: Some(2),
            ..lda()
        },
        &data,
        "feature sampling",
    );
    let mut split_only = data.clone();
    split_only.schema[0].role = Role::Split;
    fails(lda(), &split_only, "role fit or both");
    let ensemble = FittedModel::fit_ensemble(&data, &lda(), &EnsembleOptions::default());
    assert!(ensemble.unwrap_err().to_string().contains("ensembles"));
}

/// Refits a frozen native LDA tree case and compares training predictions and
/// the partition of rows into leaves.
fn compare_tree(dir: &std::path::Path) -> (bool, bool) {
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join(name)).unwrap()).unwrap()
    };
    let case = read("case.json");
    let result = &read("manifest.json")["result"];
    let names: Vec<String> = case["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap().to_string())
        .collect();
    let categorical = |n: &str| {
        case["categorical"]
            .as_array()
            .is_some_and(|c| c.iter().any(|v| v == n))
    };
    let data = Dataset {
        schema: names
            .iter()
            .map(|n| FeatureSchema {
                name: n.clone(),
                feature_type: if categorical(n) {
                    FeatureType::Categorical
                } else {
                    FeatureType::Numeric
                },
                role: Role::Both,
                flags: None,
            })
            .collect(),
        rows: case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => Value::Text(s.clone()),
                        v => v.as_f64().map_or(Value::Missing, Value::Number),
                    })
                    .collect()
            })
            .collect(),
        targets: case["targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| Value::Text(t.as_str().unwrap().into()))
            .collect(),
        weights: vec![],
        offsets: vec![],
    };
    let options = FitOptions {
        max_depth: case["depth"].as_u64().unwrap() as usize,
        min_samples_leaf: case["min_node"].as_u64().unwrap() as usize,
        ..lda()
    };
    let model = FittedModel::fit(&data, &options).unwrap();
    let predicted: Vec<String> = model
        .predict(&data.rows)
        .unwrap()
        .into_iter()
        .map(|p| match p {
            Prediction::Classification { label, .. } => label,
            _ => unreachable!(),
        })
        .collect();
    let native: Vec<&str> = result["predicted"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    // Group rows by leaf in both trees; node numbering differs.
    let group = |keys: Vec<usize>| {
        let mut sets: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (i, k) in keys.into_iter().enumerate() {
            sets.entry(k).or_default().push(i);
        }
        let mut v: Vec<Vec<usize>> = sets.into_values().collect();
        v.sort();
        v
    };
    let ours = group(
        data.rows
            .iter()
            .map(|r| model.trees[0].root.leaf(r).id)
            .collect(),
    );
    let theirs = group(
        result["leaves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_u64().unwrap() as usize)
            .collect(),
    );
    (ours == theirs, predicted == native)
}

#[test]
fn lda_trees_reproduce_native_captures() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/discriminant");
    for version in ["guide-46.2", "guide-46.3"] {
        let mut dirs: Vec<_> = std::fs::read_dir(root.join(version))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("lda-tree-")
            })
            .collect();
        dirs.sort();
        assert_eq!(dirs.len(), 12);
        for dir in dirs {
            let (leaves, predictions) = compare_tree(&dir);
            assert!(leaves, "{version} {dir:?}: leaf partition");
            assert!(predictions, "{version} {dir:?}: training predictions");
        }
    }
}
