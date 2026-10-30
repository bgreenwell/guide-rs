use guide_rs::*;

fn options() -> FitOptions {
    FitOptions {
        pruning: Pruning::None,
        max_depth: 4,
        ..Default::default()
    }
}
fn data(x: Vec<Vec<f64>>, y: Vec<f64>) -> Dataset {
    let p = x.first().map(Vec::len).unwrap_or(0);
    Dataset {
        schema: (0..p)
            .map(|j| FeatureSchema {
                name: format!("x{j}"),
                feature_type: FeatureType::Numeric,
                role: Role::Split,
                flags: None,
            })
            .collect(),
        rows: x
            .into_iter()
            .map(|r| r.into_iter().map(Value::Number).collect())
            .collect(),
        targets: y.into_iter().map(Value::Number).collect(),
        weights: vec![],
        offsets: vec![],
    }
}
fn values(p: Vec<Prediction>) -> Vec<f64> {
    p.into_iter()
        .map(|p| match p {
            Prediction::Regression { value } => value,
            _ => panic!("expected regression"),
        })
        .collect()
}
fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-7, "{a} != {b}");
}

#[test]
fn exact_stump_predictions_and_roundtrip() {
    let d = data(
        (0..20).map(|i| vec![i as f64]).collect(),
        (0..20).map(|i| if i < 10 { 2.0 } else { 8.0 }).collect(),
    );
    let m = FittedModel::fit(&d, &options()).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 2);
    assert!(matches!(
        m.trees[0].root.split,
        Some(Split::Numeric { threshold: 9.5, .. })
    ));
    let p = m.predict(&d.rows).unwrap();
    let m2 = FittedModel::from_json(&m.to_json().unwrap()).unwrap();
    assert_eq!(p, m2.predict(&d.rows).unwrap());
    assert_eq!(
        values(p),
        d.targets
            .iter()
            .map(|v| v.number().unwrap())
            .collect::<Vec<_>>()
    );
}
#[test]
fn xor_requires_interaction_lookahead() {
    let mut x = vec![];
    let mut y = vec![];
    for _ in 0..30 {
        for a in 0..2 {
            for b in 0..2 {
                x.push(vec![a as f64, b as f64]);
                y.push((a ^ b) as f64);
            }
        }
    }
    let d = data(x, y);
    let mut opt = options();
    opt.task = Task::Classification;
    opt.max_depth = 2;
    let m = FittedModel::fit(&d, &opt).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 4);
    for (p, y) in m.predict(&d.rows).unwrap().iter().zip(&d.targets) {
        if let Prediction::Classification { label, .. } = p {
            assert_eq!(*label, y.label().unwrap());
        } else {
            panic!();
        }
    }
    opt.interactions = false;
    let no = FittedModel::fit(&d, &opt).unwrap();
    assert_eq!(no.trees[0].root.leaves(), 1);
}

#[test]
fn linear_discriminant_split_separates_parallel_classes() {
    let mut x = vec![];
    let mut y = vec![];
    for i in 0..100 {
        for class in 0..2 {
            let t = i as f64;
            x.push(vec![t, t + if class == 0 { -0.05 } else { 0.05 }]);
            y.push(class as f64);
        }
    }
    let d = data(x, y);
    let o = FitOptions {
        task: Task::Classification,
        linear_splits: true,
        interactions: false,
        // Examine every projected cutpoint; the default caps the search at 100.
        max_split_candidates: 0,
        max_depth: 1,
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    assert!(
        matches!(m.trees[0].root.split, Some(Split::Linear { .. })),
        "{}",
        m.to_text().unwrap()
    );
    for (p, y) in m.predict(&d.rows).unwrap().iter().zip(&d.targets) {
        if let Prediction::Classification { label, .. } = p {
            assert_eq!(Some(label.clone()), y.label());
        } else {
            panic!();
        }
    }
    assert_eq!(
        m.predict(&d.rows).unwrap(),
        FittedModel::from_json(&m.to_json().unwrap())
            .unwrap()
            .predict(&d.rows)
            .unwrap()
    );
}
#[test]
fn zero_posterior_correction_keeps_cost_based_class_decision() {
    // With a large false-negative cost, recomputing the decision from the
    // corrected (nonzero) posteriors would change a pure leaf's label.
    let d = dataset_from_csv(
        "x,y\n0,0\n0,0\n1,1\n1,1\n",
        &CsvOptions {
            target: "y".into(),
            ..Default::default()
        },
        &Task::Classification,
    )
    .unwrap();
    let o = FitOptions {
        task: Task::Classification,
        min_samples_leaf: 1,
        class_costs: Some(vec![vec![0.0, 1.0], vec![1000.0, 0.0]]),
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    let labels: Vec<String> = m
        .predict(&d.rows)
        .unwrap()
        .into_iter()
        .map(|p| match p {
            Prediction::Classification { label, .. } => label,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(labels, ["0", "0", "1", "1"]);
}
#[test]
fn tied_leaf_counts_predict_the_class_with_the_larger_corrected_posterior() {
    // A native GUIDE 46.3 probe: the left leaf holds two c1 and two c2 rows
    // and no c0. GUIDE breaks the tie by the zero-posterior-corrected
    // probabilities, which favor c2, the class with the larger prior.
    let mut csv = String::from("x,y\n1,c1\n2,c1\n1.5,c2\n2.5,c2\n9,c2\n9.5,c2\n");
    for i in 0..10 {
        csv.push_str(&format!("{},c0\n", 5.0 + 0.3 * i as f64));
    }
    let d = dataset_from_csv(
        &csv,
        &CsvOptions {
            target: "y".into(),
            ..Default::default()
        },
        &Task::Classification,
    )
    .unwrap();
    let o = FitOptions {
        task: Task::Classification,
        max_depth: 1,
        min_samples_leaf: 2,
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    let labels: Vec<String> = m
        .predict(&d.rows[..4])
        .unwrap()
        .into_iter()
        .map(|p| match p {
            Prediction::Classification { label, .. } => label,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(labels, ["c2", "c2", "c2", "c2"]);
}
#[test]
fn zero_gain_splits_follow_min_gain() {
    // Every cut leaves both child means at 5, but the residual signs still
    // associate with x. GUIDE splits anyway; a positive min_gain does not.
    let mut x = vec![];
    let mut y = vec![];
    for (v, pair) in [
        (0.0, [0.0, 10.0]),
        (1.0, [5.0, 5.0]),
        (2.0, [5.0, 5.0]),
        (3.0, [0.0, 10.0]),
    ] {
        for _ in 0..3 {
            for t in pair {
                x.push(vec![v]);
                y.push(t);
            }
        }
    }
    let d = data(x, y);
    let mut o = FitOptions {
        max_depth: 1,
        min_gain: 0.0,
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 2);
    assert!(matches!(
        m.trees[0].root.split,
        Some(Split::Numeric { threshold, .. }) if threshold == 0.5
    ));
    o.min_gain = 1e-12;
    assert_eq!(FittedModel::fit(&d, &o).unwrap().trees[0].root.leaves(), 1);
}
#[test]
fn sibling_leaves_with_the_same_class_are_merged_bottom_up() {
    // Cells (A, B) with P(class a): the root split on A and the child split
    // on B both leave class a predicted everywhere, so the whole tree merges.
    let mut rows = vec![];
    let mut targets = vec![];
    for ((a, b), share) in [((0, 0), 40), ((0, 1), 40), ((1, 0), 38), ((1, 1), 24)] {
        for i in 0..40 {
            rows.push(vec![Value::Number(a as f64), Value::Number(b as f64)]);
            targets.push(Value::Text(if i < share { "a" } else { "b" }.into()));
        }
    }
    let schema = ["A", "B"]
        .map(|name| FeatureSchema {
            name: name.into(),
            feature_type: FeatureType::Numeric,
            role: Role::Split,
            flags: None,
        })
        .to_vec();
    let d = Dataset {
        schema,
        rows,
        targets,
        weights: vec![],
        offsets: vec![],
    };
    let mut o = FitOptions {
        task: Task::Classification,
        max_depth: 2,
        min_gain: 0.0,
        ..options()
    };
    assert_eq!(FittedModel::fit(&d, &o).unwrap().trees[0].root.leaves(), 1);
    // Costs make the (A=1, B=1) cell predict b, so both splits survive.
    o.class_costs = Some(vec![vec![0.0, 1.0], vec![3.0, 0.0]]);
    assert_eq!(FittedModel::fit(&d, &o).unwrap().trees[0].root.leaves(), 3);
}
#[test]
fn flag_schemas_are_validated() {
    let feature = |name: &str, kind: FeatureType, role: Role, flags: Option<&str>| FeatureSchema {
        name: name.into(),
        feature_type: kind,
        role,
        flags: flags.map(Into::into),
    };
    let fit = |schema: Vec<FeatureSchema>| {
        let d = Dataset {
            rows: vec![vec![Value::Missing, Value::Text("A".into())]; 4],
            schema,
            targets: (0..4).map(|i| Value::Number(i as f64)).collect(),
            weights: vec![],
            offsets: vec![],
        };
        FittedModel::fit(&d, &options()).map(|_| ()).unwrap_err().0
    };
    let x = || feature("x", FeatureType::Numeric, Role::Split, None);
    let categorical = FeatureType::Categorical;
    assert!(
        fit(vec![
            x(),
            feature("f", categorical.clone(), Role::Flag, Some("z"))
        ])
        .contains("unknown feature")
    );
    assert!(
        fit(vec![
            x(),
            feature("f", FeatureType::Numeric, Role::Flag, Some("x"))
        ])
        .contains("must be categorical")
    );
    assert!(
        fit(vec![
            x(),
            feature("f", categorical.clone(), Role::Flag, None)
        ])
        .contains("used together")
    );
    assert!(
        fit(vec![x(), feature("f", categorical, Role::Split, Some("x"))]).contains("used together")
    );
}
#[test]
fn missingness_can_be_the_predictor() {
    let mut d = data(
        vec![vec![1.0]; 20],
        (0..20).map(|i| if i < 10 { 0.0 } else { 5.0 }).collect(),
    );
    for row in &mut d.rows[10..] {
        row[0] = Value::Missing;
    }
    let m = FittedModel::fit(&d, &options()).unwrap();
    assert!(matches!(m.trees[0].root.split, Some(Split::Missing { .. })));
    assert_eq!(
        values(
            m.predict(&[vec![Value::Missing], vec![Value::Number(1.0)]])
                .unwrap()
        ),
        vec![5.0, 0.0]
    );
}
#[test]
fn priors_costs_and_classification_weights() {
    let mut d = data(
        vec![vec![0.0]; 10],
        (0..10).map(|i| if i < 8 { 0.0 } else { 1.0 }).collect(),
    );
    d.weights = vec![1.0; 10];
    d.weights[9] = 500.0;
    let mut o = options();
    o.task = Task::Classification;
    o.max_depth = 0;
    o.class_priors = Some(vec![0.25, 0.75]);
    let m = FittedModel::fit(&d, &o).unwrap();
    if let Prediction::Classification {
        label,
        probabilities,
    } = &m.predict(&d.rows).unwrap()[0]
    {
        assert_eq!(label, "1");
        assert_close(probabilities[1].1, 0.75);
    } else {
        panic!();
    }
    o.class_priors = None;
    o.class_costs = Some(vec![vec![0.0, 1.0], vec![10.0, 0.0]]);
    let m = FittedModel::fit(&d, &o).unwrap();
    if let Prediction::Classification { label, .. } = &m.predict(&d.rows).unwrap()[0] {
        assert_eq!(label, "1");
    } else {
        panic!();
    }
}
#[test]
fn weighted_leaves_and_missing_targets() {
    let mut d = data(vec![vec![0.0]; 4], vec![1.0, 3.0, 100.0, 9.0]);
    d.targets[3] = Value::Missing;
    d.weights = vec![1.0, 3.0, 0.0, 1.0];
    let m = FittedModel::fit(&d, &options()).unwrap();
    assert_eq!(m.report.excluded_rows, 2);
    assert_close(values(m.predict(&d.rows).unwrap())[0], 2.5);
}
#[test]
fn categorical_splits_and_unseen_values() {
    let csv = "color,target\nred,1\nred,1\nblue,9\nblue,9\ngreen,9\ngreen,9\n";
    let d = dataset_from_csv(csv, &CsvOptions::default(), &Task::default()).unwrap();
    let m = FittedModel::fit(&d, &options()).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 2);
    let pred = values(
        m.predict(&[
            vec![Value::Text("red".into())],
            vec![Value::Text("unknown".into())],
        ])
        .unwrap(),
    );
    assert_eq!(pred, vec![1.0, 9.0]);
}
#[test]
fn nonconstant_model_and_rank_deficiency() {
    let mut d = data(
        (0..30).map(|i| vec![i as f64, 2.0 * i as f64]).collect(),
        (0..30).map(|i| 3.0 + 2.0 * i as f64).collect(),
    );
    for f in &mut d.schema {
        f.role = Role::Both;
    }
    let mut o = options();
    o.node_model = NodeModel::Linear;
    let m = FittedModel::fit(&d, &o).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 1);
    let pred = values(
        m.predict(&[vec![Value::Number(30.0), Value::Number(60.0)]])
            .unwrap(),
    );
    assert_close(pred[0], 63.0);
}

#[test]
fn stepwise_selects_signal_and_drops_irrelevant_group() {
    let mut d = data(
        (0..100)
            .map(|i| vec![(i % 10) as f64, (i / 10) as f64])
            .collect(),
        (0..100).map(|i| 2.0 + 3.0 * (i % 10) as f64).collect(),
    );
    for f in &mut d.schema {
        f.role = Role::Both;
    }
    let o = FitOptions {
        node_model: NodeModel::Stepwise {
            alpha: 0.05,
            max_variables: 0,
            backward: true,
        },
        max_depth: 0,
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    if let LeafModel::Linear { terms, .. } = &m.trees[0].root.model {
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].feature, 0);
    } else {
        panic!("expected selected linear model");
    }
    let prediction = values(
        m.predict(&[vec![Value::Number(11.0), Value::Number(100.0)]])
            .unwrap(),
    );
    assert_close(prediction[0], 35.0);
}
#[test]
fn polynomial_fit_and_imputation_are_serialized() {
    let mut d = data(
        (0..30).map(|i| vec![i as f64]).collect(),
        (0..30).map(|i| 1.0 + (i as f64).powi(2)).collect(),
    );
    d.schema[0].role = Role::Both;
    let mut o = options();
    o.node_model = NodeModel::Polynomial { degree: 2 };
    o.max_depth = 0;
    let m = FittedModel::fit(&d, &o).unwrap();
    let p = values(
        m.predict(&[vec![Value::Number(5.5)], vec![Value::Missing]])
            .unwrap(),
    );
    assert_close(p[0], 31.25);
    assert!(p[1].is_finite());
    assert_eq!(
        m.predict(&d.rows).unwrap(),
        FittedModel::from_json(&m.to_json().unwrap())
            .unwrap()
            .predict(&d.rows)
            .unwrap()
    );
}
#[test]
fn quantile_poisson_exposure_and_logistic() {
    let mut d = data(vec![vec![1.0]; 5], vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    let mut o = options();
    o.max_depth = 0;
    o.task = Task::Regression {
        loss: Loss::Quantile { probability: 0.5 },
    };
    assert_eq!(
        values(FittedModel::fit(&d, &o).unwrap().predict(&d.rows).unwrap())[0],
        4.0
    );
    o.task = Task::Regression {
        loss: Loss::Poisson,
    };
    d.offsets = vec![2.0; 5];
    let m = FittedModel::fit(&d, &o).unwrap();
    assert!(m.predict(&d.rows).is_err());
    assert_eq!(
        values(m.predict_with_offsets(&d.rows, Some(&[3.0; 5])).unwrap())[0],
        6.0
    );
    d.offsets.clear();
    d.targets = vec![
        Value::Number(0.0),
        Value::Number(0.0),
        Value::Number(1.0),
        Value::Number(1.0),
        Value::Number(1.0),
    ];
    o.task = Task::Regression {
        loss: Loss::Logistic,
    };
    assert_close(
        values(FittedModel::fit(&d, &o).unwrap().predict(&d.rows).unwrap())[0],
        0.6,
    );
}

#[test]
fn linear_quantile_trees_prune_and_bag() {
    let mut d = data(
        (0..80)
            .map(|i| vec![(i % 20) as f64, (i / 40) as f64])
            .collect(),
        (0..80)
            .map(|i| {
                2.0 + 0.5 * (i % 20) as f64
                    + 20.0 * (i / 40) as f64
                    + if i % 11 == 0 { 30.0 } else { 0.0 }
            })
            .collect(),
    );
    d.schema[0].role = Role::Fit;
    let mut o = FitOptions {
        task: Task::Regression {
            loss: Loss::Quantile { probability: 0.5 },
        },
        node_model: NodeModel::Linear,
        max_depth: 1,
        pruning: Pruning::CrossValidation {
            folds: 4,
            standard_errors: 0.0,
        },
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 2);
    assert!(m.report.warnings.is_empty(), "{:?}", m.report.warnings);
    for (i, p) in values(m.predict(&d.rows).unwrap()).iter().enumerate() {
        let expected = 2.0 + 0.5 * (i % 20) as f64 + 20.0 * (i / 40) as f64;
        assert!((p - expected).abs() < 1e-5, "{p} != {expected}");
    }
    o.pruning = Pruning::None;
    let bag = FittedModel::fit_ensemble(
        &d,
        &o,
        &EnsembleOptions {
            trees: 5,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(bag.report.out_of_bag_loss.unwrap().is_finite());
    assert!(
        !bag.report
            .warnings
            .iter()
            .any(|w| w.contains("convergence")),
        "{:?}",
        bag.report.warnings
    );
    let restored = FittedModel::from_json(&bag.to_json().unwrap()).unwrap();
    assert_eq!(
        bag.predict(&d.rows).unwrap(),
        restored.predict(&d.rows).unwrap()
    );
}

#[test]
fn dual_quantiles_detect_variance_changes_and_roundtrip() {
    let mut x = vec![];
    let mut y = vec![];
    for group in 0..2 {
        for _ in 0..20 {
            for value in [-1.0, 0.0, 1.0] {
                x.push(vec![group as f64]);
                y.push(value * if group == 0 { 1.0 } else { 10.0 });
            }
        }
    }
    let d = data(x, y);
    let options = FitOptions {
        task: Task::Regression {
            loss: Loss::DualQuantile {
                lower: 0.25,
                upper: 0.75,
            },
        },
        ..options()
    };
    let m = FittedModel::fit(&d, &options).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 2);
    let p = m
        .predict(&[vec![Value::Number(0.0)], vec![Value::Number(1.0)]])
        .unwrap();
    assert_eq!(
        p,
        vec![
            Prediction::Quantiles {
                lower: -1.0,
                upper: 1.0
            },
            Prediction::Quantiles {
                lower: -10.0,
                upper: 10.0
            }
        ]
    );
    assert_eq!(
        m.predict(&d.rows).unwrap(),
        FittedModel::from_json(&m.to_json().unwrap())
            .unwrap()
            .predict(&d.rows)
            .unwrap()
    );
    let bag = FittedModel::fit_ensemble(
        &d,
        &options,
        &EnsembleOptions {
            trees: 5,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(bag.report.out_of_bag_loss.unwrap().is_finite());
    for p in bag.predict(&d.rows).unwrap() {
        if let Prediction::Quantiles { lower, upper } = p {
            assert!(lower <= upper);
        } else {
            panic!();
        }
    }
}
#[test]
fn linear_poisson_recovers_known_log_rate() {
    let mut d = data(
        (0..20).map(|i| vec![i as f64 / 10.0]).collect(),
        (0..20)
            .map(|i| (1.0 + 0.5 * i as f64 / 10.0).exp())
            .collect(),
    );
    d.schema[0].role = Role::Both;
    let mut o = options();
    o.task = Task::Regression {
        loss: Loss::Poisson,
    };
    o.node_model = NodeModel::Linear;
    o.max_depth = 0;
    let m = FittedModel::fit(&d, &o).unwrap();
    let p = values(m.predict(&d.rows).unwrap());
    for (i, p) in p.iter().enumerate() {
        assert!((p - d.targets[i].number().unwrap()).abs() < 1e-6);
    }
}
#[test]
fn periodic_splits_wrap_at_origin() {
    let mut d = data(
        (0..24).map(|i| vec![i as f64]).collect(),
        (0..24)
            .map(|i| if !(6..18).contains(&i) { 10.0 } else { 0.0 })
            .collect(),
    );
    d.schema[0].feature_type = FeatureType::Periodic { period: 24.0 };
    let m = FittedModel::fit(&d, &options()).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 2);
    assert_eq!(
        values(
            m.predict(&[
                vec![Value::Number(0.0)],
                vec![Value::Number(24.0)],
                vec![Value::Number(-24.0)]
            ])
            .unwrap()
        ),
        vec![10.0, 10.0, 10.0]
    );
}
#[test]
fn cv_fixed_size_and_validation_pruning() {
    let d = data(
        (0..40).map(|i| vec![i as f64]).collect(),
        (0..40).map(|i| if i < 20 { 0.0 } else { 10.0 }).collect(),
    );
    let mut o = options();
    o.pruning = Pruning::CrossValidation {
        folds: 5,
        standard_errors: 0.25,
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 2);
    assert!(!m.report.pruning_scores.is_empty());
    o.pruning = Pruning::FixedLeaves { max_leaves: 1 };
    assert_eq!(FittedModel::fit(&d, &o).unwrap().trees[0].root.leaves(), 1);
    o.pruning = Pruning::Validation;
    assert!(FittedModel::fit(&d, &o).is_err());
    let mut v = d.clone();
    v.targets = vec![Value::Number(5.0); 40];
    assert_eq!(
        FittedModel::fit_with_validation(&d, &o, Some(&v))
            .unwrap()
            .trees[0]
            .root
            .leaves(),
        1
    );
}
#[test]
fn bagging_rejects_cross_validation_pruning_on_bootstrap_rows() {
    let d = data(
        (0..30).map(|i| vec![i as f64]).collect(),
        (0..30).map(|i| if i < 15 { 0.0 } else { 5.0 }).collect(),
    );
    let o = FitOptions {
        pruning: Pruning::CrossValidation {
            folds: 5,
            standard_errors: 0.0,
        },
        ..options()
    };
    let err = FittedModel::fit_ensemble(&d, &o, &EnsembleOptions::default()).unwrap_err();
    assert!(err.0.contains("cross-validation"), "{err}");
    let o = FitOptions {
        pruning: Pruning::CostComplexity { alpha: 0.1 },
        ..options()
    };
    let e = EnsembleOptions {
        trees: 5,
        ..Default::default()
    };
    assert!(FittedModel::fit_ensemble(&d, &o, &e).is_ok());
}
#[test]
fn ensemble_is_repeatable_and_has_oob_predictions() {
    let d = data(
        (0..30).map(|i| vec![i as f64]).collect(),
        (0..30).map(|i| if i < 15 { 0.0 } else { 5.0 }).collect(),
    );
    let o = options();
    let e = EnsembleOptions {
        trees: 15,
        kind: EnsembleKind::Forest,
        ..Default::default()
    };
    let a = FittedModel::fit_ensemble(&d, &o, &e).unwrap();
    let b = FittedModel::fit_ensemble(&d, &o, &e).unwrap();
    assert_eq!(a.to_json().unwrap(), b.to_json().unwrap());
    assert_eq!(a.report.out_of_bag_rows, 30);
    assert!(a.report.out_of_bag_loss.unwrap() < 2.0);
}
#[test]
fn parallel_fits_do_not_depend_on_thread_count() {
    let d = data(
        (0..80)
            .map(|i| vec![(i * 37 % 80) as f64, (i * 11 % 7) as f64, (i % 5) as f64])
            .collect(),
        (0..80)
            .map(|i| f64::from(i * 37 % 80 > 40) * 3.0 + (i % 5) as f64)
            .collect(),
    );
    let cv = FitOptions {
        pruning: Pruning::CrossValidation {
            folds: 10,
            standard_errors: 0.5,
        },
        ..options()
    };
    // Two threads draw forest samples in batches of 8, so 30 trees span four.
    let forest = EnsembleOptions {
        trees: 30,
        kind: EnsembleKind::Forest,
        ..Default::default()
    };
    let fit = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (
                    FittedModel::fit(&d, &cv).unwrap().to_json().unwrap(),
                    FittedModel::fit_ensemble(&d, &options(), &forest)
                        .unwrap()
                        .to_json()
                        .unwrap(),
                )
            })
    };
    let one = fit(1);
    assert_eq!(fit(2), one);
    assert_eq!(fit(5), one);
}
#[test]
fn importance_orders_signal_before_constant_and_is_repeatable() {
    let d = data(
        (0..40).map(|i| vec![i as f64, 1.0]).collect(),
        (0..40).map(|i| if i < 20 { 0.0 } else { 10.0 }).collect(),
    );
    let o = options();
    let io = ImportanceOptions {
        permutations: 20,
        max_depth: 3,
        seed: 7,
    };
    let a = importance(&d, &o, &io).unwrap();
    let b = importance(&d, &o, &io).unwrap();
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
    assert_eq!(a.scores[0].feature, "x0");
    assert!(a.scores[0].adjusted_score > 1.0);
    assert_eq!(a.scores[1].raw_score, 0.0);
}
#[test]
fn rejects_invalid_inputs_and_corrupt_models() {
    let mut d = data(vec![vec![0.0]; 4], vec![1.0; 4]);
    let o = options();
    let mut m = FittedModel::fit(&d, &o).unwrap();
    m.trees[0].root.model = LeafModel::Classification {
        probabilities: vec![1.0],
        predicted_class: 0,
    };
    assert!(m.to_json().is_err());
    d.weights = vec![-1.0; 4];
    assert!(FittedModel::fit(&d, &o).is_err());
    d.weights.clear();
    d.rows[0][0] = Value::Number(f64::NAN);
    assert!(FittedModel::fit(&d, &o).is_err());
    assert!(serde_json::from_str::<FitOptions>(r#"{"task":{"kind":"survival"}}"#).is_err());
}

#[test]
fn rejects_inconsistent_saved_roles_domains_and_node_ids() {
    let mut d = data(
        (0..20).map(|i| vec![i as f64]).collect(),
        (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect(),
    );
    let mut m = FittedModel::fit(&d, &options()).unwrap();
    m.trees[0].root.right.as_mut().unwrap().id = m.trees[0].root.left.as_ref().unwrap().id;
    assert!(m.to_json().is_err());
    let mut m = FittedModel::fit(&d, &options()).unwrap();
    m.schema[0].role = Role::Fit;
    assert!(m.predict(&d.rows).is_err());
    for (loss, value) in [(Loss::Poisson, -1.0), (Loss::Logistic, 1.1)] {
        let o = FitOptions {
            task: Task::Regression { loss },
            max_depth: 0,
            ..options()
        };
        let mut m = FittedModel::fit(&d, &o).unwrap();
        m.trees[0].root.model = LeafModel::Constant { value };
        assert!(FittedModel::from_json(&serde_json::to_string(&m).unwrap()).is_err());
    }
    d.schema[0].role = Role::Both;
    let o = FitOptions {
        node_model: NodeModel::Linear,
        max_depth: 0,
        ..options()
    };
    let mut m = FittedModel::fit(&d, &o).unwrap();
    m.schema[0].role = Role::Split;
    assert!(m.to_json().is_err());
    assert!(FittedModel::fit_with_validation(&d, &o, Some(&d)).is_err());
}

#[test]
fn quantile_numerical_fallback_is_reported() {
    // Five rows: quantile leaves fit linear models from p + 4 rows.
    let mut d = data(
        (0..5).map(|i| vec![f64::from(i)]).collect(),
        vec![2.0, 4.0, 8.0, 16.0, 32.0],
    );
    d.schema[0].role = Role::Both;
    d.weights = vec![1.0, 1e-310, 1e-310, 1e-310, 1e-310];
    let o = FitOptions {
        task: Task::Regression {
            loss: Loss::Quantile { probability: 0.5 },
        },
        node_model: NodeModel::Linear,
        max_depth: 0,
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    assert!(matches!(
        m.trees[0].root.model,
        LeafModel::Linear {
            converged: false,
            ..
        }
    ));
    assert_eq!(
        m.report
            .warnings
            .iter()
            .filter(|w| w.contains("convergence"))
            .count(),
        1
    );
    assert_eq!(values(m.predict(&d.rows).unwrap()), vec![2.0; 5]);
}
#[test]
fn all_missing_and_constant_predictors_give_valid_root() {
    let mut d = data(
        vec![vec![1.0, 2.0]; 10],
        (0..10).map(|i| i as f64).collect(),
    );
    for r in &mut d.rows {
        r[0] = Value::Missing;
    }
    let m = FittedModel::fit(&d, &options()).unwrap();
    assert_eq!(m.trees[0].root.leaves(), 1);
    assert_close(values(m.predict(&d.rows).unwrap())[0], 4.5);
}
#[test]
fn csv_quotes_multiline_and_schema_roundtrip() {
    let csv = "\u{feff}name,target\r\n\"a,\"\"b\"\"\",1\r\n\"line\nbreak\",2\r\n";
    let r = parse_csv(csv, ',').unwrap();
    assert_eq!(r[1][0], "a,\"b\"");
    assert_eq!(r[2][0], "line\nbreak");
    assert!(parse_csv("a,b\n1\n", ',').is_err());
    assert!(parse_csv("a,a\n1,2", ',').is_err());
    assert!(parse_csv("a\n\"bad", ',').is_err());
    assert_eq!(
        parse_csv("a,b\n1,2\n\n3,4\n\n\n", ',').unwrap(),
        vec![vec!["a", "b"], vec!["1", "2"], vec!["3", "4"]]
    );
    assert_eq!(parse_csv("a,b\r\n1,2\r\n\r\n", ',').unwrap().len(), 2);
    let err = parse_csv("a,b\n1,2\n\n3\n", ',').unwrap_err();
    assert!(err.0.contains("CSV line 4 has 1 fields"), "{err}");
    let err = parse_csv("a,b\n\"x\ny\",1\n2\n", ',').unwrap_err();
    assert!(err.0.contains("CSV line 4 has 1 fields"), "{err}");
    let numeric: CsvOptions = serde_json::from_str(
        r#"{"target":"y","features":[{"name":"x","feature_type":{"kind":"numeric"}}]}"#,
    )
    .unwrap();
    let err = dataset_from_csv("x,y\n1,2\n\nabc,3\n", &numeric, &Task::default()).unwrap_err();
    assert!(
        err.0.contains("CSV line 4, column \"x\": expected number"),
        "{err}"
    );
    let cfg:CsvOptions=serde_json::from_str(r#"{"target":"target","features":[{"name":"name","feature_type":{"kind":"categorical"},"role":"split"}]}"#).unwrap();
    assert_eq!(
        dataset_from_csv(csv, &cfg, &Task::default())
            .unwrap()
            .schema
            .len(),
        1
    );
}

fn classes(x: Vec<Vec<f64>>, labels: Vec<&str>) -> Dataset {
    let mut d = data(x, vec![0.0; labels.len()]);
    d.targets = labels.into_iter().map(|l| Value::Text(l.into())).collect();
    d
}
#[test]
fn proper_scoring_rules_choose_pruned_classification_trees() {
    let n = 240;
    // A noisy two-class signal on x0 with a weaker interaction on x1.
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![(i * 37 % n) as f64, (i * 11 % 13) as f64])
        .collect();
    let labels: Vec<&str> = x
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let signal = r[0] > 120.0 || (r[1] > 8.0 && r[0] > 60.0);
            if signal != (i % 7 == 0) { "a" } else { "b" }
        })
        .collect();
    let d = classes(x, labels);
    let fit = |loss| {
        FittedModel::fit(
            &d,
            &FitOptions {
                task: Task::Classification,
                pruning: Pruning::CrossValidation {
                    folds: 5,
                    standard_errors: 0.0,
                },
                pruning_loss: loss,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let cost = fit(PruningLoss::Misclassification);
    let log = fit(PruningLoss::LogLoss);
    let brier = fit(PruningLoss::Brier);
    // The default is not serialized, so existing saved models are unchanged.
    assert!(!cost.to_json().unwrap().contains("pruning_loss"));
    assert!(
        log.to_json()
            .unwrap()
            .contains("\"pruning_loss\": \"log_loss\"")
    );
    for model in [&cost, &log, &brier] {
        let scores = &model.report.pruning_scores;
        assert!(scores.len() > 2 && scores.iter().all(|s| s.mean_loss.is_finite()));
    }
    // Each criterion reports its own held-out loss on the same pruning sequence.
    let alphas = |m: &FittedModel| {
        m.report
            .pruning_scores
            .iter()
            .map(|s| s.alpha)
            .collect::<Vec<_>>()
    };
    assert_eq!(alphas(&cost), alphas(&log));
    let first = |m: &FittedModel| m.report.pruning_scores[0].mean_loss;
    assert!(first(&cost) < 1.0 && first(&brier) < 2.0 && first(&log) > first(&brier) / 2.0);
    let restored = FittedModel::from_json(&log.to_json().unwrap()).unwrap();
    assert_eq!(restored.options.pruning_loss, PruningLoss::LogLoss);
    // Forests report out-of-bag log loss.
    let forest = FittedModel::fit_ensemble(
        &d,
        &FitOptions {
            task: Task::Classification,
            pruning: Pruning::None,
            pruning_loss: PruningLoss::LogLoss,
            ..Default::default()
        },
        &EnsembleOptions {
            trees: 20,
            kind: EnsembleKind::Forest,
            ..Default::default()
        },
    )
    .unwrap();
    let oob = forest.report.out_of_bag_loss.unwrap();
    assert!(oob.is_finite() && oob > 0.0 && oob < 2.0_f64.ln() + 0.2);
}
#[test]
fn pruning_loss_is_validated() {
    let d = data(
        (0..20).map(|i| vec![i as f64]).collect(),
        (0..20).map(f64::from).collect(),
    );
    let regression = FitOptions {
        pruning_loss: PruningLoss::Brier,
        ..options()
    };
    let error = FittedModel::fit(&d, &regression).unwrap_err().to_string();
    assert!(error.contains("only to classification"), "{error}");
    let labels = classes(
        (0..20).map(|i| vec![i as f64]).collect(),
        (0..20).map(|i| if i < 10 { "a" } else { "b" }).collect(),
    );
    let costs = FitOptions {
        task: Task::Classification,
        class_costs: Some(vec![vec![0.0, 1.0], vec![2.0, 0.0]]),
        pruning_loss: PruningLoss::LogLoss,
        ..options()
    };
    let error = FittedModel::fit(&labels, &costs).unwrap_err().to_string();
    assert!(error.contains("ignore class costs"), "{error}");
}
#[test]
fn quantile_linear_leaves_need_p_plus_four_rows() {
    // GUIDE 46.3 fits a linear quantile leaf from p + 4 rows (complete
    // numeric regressors) and a constant below that.
    for p in 1..=2 {
        for n in [p + 3, p + 4] {
            let x: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..p)
                        .map(|k| ((i * (k + 3)) % 7) as f64 + i as f64)
                        .collect()
                })
                .collect();
            let y = (0..n).map(|i| 1.0 + 2.0 * i as f64).collect();
            let mut d = data(x, y);
            for f in &mut d.schema {
                f.role = Role::Both;
            }
            let o = FitOptions {
                task: Task::Regression {
                    loss: Loss::Quantile { probability: 0.5 },
                },
                node_model: NodeModel::Linear,
                max_depth: 0,
                ..options()
            };
            let m = FittedModel::fit(&d, &o).unwrap();
            let linear = matches!(m.trees[0].root.model, LeafModel::Linear { .. });
            assert_eq!(linear, n == p + 4, "p = {p}, n = {n}");
        }
    }
}
