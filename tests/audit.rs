//! Regressions found by independent objectives and invariance checks.
use guide_rs::*;

fn options() -> FitOptions {
    FitOptions {
        pruning: Pruning::None,
        min_gain: 0.0,
        max_split_candidates: 0,
        max_depth: 1,
        ..Default::default()
    }
}
fn data(y: Vec<f64>) -> Dataset {
    Dataset {
        schema: vec![FeatureSchema {
            name: "x".into(),
            feature_type: FeatureType::Numeric,
            role: Role::Both,
        }],
        rows: (0..y.len())
            .map(|i| vec![Value::Number(i as f64)])
            .collect(),
        targets: y.into_iter().map(Value::Number).collect(),
        weights: vec![],
        offsets: vec![],
    }
}
fn predictions(m: &FittedModel, d: &Dataset) -> Vec<f64> {
    m.predict_with_offsets(
        &d.rows,
        (!d.offsets.is_empty()).then_some(d.offsets.as_slice()),
    )
    .unwrap()
    .into_iter()
    .map(|p| match p {
        Prediction::Regression { value } => value,
        _ => panic!(),
    })
    .collect()
}

#[test]
fn numerical_cut_minimum_counts_observed_split_values_before_missing_routing() {
    // Native GUIDE 46.3 control: 30 observed x values, ten missing, minimum 12.
    // Counting routed missing rows admits x=27.5 with only two observed values
    // on the right. Native's constrained optimum is 16.5 (17/13 observed),
    // including its exclusive upper rank bound (right > minimum).
    let mut d = data(
        (0..40)
            .map(|i| {
                if i < 15 {
                    0.0
                } else if i < 30 {
                    5.0
                } else {
                    30.0
                }
            })
            .collect(),
    );
    for row in &mut d.rows[30..] {
        row[0] = Value::Missing;
    }
    let model = FittedModel::fit(
        &d,
        &FitOptions {
            node_model: NodeModel::Constant,
            min_samples_leaf: 12,
            ..options()
        },
    )
    .unwrap();
    let Some(Split::Numeric { threshold, .. }) = &model.trees[0].root.split else {
        panic!()
    };
    assert_eq!(*threshold, 16.5);
    for node in [&model.trees[0].root.left, &model.trees[0].root.right] {
        assert!(node.as_ref().unwrap().sample_count >= 12);
    }
}

#[test]
fn numerical_cut_rank_bounds_match_native_asymmetric_endpoints() {
    for minimum in [2, 12] {
        for reverse in [false, true] {
            let mut d = data((0..30).map(|i| if i == 29 { 100.0 } else { 0.0 }).collect());
            if reverse {
                d.rows.reverse();
            }
            let model = FittedModel::fit(
                &d,
                &FitOptions {
                    min_samples_leaf: minimum,
                    ..options()
                },
            )
            .unwrap();
            let Some(Split::Numeric { threshold, .. }) = &model.trees[0].root.split else {
                panic!()
            };
            let expected = if reverse {
                minimum as f64 - 0.5
            } else {
                28.5 - minimum as f64
            };
            assert_eq!(*threshold, expected);
        }
    }
}

fn large_signal_data() -> Dataset {
    Dataset {
        schema: ["weaker", "stronger"]
            .into_iter()
            .map(|name| FeatureSchema {
                name: name.into(),
                feature_type: FeatureType::Numeric,
                role: Role::Split,
            })
            .collect(),
        rows: (0..6000)
            .map(|i| {
                let y = (i % 2) as f64;
                vec![
                    Value::Number(if i % 10 == 0 { 1.0 - y } else { y }),
                    Value::Number(y),
                ]
            })
            .collect(),
        targets: (0..6000).map(|i| Value::Number((i % 2) as f64)).collect(),
        weights: vec![],
        offsets: vec![],
    }
}

#[test]
fn stepwise_extreme_f_tails_select_the_stronger_feature_in_either_order() {
    for reverse in [false, true] {
        let mut d = data(
            (0..3200)
                .map(|i| {
                    let x = if i % 2 == 0 { -1.0 } else { 1.0 };
                    let z = if i / 2 % 2 == 0 { -1.0 } else { 1.0 };
                    x + 1.1 * z + 0.01 * if i / 4 % 2 == 0 { -1.0 } else { 1.0 }
                })
                .collect(),
        );
        d.schema = ["weaker", "stronger"]
            .into_iter()
            .map(|name| FeatureSchema {
                name: name.into(),
                feature_type: FeatureType::Numeric,
                role: Role::Fit,
            })
            .collect();
        d.rows = (0..3200)
            .map(|i| {
                vec![
                    Value::Number(if i % 2 == 0 { -1.0 } else { 1.0 }),
                    Value::Number(if i / 2 % 2 == 0 { -1.0 } else { 1.0 }),
                ]
            })
            .collect();
        if reverse {
            d.schema.reverse();
            for row in &mut d.rows {
                row.reverse();
            }
        }
        let o = FitOptions {
            max_depth: 0,
            node_model: NodeModel::Stepwise {
                alpha: 0.05,
                max_variables: 1,
                backward: true,
            },
            ..options()
        };
        // Balanced orthogonal columns give these univariate residual sums of
        // squares analytically. Both ordinary F probabilities are zero.
        assert_eq!(stats::f_survival(3198.0 / 1.2101, 1, 3198), 0.0);
        assert_eq!(stats::f_survival(3198.0 * 1.21 / 1.0001, 1, 3198), 0.0);
        let m = FittedModel::fit(&d, &o).unwrap();
        let LeafModel::Linear { terms, .. } = &m.trees[0].root.model else {
            panic!("expected one selected feature")
        };
        assert_eq!(terms.len(), 1);
        assert_eq!(d.schema[terms[0].feature].name, "stronger");
    }
}

#[test]
fn underflowing_association_tails_do_not_make_selection_follow_column_order() {
    for task in [Task::default(), Task::Classification] {
        let mut d = large_signal_data();
        for _ in 0..2 {
            let o = FitOptions {
                task: task.clone(),
                interactions: false,
                ..options()
            };
            let m = FittedModel::fit(&d, &o).unwrap();
            let root = &m.trees[0].root;
            assert!(root.diagnostics.iter().all(|v| v.p_value == 0.0));
            assert_eq!(
                d.schema[root.split.as_ref().unwrap().feature()].name,
                "stronger"
            );
            let tiny_alpha = FitOptions {
                interactions: true,
                significance_level: f64::from_bits(1),
                ..o
            };
            let tiny = FittedModel::fit(&d, &tiny_alpha).unwrap();
            assert_eq!(
                d.schema[tiny.trees[0].root.split.as_ref().unwrap().feature()].name,
                "stronger"
            );
            assert!(
                tiny.trees[0]
                    .root
                    .diagnostics
                    .iter()
                    .all(|d| d.interaction_with.is_none())
            );
            d.schema.swap(0, 1);
            for row in &mut d.rows {
                row.swap(0, 1);
            }
        }
    }
}

#[test]
fn importance_scores_do_not_saturate_when_tail_probabilities_underflow() {
    let d = large_signal_data();
    for task in [Task::default(), Task::Classification] {
        let o = FitOptions {
            task,
            interactions: false,
            ..options()
        };
        let scores = importance(
            &d,
            &o,
            &ImportanceOptions {
                permutations: 2,
                max_depth: 1,
                seed: 18,
            },
        )
        .unwrap();
        assert_eq!(scores.algorithm_version.as_deref(), Some(ALGORITHM_VERSION));
        let mut legacy = serde_json::to_value(&scores).unwrap();
        legacy.as_object_mut().unwrap().remove("algorithm_version");
        let restored: ImportanceResult = serde_json::from_value(legacy).unwrap();
        assert!(restored.algorithm_version.is_none());
        let score = |name| {
            scores
                .scores
                .iter()
                .find(|s| s.feature == name)
                .unwrap()
                .raw_score
        };
        let multiplier = (d.rows.len() as f64).sqrt();
        assert!((score("stronger") / multiplier - 6000.0).abs() < 1e-7);
        assert!((score("weaker") / multiplier - 4000.0).abs() < 1e-7);
    }
}

#[test]
fn extreme_tail_selection_compares_significance_across_degrees_of_freedom() {
    let mut d = large_signal_data();
    d.schema[0].name = "many_levels".into();
    d.schema[0].feature_type = FeatureType::Categorical;
    d.schema[1].name = "binary".into();
    for (i, row) in d.rows.iter_mut().enumerate() {
        let y = i % 2;
        row[0] = Value::Text(format!("{}_{}", y, (i / 2) % 500));
        row[1] = Value::Number(if i % 10 == 0 {
            1.0 - y as f64
        } else {
            y as f64
        });
    }
    for task in [Task::default(), Task::Classification] {
        let o = FitOptions {
            task,
            interactions: false,
            max_split_candidates: 2,
            ..options()
        };
        let m = FittedModel::fit(&d, &o).unwrap();
        let root = &m.trees[0].root;
        assert_eq!(root.diagnostics[0].degrees_of_freedom, 999);
        assert!(root.diagnostics[0].chi_square > root.diagnostics[1].chi_square);
        assert!(root.diagnostics.iter().all(|d| d.p_value == 0.0));
        assert_eq!(root.split.as_ref().unwrap().feature(), 1);
        assert!(
            root.diagnostics[1].log_p_value.unwrap() < root.diagnostics[0].log_p_value.unwrap()
        );
        let encoded = m.to_json().unwrap();
        let mut legacy: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        fn remove_logs(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Object(object) => {
                    object.remove("log_p_value");
                    object.remove("importance_log_p_value");
                    for v in object.values_mut() {
                        remove_logs(v);
                    }
                }
                serde_json::Value::Array(array) => {
                    for v in array {
                        remove_logs(v);
                    }
                }
                _ => {}
            }
        }
        remove_logs(&mut legacy);
        legacy["algorithm_version"] = "guide-rs-0.1-experimental-reference3".into();
        let restored = FittedModel::from_json(&legacy.to_string()).unwrap();
        assert!(
            restored.trees[0]
                .root
                .diagnostics
                .iter()
                .all(|d| d.log_p_value.is_none())
        );
        assert_eq!(
            m.predict(&d.rows).unwrap(),
            restored.predict(&d.rows).unwrap()
        );
        let mut corrupt: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        corrupt["trees"][0]["root"]["diagnostics"][0]["log_p_value"] = 0.1.into();
        assert!(FittedModel::from_json(&corrupt.to_string()).is_err());
    }
}

#[test]
fn interaction_importance_reaches_identical_predictor_copies() {
    let d = Dataset {
        schema: ["a", "b", "copy", "noise"]
            .into_iter()
            .map(|name| FeatureSchema {
                name: name.into(),
                feature_type: FeatureType::Numeric,
                role: Role::Split,
            })
            .collect(),
        rows: (0..80)
            .map(|i| {
                vec![
                    Value::Number(((i / 20) % 2) as f64),
                    Value::Number(((i / 40) % 2) as f64),
                    Value::Number(((i / 20) % 2) as f64),
                    Value::Number((i % 2) as f64),
                ]
            })
            .collect(),
        targets: (0..80)
            .map(|i| Value::Number(f64::from((i / 20) % 2 != (i / 40) % 2)))
            .collect(),
        weights: vec![],
        offsets: vec![],
    };
    for task in [Task::default(), Task::Classification] {
        let r = importance(
            &d,
            &FitOptions { task, ..options() },
            &ImportanceOptions {
                permutations: 2,
                max_depth: 1,
                seed: 0,
            },
        )
        .unwrap();
        let raw = |name| {
            r.scores
                .iter()
                .find(|s| s.feature == name)
                .unwrap()
                .raw_score
        };
        assert!(raw("a") > 0.0);
        assert_eq!(raw("a"), raw("copy"));
        assert_eq!(raw("a"), raw("b"));
        assert_eq!(raw("noise"), 0.0);
    }
}

#[test]
fn importance_rejects_nonfinite_tree_statistics_instead_of_zero_scores() {
    let d = data(vec![1e200, -1e200, 1e200, -1e200]);
    assert!(FittedModel::fit(&d, &options()).is_err());
    assert!(
        importance(
            &d,
            &options(),
            &ImportanceOptions {
                permutations: 2,
                max_depth: 1,
                seed: 0,
            }
        )
        .is_err()
    );
}

#[test]
fn classification_cost_units_preserve_the_native_cutpoint() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/reference/guide-46.3/class-costs");
    let d: Dataset =
        serde_json::from_str(&std::fs::read_to_string(base.join("dataset.json")).unwrap()).unwrap();
    let mut o: FitOptions =
        serde_json::from_str(&std::fs::read_to_string(base.join("options.json")).unwrap()).unwrap();
    for scale in [1e-200, 1.0, 1e200] {
        o.class_costs = Some(vec![vec![0.0, scale], vec![4.0 * scale, 0.0]]);
        let m = FittedModel::fit(&d, &o).unwrap();
        assert!(
            matches!(
                m.trees[0].root.split,
                Some(Split::Numeric { threshold: 7.5, .. })
            ),
            "scale={scale}"
        );
    }
}

#[test]
fn split_search_and_validation_pruning_preserve_response_units() {
    for scale in [1.0, 1e-8, 1e100] {
        let d = data(
            (0..20)
                .map(|i| scale * if i < 10 { 2.0 } else { 8.0 })
                .collect(),
        );
        let o = options();
        let m = FittedModel::fit(&d, &o).unwrap();
        assert!(
            matches!(
                m.trees[0].root.split,
                Some(Split::Numeric { threshold: 9.5, .. })
            ),
            "scale={scale}, {:?}",
            m.trees[0].root.split
        );
        let o = FitOptions {
            pruning: Pruning::Validation,
            ..o
        };
        let m = FittedModel::fit_with_validation(&d, &o, Some(&d)).unwrap();
        assert_eq!(m.trees[0].root.leaves(), 2, "scale={scale}");
    }
}

#[test]
fn split_search_preserves_weight_units() {
    for scale in [1.0, 1e-20, 1e100] {
        let mut d = data((0..20).map(|i| if i < 10 { 2.0 } else { 8.0 }).collect());
        d.weights = vec![scale; 20];
        let m = FittedModel::fit(&d, &options()).unwrap();
        assert!(
            matches!(
                m.trees[0].root.split,
                Some(Split::Numeric { threshold: 9.5, .. })
            ),
            "weight scale={scale}"
        );
    }
}

#[test]
fn quantile_optimizer_preserves_response_origin() {
    let x: Vec<Vec<f64>> = (0..31).map(|i| vec![1.0, i as f64]).collect();
    let y: Vec<f64> = (0..31)
        .map(|i| 2.0 + 0.5 * i as f64 + ((i * 7) % 11) as f64)
        .collect();
    let w = vec![1.0; 31];
    for tau in [0.1, 0.5, 0.9] {
        let base = stats::quantile_regression(&x, &y, &w, tau).unwrap();
        let shifted: Vec<f64> = y.iter().map(|v| v + 1e9).collect();
        let fit = stats::quantile_regression(&x, &shifted, &w, tau).unwrap();
        assert!(fit.converged, "{fit:?}");
        assert!(
            (fit.coefficients[0] - 1e9 - base.coefficients[0]).abs() < 5e-6,
            "tau={tau}: {:?} vs {:?}",
            base.coefficients,
            fit.coefficients
        );
        assert!((fit.coefficients[1] - base.coefficients[1]).abs() < 5e-6);
    }
}

#[test]
fn poisson_linear_fit_preserves_exposure_units() {
    let mut d = data(
        (0..30)
            .map(|i| (0.5 + 0.04 * i as f64).exp() * (1 + i % 3) as f64)
            .collect(),
    );
    let o = FitOptions {
        task: Task::Regression {
            loss: Loss::Poisson,
        },
        node_model: NodeModel::Linear,
        max_depth: 0,
        ..options()
    };
    for scale in [1.0, 1e40, 1e-40, 1e307] {
        d.offsets = (0..30).map(|i| scale * (1 + i % 3) as f64).collect();
        let m = FittedModel::fit(&d, &o).unwrap();
        for (actual, y) in predictions(&m, &d).iter().zip(&d.targets) {
            assert!(
                (actual - y.number().unwrap()).abs() < 1e-6,
                "exposure scale={scale}: {actual} vs {y:?}"
            );
        }
        assert!(m.report.warnings.is_empty(), "{:?}", m.report.warnings);
    }
}

#[test]
fn poisson_deviance_is_accurate_near_a_perfect_fit() {
    let delta = 1e-7;
    let d = data(vec![7.3 - delta, 7.3 + delta]);
    let o = FitOptions {
        task: Task::Regression {
            loss: Loss::Poisson,
        },
        max_depth: 0,
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    // Leading quadratic term of the analytic deviance expansion. The omitted
    // relative correction is < 1e-8; the tolerance is fixed at 1e-6.
    let LeafModel::Constant { value } = m.trees[0].root.model else {
        panic!()
    };
    let expected = d
        .targets
        .iter()
        .map(|y| (y.number().unwrap() - value).powi(2) / value)
        .sum::<f64>()
        / 2.0;
    assert!(
        (m.trees[0].root.risk / expected - 1.0).abs() < 1e-6,
        "{} vs {expected}",
        m.trees[0].root.risk
    );
}

#[test]
fn poisson_categorical_search_groups_equal_rates() {
    let mut d = data(vec![0.0; 30]);
    d.schema[0].feature_type = FeatureType::Categorical;
    for i in 0..30 {
        let (category, count, exposure) = match i % 3 {
            0 => ("a", 1.0, 1.0),
            1 => ("b", 2.0, 0.02),
            _ => ("c", 100.0, 100.0),
        };
        d.rows[i][0] = Value::Text(category.into());
        d.targets[i] = Value::Number(count);
        d.offsets.push(exposure);
    }
    let o = FitOptions {
        task: Task::Regression {
            loss: Loss::Poisson,
        },
        ..options()
    };
    let m = FittedModel::fit(&d, &o).unwrap();
    // The partition {a,c}|{b} has exactly zero deviance, the global minimum.
    for (actual, y) in predictions(&m, &d).iter().zip(&d.targets) {
        assert!(
            (actual - y.number().unwrap()).abs() < 1e-8,
            "{actual} vs {y:?}"
        );
    }
}

#[test]
fn pruning_path_preserves_small_and_large_loss_units() {
    let mut baseline = None;
    for scale in [1.0, 1e-8, 1e100] {
        let d = data(
            (0..40)
                .map(|i| [0.0, 2.0, 10.0, 20.0][i / 10] * scale)
                .collect(),
        );
        let o = FitOptions {
            max_depth: 3,
            pruning: Pruning::FixedLeaves { max_leaves: 3 },
            ..options()
        };
        let m = FittedModel::fit(&d, &o).unwrap();
        let leaves = m.trees[0].root.leaves();
        assert_eq!(leaves, *baseline.get_or_insert(leaves), "scale={scale}");
        let o = FitOptions {
            pruning: Pruning::CrossValidation {
                folds: 4,
                standard_errors: 0.0,
            },
            ..o
        };
        let cv = FittedModel::fit(&d, &o).unwrap();
        assert!(
            cv.report
                .pruning_scores
                .iter()
                .all(|s| s.mean_loss.is_finite() && s.standard_error.is_finite())
        );
    }
}

#[test]
fn glm_boundaries_are_exact_and_separation_is_not_convergence() {
    for (loss, response) in [
        (Loss::Poisson, 0.0),
        (Loss::Logistic, 0.0),
        (Loss::Logistic, 1.0),
    ] {
        let d = data(vec![response; 20]);
        let o = FitOptions {
            task: Task::Regression { loss },
            node_model: NodeModel::Linear,
            max_depth: 0,
            ..options()
        };
        let m = FittedModel::fit(&d, &o).unwrap();
        assert!(matches!(m.trees[0].root.model, LeafModel::Constant { .. }));
        assert_eq!(predictions(&m, &d), vec![response; 20]);
    }
    let d = data((0..40).map(|i| if i < 20 { 0.0 } else { 1.0 }).collect());
    let o = FitOptions {
        task: Task::Regression {
            loss: Loss::Logistic,
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
    assert!(m.report.warnings.iter().any(|w| w.contains("convergence")));
}

#[test]
fn exposure_validation_and_overflow_fail_before_pruning() {
    let mut d = data(vec![1.0, 2.0, 3.0, 4.0]);
    d.offsets = vec![1.0, 2.0, 3.0, 4.0];
    let mut validation = d.clone();
    validation.offsets.clear();
    let o = FitOptions {
        task: Task::Regression {
            loss: Loss::Poisson,
        },
        pruning: Pruning::Validation,
        ..options()
    };
    assert!(FittedModel::fit_with_validation(&d, &o, Some(&validation)).is_err());
    let overflow = data(vec![-1e300, 1e300, -1e300, 1e300]);
    assert!(
        FittedModel::fit(
            &overflow,
            &FitOptions {
                pruning: Pruning::FixedLeaves { max_leaves: 1 },
                ..options()
            }
        )
        .is_err()
    );
}

#[test]
fn weighted_cv_uncertainty_preserves_weight_units() {
    let mut reference = None;
    for scale in [1.0, 1e-200, 1e200] {
        let mut d = data((0..24).map(|i| (i % 7) as f64).collect());
        d.weights = (0..24).map(|i| scale * (1 + i % 3) as f64).collect();
        let o = FitOptions {
            max_depth: 0,
            pruning: Pruning::CrossValidation {
                folds: 4,
                standard_errors: 0.25,
            },
            ..options()
        };
        let m = FittedModel::fit(&d, &o).unwrap();
        let score = &m.report.pruning_scores[0];
        assert!(score.standard_error > 0.0 && score.standard_error.is_finite());
        let (mean, se) = *reference.get_or_insert((score.mean_loss, score.standard_error));
        assert!((mean - score.mean_loss).abs() < 1e-10);
        assert!((se - score.standard_error).abs() < 1e-10);
    }
}

#[test]
fn stepwise_tests_preserve_response_units() {
    let mut reference = None;
    for scale in [1.0, 1e-14] {
        let mut d = data(
            (0..100)
                .map(|i| scale * (2.0 + 0.3 * i as f64 + ((i * 17) % 23) as f64 / 23.0))
                .collect(),
        );
        d.schema.push(FeatureSchema {
            name: "z".into(),
            feature_type: FeatureType::Numeric,
            role: Role::Both,
        });
        for (i, row) in d.rows.iter_mut().enumerate() {
            row.push(Value::Number(((i * 13) % 29) as f64));
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
        let LeafModel::Linear { terms, .. } = &m.trees[0].root.model else {
            panic!()
        };
        let features: Vec<usize> = terms.iter().map(|t| t.feature).collect();
        assert_eq!(
            &features,
            reference.get_or_insert(features.clone()),
            "scale={scale}"
        );
    }
}
