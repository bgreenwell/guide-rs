use guide_rs::*;

fn numeric(names: &[&str]) -> Vec<FeatureSchema> {
    names
        .iter()
        .map(|n| FeatureSchema {
            name: (*n).into(),
            feature_type: FeatureType::Numeric,
            role: Role::Split,
            flags: None,
        })
        .collect()
}
fn dataset(n: usize, target: impl Fn(f64, f64) -> Value) -> Dataset {
    let x = |i: usize| ((i * 37) % 101) as f64 / 100.0;
    let z = |i: usize| ((i * 53) % 97) as f64 / 96.0;
    Dataset {
        schema: numeric(&["x", "z"]),
        rows: (0..n)
            .map(|i| vec![Value::Number(x(i)), Value::Number(z(i))])
            .collect(),
        targets: (0..n).map(|i| target(x(i), z(i))).collect(),
        weights: vec![],
        offsets: vec![],
    }
}
fn options(task: Task) -> FitOptions {
    FitOptions {
        task,
        pruning: Pruning::None,
        ..Default::default()
    }
}
fn values(p: &[Prediction]) -> Vec<f64> {
    p.iter()
        .map(|p| match p {
            Prediction::Regression { value } => *value,
            _ => panic!("expected regression"),
        })
        .collect()
}

#[test]
fn boosting_fits_a_smooth_surface_and_round_trips() {
    let d = dataset(400, |x, z| Value::Number(4.0 * x * x + (6.0 * z).sin()));
    let truth: Vec<f64> = d
        .rows
        .iter()
        .map(|r| {
            let (x, z) = (r[0].number().unwrap(), r[1].number().unwrap());
            4.0 * x * x + (6.0 * z).sin()
        })
        .collect();
    let error = |rounds| {
        let b = BoostingOptions {
            rounds,
            ..Default::default()
        };
        let m = FittedModel::fit_boosting(&d, &options(Task::default()), &b, None).unwrap();
        let p = values(&m.predict(&d.rows).unwrap());
        p.iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / p.len() as f64
    };
    // More rounds fit the noise-free surface better.
    assert!(error(200) < error(20) / 4.0);
    assert!(error(200) < 0.02);
    let b = BoostingOptions::default();
    let m = FittedModel::fit_boosting(&d, &options(Task::default()), &b, None).unwrap();
    let again = FittedModel::fit_boosting(&d, &options(Task::default()), &b, None).unwrap();
    assert_eq!(m.to_json().unwrap(), again.to_json().unwrap());
    assert_eq!(m.trees.len(), 100);
    let restored = FittedModel::from_json(&m.to_json().unwrap()).unwrap();
    assert_eq!(
        values(&restored.predict(&d.rows).unwrap()),
        values(&m.predict(&d.rows).unwrap())
    );
    assert!(m.to_text().unwrap().contains("score"));
}

#[test]
fn boosting_classifies_two_and_three_classes() {
    let binary = dataset(300, |x, z| {
        Value::Text(if x + z > 1.0 { "b" } else { "a" }.into())
    });
    let classes = dataset(300, |x, z| {
        Value::Text(
            if x < 0.33 {
                "a"
            } else if z < 0.5 {
                "b"
            } else {
                "c"
            }
            .into(),
        )
    });
    for (d, k) in [(&binary, 2), (&classes, 3)] {
        let b = BoostingOptions {
            rounds: 60,
            ..Default::default()
        };
        let m = FittedModel::fit_boosting(d, &options(Task::Classification), &b, None).unwrap();
        assert_eq!(m.trees.len(), 60 * if k == 2 { 1 } else { k });
        let mut correct = 0;
        for (p, y) in m.predict(&d.rows).unwrap().iter().zip(&d.targets) {
            let Prediction::Classification {
                label,
                probabilities,
            } = p
            else {
                panic!("expected classification")
            };
            assert_eq!(probabilities.len(), k);
            assert!((probabilities.iter().map(|q| q.1).sum::<f64>() - 1.0).abs() < 1e-9);
            correct += usize::from(Some(label.as_str()) == y.label().as_deref());
        }
        assert!(correct as f64 / d.rows.len() as f64 > 0.95, "{k} classes");
    }
}

#[test]
fn poisson_and_logistic_boosting_use_their_links() {
    let mut d = dataset(400, |x, _| Value::Number(if x > 0.5 { 6.0 } else { 2.0 }));
    d.offsets = (0..400).map(|i| 1.0 + (i % 3) as f64).collect();
    // Counts scale with exposure: rate 2 or 6 per unit.
    for (t, e) in d.targets.iter_mut().zip(&d.offsets) {
        *t = Value::Number(t.number().unwrap() * e);
    }
    let poisson = options(Task::Regression {
        loss: Loss::Poisson,
    });
    let m = FittedModel::fit_boosting(&d, &poisson, &BoostingOptions::default(), None).unwrap();
    assert!(m.predict(&d.rows).is_err(), "exposure is required");
    let rows = vec![vec![Value::Number(0.9), Value::Number(0.5)]; 2];
    let p = values(&m.predict_with_offsets(&rows, Some(&[1.0, 2.0])).unwrap());
    assert!(
        (p[0] - 6.0).abs() < 0.3 && (p[1] / p[0] - 2.0).abs() < 1e-9,
        "{p:?}"
    );
    let logistic = options(Task::Regression {
        loss: Loss::Logistic,
    });
    let d = dataset(300, |x, _| Value::Number(f64::from(u8::from(x > 0.5))));
    let m = FittedModel::fit_boosting(&d, &logistic, &BoostingOptions::default(), None).unwrap();
    let p = values(&m.predict(&d.rows).unwrap());
    assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(
        p.iter()
            .zip(&d.targets)
            .all(|(v, y)| (v - y.number().unwrap()).abs() < 0.2)
    );
}

#[test]
fn early_stopping_keeps_the_best_validation_round() {
    let noisy = |seed: usize| {
        dataset(300, move |x, z| {
            let noise = (((x * 1000.0) as usize * 31 + (z * 1000.0) as usize * 17 + seed) % 23)
                as f64
                / 11.0
                - 1.0;
            Value::Number(2.0 * x + noise)
        })
    };
    let (train, valid) = (noisy(0), noisy(5));
    let b = BoostingOptions {
        rounds: 500,
        learning_rate: 0.3,
        max_depth: 4,
        early_stopping_rounds: Some(10),
        ..Default::default()
    };
    let o = options(Task::default());
    let m = FittedModel::fit_boosting(&train, &o, &b, Some(&valid)).unwrap();
    let boosted = m.boosting.as_ref().unwrap();
    assert!(boosted.rounds < 500 && boosted.rounds >= 1);
    assert_eq!(m.trees.len(), boosted.rounds);
    let best = boosted.validation_loss[boosted.rounds - 1];
    assert!(boosted.validation_loss.iter().all(|l| *l >= best - 1e-12));
    assert!(
        boosted.validation_loss.len() >= boosted.rounds + 10
            || boosted.validation_loss.len() == 500
    );
    let error = FittedModel::fit_boosting(&train, &o, &b, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("used together"), "{error}");
}

#[test]
fn boosting_results_do_not_depend_on_threads() {
    let d = dataset(1200, |x, z| Value::Number((3.0 * x).cos() + z * z));
    let b = BoostingOptions {
        rounds: 20,
        subsample: 0.7,
        ..Default::default()
    };
    let fit = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                FittedModel::fit_boosting(&d, &options(Task::default()), &b, None)
                    .unwrap()
                    .to_json()
                    .unwrap()
            })
    };
    assert_eq!(fit(1), fit(4));
}

#[test]
fn invalid_boosting_configurations_and_models_are_rejected() {
    let d = dataset(100, |x, _| Value::Number(x));
    let b = BoostingOptions::default();
    let pruned = FitOptions::default();
    assert!(FittedModel::fit_boosting(&d, &pruned, &b, None).is_err());
    let quantile = options(Task::Regression {
        loss: Loss::Quantile { probability: 0.5 },
    });
    assert!(FittedModel::fit_boosting(&d, &quantile, &b, None).is_err());
    let zero = BoostingOptions {
        learning_rate: 0.0,
        ..Default::default()
    };
    assert!(FittedModel::fit_boosting(&d, &options(Task::default()), &zero, None).is_err());
    let m = FittedModel::fit_boosting(&d, &options(Task::default()), &b, None).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&m.to_json().unwrap()).unwrap();
    json["boosting"]["rounds"] = 99.into();
    assert!(FittedModel::from_json(&json.to_string()).is_err());
    // Score leaves need boosting metadata.
    let mut json: serde_json::Value = serde_json::from_str(&m.to_json().unwrap()).unwrap();
    json.as_object_mut().unwrap().remove("boosting");
    json["trees"] = serde_json::Value::Array(vec![json["trees"][0].clone()]);
    assert!(FittedModel::from_json(&json.to_string()).is_err());
}
