//! Prespecified Monte Carlo smoke checks; these are not a GUIDE parity certificate.
use guide_rs::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn null_data(rng: &mut ChaCha8Rng, n: usize, missing: bool) -> Dataset {
    let schema = vec![
        FeatureSchema {
            name: "binary".into(),
            feature_type: FeatureType::Numeric,
            role: Role::Split,
        },
        FeatureSchema {
            name: "categorical".into(),
            feature_type: FeatureType::Categorical,
            role: Role::Split,
        },
        FeatureSchema {
            name: "continuous".into(),
            feature_type: FeatureType::Numeric,
            role: Role::Split,
        },
    ];
    let rows = (0..n)
        .map(|_| {
            vec![
                Value::Number(f64::from(rng.gen_bool(0.5))),
                Value::Text(rng.gen_range(0..4).to_string()),
                if missing && rng.gen_bool(0.25) {
                    Value::Missing
                } else {
                    Value::Number(rng.r#gen())
                },
            ]
        })
        .collect();
    let targets = (0..n)
        .map(|_| Value::Number(f64::from(rng.gen_bool(0.5))))
        .collect();
    Dataset {
        schema,
        rows,
        targets,
        weights: vec![],
        offsets: vec![],
    }
}

#[test]
#[ignore = "Monte Carlo acceptance smoke check; run cargo test --release --test statistical -- --ignored --nocapture"]
fn null_root_selection_is_not_grossly_biased() {
    // Four comparisons, each at 0.001/4. Declared before collecting results.
    for missing in [false, true] {
        for classification in [false, true] {
            let mut rng = ChaCha8Rng::seed_from_u64(
                71027 + u64::from(missing) * 100 + u64::from(classification),
            );
            let mut count = [0usize; 3];
            let opts = FitOptions {
                task: if classification {
                    Task::Classification
                } else {
                    Task::default()
                },
                pruning: Pruning::None,
                max_depth: 1,
                ..Default::default()
            };
            for _ in 0..500 {
                let d = null_data(&mut rng, 200, missing);
                let m = FittedModel::fit(&d, &opts).unwrap();
                let root = &m.trees[0].root;
                // Count the selected variable even when its split was merged
                // away because both children predict the same class.
                let selected = root.split.as_ref().map(Split::feature).or_else(|| {
                    root.diagnostics
                        .iter()
                        .filter_map(|g| g.log_p_value.map(|p| (p, g.feature)))
                        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                        .map(|(_, j)| j)
                });
                if let Some(j) = selected {
                    count[j] += 1;
                }
            }
            let n = count.iter().sum::<usize>() as f64;
            let expected = n / 3.0;
            let statistic = count
                .iter()
                .map(|&c| (c as f64 - expected).powi(2) / expected)
                .sum::<f64>();
            let p = stats::chi_square_survival(statistic, 2);
            println!(
                "classification={classification}, missing={missing}, counts={count:?}, goodness-of-fit p={p:.6}"
            );
            assert!(n >= 490.0);
            assert!(
                p > 0.00025,
                "gross selection imbalance; do not relax the bound to pass"
            );
        }
    }
}

#[test]
#[ignore = "slower importance calibration smoke check; run cargo test --release --test statistical -- --ignored --nocapture"]
fn importance_null_familywise_error_smoke_check() {
    // 50 independent null datasets; at nominal 5%, >=10 discoveries is a
    // strong excess (binomial upper-tail probability below 0.001).
    let mut rng = ChaCha8Rng::seed_from_u64(61083);
    let mut rejected = 0;
    for trial in 0..50 {
        let d = null_data(&mut rng, 80, true);
        let result = importance(
            &d,
            &FitOptions {
                pruning: Pruning::None,
                ..Default::default()
            },
            &ImportanceOptions {
                seed: trial,
                max_depth: 4,
                permutations: 300,
            },
        )
        .unwrap();
        if result.thresholds[2].important_count > 0 {
            rejected += 1;
        }
    }
    println!("Null 95% importance threshold: {rejected}/50 datasets selected any variable");
    assert!(
        rejected < 10,
        "excess null discoveries; investigate calibration instead of relaxing the bound"
    );
}
