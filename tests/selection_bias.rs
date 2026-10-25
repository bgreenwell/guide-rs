//! Prespecified root selection bias study; see docs/selection-bias-study.md.
//! The design, seeds and criteria here must match that protocol.
use guide_rs::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const N: usize = 400;
const REPLICATIONS: u64 = 1000;
const NAMES: [&str; 5] = ["binary", "cat4", "cat50", "cont", "cont_missing"];

#[derive(Clone, Copy, PartialEq)]
enum Response {
    Gaussian,
    Binary,
    ThreeClass,
    Linear,
    WeakGaussian,
    WeakBinary,
}

struct Config {
    k: u64,
    name: &'static str,
    response: Response,
    interactions: bool,
}

const CONFIGS: [Config; 7] = [
    Config {
        k: 1,
        name: "constant least squares",
        response: Response::Gaussian,
        interactions: true,
    },
    Config {
        k: 2,
        name: "binary classification",
        response: Response::Binary,
        interactions: true,
    },
    Config {
        k: 3,
        name: "three-class classification",
        response: Response::ThreeClass,
        interactions: true,
    },
    Config {
        k: 4,
        name: "multiple-linear least squares",
        response: Response::Linear,
        interactions: true,
    },
    Config {
        k: 5,
        name: "constant least squares, no interactions",
        response: Response::Gaussian,
        interactions: false,
    },
    Config {
        k: 6,
        name: "least squares, weak binary signal",
        response: Response::WeakGaussian,
        interactions: true,
    },
    Config {
        k: 7,
        name: "classification, weak binary signal",
        response: Response::WeakBinary,
        interactions: true,
    },
];

fn normal(rng: &mut ChaCha8Rng) -> f64 {
    // Box-Muller; 1 - u keeps the logarithm finite.
    let u: f64 = rng.r#gen();
    let v: f64 = rng.r#gen();
    (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn dataset(config: &Config, seed: u64) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let linear = config.response == Response::Linear;
    let mut schema: Vec<FeatureSchema> = NAMES
        .iter()
        .map(|name| FeatureSchema {
            name: (*name).into(),
            feature_type: if name.starts_with("cat") {
                FeatureType::Categorical
            } else {
                FeatureType::Numeric
            },
            role: Role::Split,
            flags: None,
        })
        .collect();
    if linear {
        schema.push(FeatureSchema {
            name: "z".into(),
            feature_type: FeatureType::Numeric,
            role: Role::Fit,
            flags: None,
        });
    }
    let mut rows = vec![];
    let mut targets = vec![];
    for _ in 0..N {
        let binary = f64::from(rng.gen_bool(0.5));
        let mut row = vec![
            Value::Number(binary),
            Value::Text(rng.gen_range(0..4).to_string()),
            Value::Text(rng.gen_range(0..50).to_string()),
            Value::Number(normal(&mut rng)),
            if rng.gen_bool(0.3) {
                Value::Missing
            } else {
                Value::Number(normal(&mut rng))
            },
        ];
        let y = match config.response {
            Response::Gaussian => normal(&mut rng),
            Response::Binary => f64::from(rng.gen_bool(0.5)),
            Response::ThreeClass => rng.gen_range(0..3) as f64,
            Response::Linear => {
                let z = normal(&mut rng);
                row.push(Value::Number(z));
                1.0 + 2.0 * z + normal(&mut rng)
            }
            Response::WeakGaussian => 0.25 * binary + normal(&mut rng),
            Response::WeakBinary => f64::from(rng.gen_bool(0.42 + 0.16 * binary)),
        };
        rows.push(row);
        targets.push(Value::Number(y));
    }
    Dataset {
        schema,
        rows,
        targets,
        weights: vec![],
        offsets: vec![],
    }
}

fn classifying(config: &Config) -> bool {
    matches!(
        config.response,
        Response::Binary | Response::ThreeClass | Response::WeakBinary
    )
}

/// The counted variable defined in the protocol, or None. Fit errors mean
/// the configuration is not evaluable and are reported as failures.
fn guide_selection(config: &Config, data: &Dataset, seed: u64) -> Result<Option<usize>> {
    let options = FitOptions {
        task: if classifying(config) {
            Task::Classification
        } else {
            Task::default()
        },
        node_model: if config.response == Response::Linear {
            NodeModel::Linear
        } else {
            NodeModel::Constant
        },
        interactions: config.interactions,
        max_depth: 1,
        pruning: Pruning::None,
        seed,
        ..Default::default()
    };
    let model = FittedModel::fit(data, &options)?;
    let root = &model.trees[0].root;
    Ok(root.split.as_ref().map(Split::feature).or_else(|| {
        root.diagnostics
            .iter()
            .filter_map(|g| g.log_p_value.map(|p| (p, g.feature)))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .map(|(_, j)| j)
    }))
}

/// Impurity of a node from (count, sum, sum of squares) or class counts.
#[derive(Clone, Copy, Default)]
struct Stats {
    n: f64,
    sum: f64,
    squares: f64,
}
impl Stats {
    fn add(&mut self, y: f64) {
        self.n += 1.0;
        self.sum += y;
        self.squares += y * y;
    }
    fn merge(self, o: Stats) -> Stats {
        Stats {
            n: self.n + o.n,
            sum: self.sum + o.sum,
            squares: self.squares + o.squares,
        }
    }
    fn minus(self, o: Stats) -> Stats {
        Stats {
            n: self.n - o.n,
            sum: self.sum - o.sum,
            squares: self.squares - o.squares,
        }
    }
    /// Sum of squares for regression; n times Gini for a 0/1 response, where
    /// sum counts class 1.
    fn impurity(self, gini: bool) -> f64 {
        if self.n == 0.0 {
            0.0
        } else if gini {
            let p = self.sum / self.n;
            self.n * 2.0 * p * (1.0 - p)
        } else {
            self.squares - self.sum * self.sum / self.n
        }
    }
}

/// Best reduction over ordered groups of observed values, with the missing
/// group sent to whichever side reduces impurity more.
fn best_ordered_split(groups: &[Stats], missing: Stats, gini: bool) -> f64 {
    let observed = groups.iter().fold(Stats::default(), |a, &g| a.merge(g));
    let parent = observed.merge(missing).impurity(gini);
    let mut best = 0.0_f64;
    let mut left = Stats::default();
    for g in &groups[..groups.len().saturating_sub(1)] {
        left = left.merge(*g);
        let right = observed.minus(left);
        for (l, r) in [(left.merge(missing), right), (left, right.merge(missing))] {
            best = best.max(parent - l.impurity(gini) - r.impurity(gini));
        }
    }
    best
}

fn exhaustive_selection(data: &Dataset, gini: bool) -> usize {
    let y: Vec<f64> = data.targets.iter().map(|t| t.number().unwrap()).collect();
    let reduction = |j: usize| {
        let mut missing = Stats::default();
        let mut keyed: Vec<(String, f64, f64)> = vec![];
        for (row, &yi) in data.rows.iter().zip(&y) {
            match &row[j] {
                Value::Missing => missing.add(yi),
                Value::Number(x) => keyed.push((String::new(), *x, yi)),
                Value::Text(s) => keyed.push((s.clone(), 0.0, yi)),
            }
        }
        let mut groups: Vec<(f64, Stats)> = vec![];
        if matches!(data.schema[j].feature_type, FeatureType::Categorical) {
            let mut levels = std::collections::BTreeMap::<String, Stats>::new();
            for (s, _, yi) in &keyed {
                levels.entry(s.clone()).or_default().add(*yi);
            }
            // Ordering by mean (or class-1 share) is optimal for SSE and
            // two-class Gini (Breiman et al. 1984, Theorem 4.5).
            groups = levels.into_values().map(|s| (s.sum / s.n, s)).collect();
        } else {
            keyed.sort_by(|a, b| a.1.total_cmp(&b.1));
            for (_, x, yi) in &keyed {
                match groups.last_mut() {
                    Some((value, s)) if *value == *x => s.add(*yi),
                    _ => {
                        let mut s = Stats::default();
                        s.add(*yi);
                        groups.push((*x, s));
                    }
                }
            }
        }
        groups.sort_by(|a, b| a.0.total_cmp(&b.0));
        let ordered: Vec<Stats> = groups.into_iter().map(|(_, s)| s).collect();
        best_ordered_split(&ordered, missing, gini)
    };
    (0..NAMES.len())
        .map(|j| (reduction(j), j))
        .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
        .unwrap()
        .1
}

fn goodness_of_fit(counts: &[usize; 5]) -> f64 {
    let n = counts.iter().sum::<usize>() as f64;
    let expected = n / 5.0;
    let statistic = counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum::<f64>();
    stats::chi_square_survival(statistic, 4)
}

struct Outcome {
    guide: [usize; 5],
    none: usize,
    exhaustive: Option<[usize; 5]>,
}

fn run(config: &Config) -> Result<Outcome> {
    let with_baseline = matches!(
        config.response,
        Response::Gaussian | Response::Binary | Response::WeakGaussian | Response::WeakBinary
    ) && config.interactions;
    let mut outcome = Outcome {
        guide: [0; 5],
        none: 0,
        exhaustive: with_baseline.then_some([0; 5]),
    };
    for r in 0..REPLICATIONS {
        let seed = 20261025 + 1_000_000 * config.k + r;
        let data = dataset(config, seed);
        match guide_selection(config, &data, seed)? {
            Some(j) => outcome.guide[j] += 1,
            None => outcome.none += 1,
        }
        if let Some(counts) = &mut outcome.exhaustive {
            counts[exhaustive_selection(&data, classifying(config))] += 1;
        }
    }
    Ok(outcome)
}

fn frequencies(counts: &[usize; 5]) -> String {
    let n = counts.iter().sum::<usize>() as f64;
    NAMES
        .iter()
        .zip(counts)
        .map(|(name, &c)| format!("{name}={:.3}", c as f64 / n))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "prespecified Monte Carlo study (seconds in release); run cargo test --release --test selection_bias -- --ignored --nocapture"]
fn root_selection_bias_study() {
    let mut failures = vec![];
    for config in &CONFIGS {
        let outcome = match run(config) {
            Ok(outcome) => outcome,
            Err(e) => {
                println!("k={} {}: not evaluable: {e}", config.k, config.name);
                failures.push(format!("k={}: not evaluable: {e}", config.k));
                continue;
            }
        };
        let p = goodness_of_fit(&outcome.guide);
        println!(
            "k={} {}: guide-rs {} none={} p={p:.4}",
            config.k,
            config.name,
            frequencies(&outcome.guide),
            outcome.none
        );
        if let Some(exhaustive) = &outcome.exhaustive {
            println!(
                "    exhaustive {} p={:.3e}",
                frequencies(exhaustive),
                goodness_of_fit(exhaustive)
            );
        }
        let total = outcome.guide.iter().sum::<usize>() as f64;
        if config.k <= 5 {
            if p <= 0.002 {
                failures.push(format!("k={}: goodness-of-fit p={p:.4} <= 0.002", config.k));
            }
            for (name, &c) in NAMES.iter().zip(&outcome.guide) {
                let f = c as f64 / total;
                if !(0.14..=0.26).contains(&f) {
                    failures.push(format!(
                        "k={}: {name} frequency {f:.3} outside [0.14, 0.26]",
                        config.k
                    ));
                }
            }
            if outcome.none as f64 > 0.01 * REPLICATIONS as f64 {
                failures.push(format!(
                    "k={}: {} fits without a selection",
                    config.k, outcome.none
                ));
            }
            if let Some(exhaustive) = &outcome.exhaustive {
                let q = goodness_of_fit(exhaustive);
                if q >= 1e-6 {
                    failures.push(format!(
                        "k={}: exhaustive baseline p={q:.3e} >= 1e-6; design underpowered",
                        config.k
                    ));
                }
            }
        } else {
            let guide = outcome.guide[0] as f64 / total;
            let exhaustive = outcome
                .exhaustive
                .map(|c| c[0] as f64 / REPLICATIONS as f64);
            if guide <= 0.2 || exhaustive.is_some_and(|e| guide <= e) {
                failures.push(format!(
                    "k={}: binary selected {guide:.3} (exhaustive {exhaustive:?})",
                    config.k
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "prespecified criteria failed; do not relax them:\n{}",
        failures.join("\n")
    );
}
