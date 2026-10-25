//! Optional independent validation against R's stats, quantreg and rpart packages.
use guide_rs::*;
use std::{
    fs,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "guide-rs-rtest-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn r(temp: &Temp, script: &str) -> Vec<f64> {
    fs::write(temp.0.join("test.R"), script).unwrap();
    let out = Command::new("Rscript")
        .arg("--vanilla")
        .arg("test.R")
        .current_dir(&temp.0)
        .output()
        .expect("Rscript is required for ignored interoperability tests");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}
fn numeric(pred: Vec<Prediction>) -> Vec<f64> {
    pred.into_iter()
        .map(|p| {
            if let Prediction::Regression { value } = p {
                value
            } else {
                panic!()
            }
        })
        .collect()
}
fn near(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{a} != {b}");
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn f_log_tails_agree_with_r_including_underflow_and_overflow_arguments() {
    let t = Temp::new();
    let mut csv = "f,numerator,denominator\n".to_owned();
    let mut actual = vec![];
    for (numerator, denominator) in [
        (1, 1),
        (1, 3198),
        (2, 2),
        (5, 10),
        (100, 1000),
        (100000, 100000),
    ] {
        for f in [1e-100, 0.01, 0.5, 1.0, 2.0, 10.0, 1000.0, 1e100, f64::MAX] {
            csv.push_str(&format!("{f},{numerator},{denominator}\n"));
            actual.push(stats::f_log_survival(f, numerator, denominator));
        }
    }
    fs::write(t.0.join("tails.csv"), csv).unwrap();
    let expected = r(
        &t,
        "d<-read.csv('tails.csv');cat(format(pf(d$f,d$numerator,d$denominator,lower.tail=FALSE,log.p=TRUE),digits=17))",
    );
    near(&actual, &expected, 2e-8);
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn native_linear_cases_and_fallback_counterexamples_agree_with_r_lm() {
    let t = Temp::new();
    let base =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/linear/guide-46.3");
    for entry in fs::read_dir(base).unwrap() {
        let path = entry.unwrap().path();
        let data: Dataset =
            serde_json::from_str(&fs::read_to_string(path.join("dataset.json")).unwrap()).unwrap();
        let options: FitOptions =
            serde_json::from_str(&fs::read_to_string(path.join("options.json")).unwrap()).unwrap();
        let model = FittedModel::fit(&data, &options).unwrap();
        fs::copy(path.join("data.csv"), t.0.join("data.csv")).unwrap();
        let selected: std::collections::BTreeSet<_> = match &model.trees[0].root.model {
            LeafModel::Linear { terms, .. } => terms
                .iter()
                .map(|term| {
                    format!(
                        "{}{}",
                        data.schema[term.feature].name,
                        if term.missing_indicator { ".NA" } else { "" }
                    )
                })
                .collect(),
            LeafModel::Constant { .. } => Default::default(),
            _ => panic!("unexpected leaf model"),
        };
        let formula = if selected.is_empty() {
            "1".into()
        } else {
            selected.into_iter().collect::<Vec<_>>().join("+")
        };
        // This checks the fitted leaf given its selected columns; R is not
        // claimed to implement GUIDE's stepwise variable selection algorithm.
        let expected = r(
            &t,
            &format!(
                "d<-read.csv('data.csv');w<-if('weight'%in%names(d))d$weight else rep(1,nrow(d));for(n in c('x','z')){{v<-d[[n]];d[[paste0(n,'.NA')]]<-as.numeric(is.na(v));m<-if(all(is.na(v)))0 else weighted.mean(v[!is.na(v)],w[!is.na(v)]);d[[n]][is.na(v)]<-m}};fit<-lm(y~{formula},data=d,weights=w);cat(format(fitted(fit),digits=17))"
            ),
        );
        near(
            &numeric(model.predict(&data.rows).unwrap()),
            &expected,
            2e-10,
        );
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn multiple_linear_tree_leaves_and_loocv_agree_with_independent_r_fits() {
    let t = Temp::new();
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/linear-tree/guide-46.3");
    for entry in fs::read_dir(base).unwrap() {
        let path = entry.unwrap().path();
        let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
        let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
        let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
        let mut fit_options = options.clone();
        if matches!(
            manifest["case"].as_str(),
            Some("linear-multilevel-loocv" | "linear-multilevel-loocv-weighted")
        ) {
            // Check the preserved native terminal partition separately from the
            // correct four-leaf CV result verified by the independent oracle.
            fit_options.pruning = Pruning::FixedLeaves { max_leaves: 3 };
        }
        let model = FittedModel::fit(&data, &fit_options).unwrap();
        fs::copy(path.join("data.csv"), t.0.join("data.csv")).unwrap();
        let leaves = manifest["result"]["terminal_node_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(t.0.join("leaves.txt"), leaves).unwrap();
        let formula = data
            .schema
            .iter()
            .filter(|f| f.role.fits())
            .map(|f| f.name.clone())
            .collect::<Vec<_>>()
            .join("+");
        let flags = data
            .schema
            .iter()
            .enumerate()
            .filter(|(j, f)| f.role.fits() && data.rows.iter().any(|row| row[*j].is_missing()))
            .map(|(_, f)| format!("{}.NA", f.name))
            .collect::<Vec<_>>();
        let formula = if flags.is_empty() {
            formula
        } else {
            format!("{formula}+{}", flags.join("+"))
        };
        let expected = r(
            &t,
            &format!(
                "d<-read.csv('data.csv');g<-scan('leaves.txt',quiet=TRUE);p<-numeric(nrow(d));for(k in unique(g)){{v<-d[g==k,];w<-if('weight'%in%names(v))v$weight else rep(1,nrow(v));for(n in c('x','z')){{seen<-!is.na(v[[n]]);v[[paste0(n,'.NA')]]<-as.numeric(!seen);v[[n]][!seen]<-weighted.mean(v[[n]][seen],w[seen])}};p[g==k]<-fitted(lm(y~{formula},data=v,weights=w))}};cat(format(p,digits=17))"
            ),
        );
        near(
            &numeric(model.predict(&data.rows).unwrap()),
            &expected,
            2e-10,
        );
        if manifest["case"] == "linear-tree-loocv-default" {
            // Independently enumerate every valid cut with R lm in each fold.
            // Evaluate the full stump and root candidates, then compute naive
            // population SE directly from the forty held-out squared errors.
            let expected = r(
                &t,
                r#"
                d<-read.csv('data.csv');n<-nrow(d);errors<-matrix(0,n,2)
                for(h in seq_len(n)) {
                    v<-d[-h,];cuts<-(head(v$x,-1)+tail(v$x,-1))/2
                    cuts<-cuts[vapply(cuts,function(c)sum(v$x<=c)>=10&&sum(v$x>c)>=10,logical(1))]
                    losses<-vapply(cuts,function(c) {
                        sum(resid(lm(y~z,data=v[v$x<=c,]))^2)+sum(resid(lm(y~z,data=v[v$x>c,]))^2)
                    },numeric(1))
                    cut<-cuts[which.min(losses)]
                    side<-if(d$x[h]<=cut)v[v$x<=cut,] else v[v$x>cut,]
                    errors[h,1]<-(d$y[h]-predict(lm(y~z,data=side),newdata=d[h,]))^2
                    errors[h,2]<-(d$y[h]-predict(lm(y~z,data=v),newdata=d[h,]))^2
                }
                means<-colMeans(errors);ses<-sqrt(colSums(sweep(errors,2,means)^2))/n
                cat(format(c(means,ses),digits=17))
            "#,
            );
            let scores = &model.report.pruning_scores;
            assert_eq!(scores.len(), 2);
            assert_eq!(scores[0].leaves, 2);
            assert_eq!(scores[1].leaves, 1);
            near(
                &[
                    scores[0].mean_loss,
                    scores[1].mean_loss,
                    scores[0].standard_error,
                    scores[1].standard_error,
                ],
                &expected,
                2e-10,
            );
        }
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn multilevel_cv_and_each_pruning_stage_have_reproducible_independent_r_evidence() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let evidence: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(
            root.join("tests/fixtures/linear-tree/diagnostics/multilevel-cv-oracle.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (name, expected) in evidence["cases"].as_object().unwrap() {
        let t = Temp::new();
        let directory = root
            .join("tests/fixtures/linear-tree/guide-46.3")
            .join(name);
        let options: FitOptions =
            serde_json::from_str(&fs::read_to_string(directory.join("options.json")).unwrap())
                .unwrap();
        fs::copy(directory.join("data.csv"), t.0.join("data.csv")).unwrap();
        let process = Command::new("Rscript")
            .arg("--vanilla")
            .arg(root.join("tools/multilevel_linear_oracle.R"))
            .arg(options.max_depth.to_string())
            .arg(options.min_samples_leaf.to_string())
            .arg("0")
            .current_dir(&t.0)
            .output()
            .unwrap();
        assert!(
            process.status.success(),
            "{}",
            String::from_utf8_lossy(&process.stderr)
        );
        let compare = |file: &str, expected: &serde_json::Value| {
            let csv = fs::read_to_string(t.0.join(file)).unwrap();
            let mut lines = csv.lines();
            let names: Vec<_> = lines
                .next()
                .unwrap()
                .split(',')
                .map(|s| s.trim_matches('"'))
                .collect();
            let rows: Vec<_> = lines.collect();
            assert_eq!(rows.len(), expected.as_array().unwrap().len());
            for (i, line) in rows.iter().enumerate() {
                let actual: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
                assert_eq!(actual.len(), names.len());
                let reference: Vec<_> = names
                    .iter()
                    .map(|n| expected[i][*n].as_f64().unwrap())
                    .collect();
                near(&actual, &reference, 2e-10);
            }
        };
        compare("oracle-scores.csv", &expected["scores"]);
        compare(
            "oracle-held-predictions.csv",
            &expected["heldout_predictions"],
        );
        compare(
            "oracle-native-scan-predictions.csv",
            &expected["native_scan_heldout_predictions"],
        );
        for (i, score) in expected["scores"].as_array().unwrap().iter().enumerate() {
            let leaves = (score["leaves"].as_f64().unwrap() as usize).to_string();
            compare(
                &format!("oracle-tree-{}.csv", i + 1),
                &expected["trees"][leaves],
            );
        }
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn selection_sign_tables_and_fixed_partition_fits_agree_with_r() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let base = root.join("tests/fixtures/selection/guide-46.3");
    for entry in fs::read_dir(base).unwrap() {
        let path = entry.unwrap().path();
        let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
        let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
        let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
        let model = FittedModel::fit(&data, &options).unwrap();
        let t = Temp::new();
        fs::copy(root.join("tools/selection_oracle.R"), t.0.join("oracle.R")).unwrap();
        fs::copy(path.join("data.csv"), t.0.join("data.csv")).unwrap();
        let mut schema = "name type role\n".to_owned();
        for feature in &data.schema {
            let kind = if feature.feature_type == FeatureType::Numeric {
                "numeric"
            } else {
                "categorical"
            };
            let role = serde_json::to_value(feature.role).unwrap();
            schema.push_str(&format!(
                "{} {kind} {}\n",
                feature.name,
                role.as_str().unwrap()
            ));
        }
        fs::write(t.0.join("schema.txt"), schema).unwrap();
        let leaves = manifest["result"]["terminal_node_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(t.0.join("leaves.txt"), leaves).unwrap();
        let expected = r(&t, "source('oracle.R');cat(format(prediction,digits=17))");
        near(
            &numeric(model.predict(&data.rows).unwrap()),
            &expected,
            2e-10,
        );
        let numbers = |name: &str| -> Vec<f64> {
            fs::read_to_string(t.0.join(name))
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect()
        };
        let scores = numbers("oracle-scores.txt");
        let diagnostics = &model.trees[0].root.diagnostics;
        assert_eq!(scores.len(), 3 * diagnostics.len());
        for (diagnostic, score) in diagnostics.iter().zip(scores.as_chunks::<3>().0) {
            near(
                &[
                    diagnostic.chi_square,
                    diagnostic.degrees_of_freedom as f64,
                    diagnostic.log_p_value.unwrap(),
                ],
                score,
                2e-10,
            );
        }
        if manifest["result"]["expected_root_association"] == "interaction" {
            let joint = numbers("oracle-joint.txt");
            assert!((joint[0] - 96.0).abs() < 1e-10);
            for diagnostic in diagnostics {
                assert!((diagnostic.importance_log_p_value.unwrap() - joint[2]).abs() < 2e-10);
            }
        }
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn categorical_linear_predictions_subsets_and_loocv_agree_with_r_qr() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let paths: Vec<_> = ["categorical-linear", "orientation"]
        .into_iter()
        .flat_map(|family| {
            fs::read_dir(root.join("tests/fixtures").join(family).join("guide-46.3"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
        })
        .collect();
    assert_eq!(paths.len(), 80);
    for path in paths {
        let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
        let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
        let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
        let balanced_case = manifest["case"] == "categorical-linear-split-and-fit";
        let model = FittedModel::fit(&data, &options).unwrap();
        let t = Temp::new();
        fs::copy(path.join("data.csv"), t.0.join("data.csv")).unwrap();
        for (file, field) in [
            ("leaves.txt", "terminal_node_ids"),
            ("training.txt", "training_rows"),
        ] {
            let values = manifest["result"][field]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    if let Some(flag) = v.as_bool() {
                        if flag { "1".into() } else { "0".into() }
                    } else {
                        v.to_string()
                    }
                })
                .collect::<Vec<String>>()
                .join("\n");
            fs::write(t.0.join(file), values).unwrap();
        }
        let names = |categorical: bool| {
            let names = data
                .schema
                .iter()
                .filter(|f| {
                    f.role.fits()
                        && matches!(f.feature_type, FeatureType::Categorical) == categorical
                })
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
                .join(",");
            if names.is_empty() {
                "-".to_owned()
            } else {
                names
            }
        };
        let mode = serde_json::to_value(options.prediction_truncation).unwrap();
        let cv = matches!(options.pruning, Pruning::CrossValidation { .. });
        let output = Command::new("Rscript")
            .arg("--vanilla")
            .arg(root.join("tools/categorical_linear_oracle.R"))
            .arg(mode.as_str().unwrap())
            .arg(names(true))
            .arg(names(false))
            .arg(if cv { "cv" } else { "no_cv" })
            .current_dir(&t.0)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let numbers = |file: &str| -> Vec<f64> {
            fs::read_to_string(t.0.join(file))
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect()
        };
        near(
            &numeric(model.predict(&data.rows).unwrap()),
            &numbers("oracle-predictions.txt"),
            2e-10,
        );
        if cv {
            let scores = &model.report.pruning_scores;
            assert_eq!(scores.len(), 2);
            near(
                &[
                    scores[0].mean_loss,
                    scores[1].mean_loss,
                    scores[0].standard_error,
                    scores[1].standard_error,
                ],
                &numbers("oracle-cv.txt"),
                2e-10,
            );
            if manifest["case"] == "orientation-loocv" {
                let record: serde_json::Value = serde_json::from_str(
                    &fs::read_to_string(root.join(
                        "tests/fixtures/orientation/diagnostics/categorical-cv-difference.json",
                    ))
                    .unwrap(),
                )
                .unwrap();
                near(
                    &numbers("oracle-cv.txt"),
                    &record["correct_scores"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_f64().unwrap())
                        .collect::<Vec<_>>(),
                    2e-10,
                );
                near(
                    &numbers("oracle-heldout.txt"),
                    &record["correct_heldout_predictions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|row| row.as_array().unwrap().iter().map(|n| n.as_f64().unwrap()))
                        .collect::<Vec<_>>(),
                    2e-10,
                );
            }
        }
        if balanced_case || manifest["case"] == "categorical-linear-split-and-fit-skewed" {
            let risks = numbers("oracle-subset-risks.txt");
            assert_eq!(risks.len(), 7);
            let best = risks
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert_eq!(best, 2); // mask 3: {a,b} versus {c,d}
            if balanced_case {
                assert!(risks[2] < 1e-20);
            }
        }
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn numeric_linear_policy_predictions_and_cv_agree_with_independent_r_qr() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let base = root.join("tests/fixtures/linear-policy/guide-46.3");
    for entry in fs::read_dir(base).unwrap() {
        let path = entry.unwrap().path();
        let read = |name: &str| fs::read_to_string(path.join(name)).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&read("manifest.json")).unwrap();
        let data: Dataset = serde_json::from_str(&read("dataset.json")).unwrap();
        let options: FitOptions = serde_json::from_str(&read("options.json")).unwrap();
        let model = FittedModel::fit(&data, &options).unwrap();
        let t = Temp::new();
        fs::copy(path.join("data.csv"), t.0.join("data.csv")).unwrap();
        let result = &manifest["result"];
        let leaves = result["terminal_node_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let training = result["training_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| if v.as_bool().unwrap() { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(t.0.join("leaves.txt"), leaves).unwrap();
        fs::write(t.0.join("training.txt"), training).unwrap();
        let fitting = data
            .schema
            .iter()
            .filter(|f| f.role.fits())
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let mode = serde_json::to_value(options.prediction_truncation).unwrap();
        let cv = matches!(options.pruning, Pruning::CrossValidation { .. });
        let process = Command::new("Rscript")
            .arg("--vanilla")
            .arg(root.join("tools/linear_policy_oracle.R"))
            .arg(mode.as_str().unwrap())
            .arg(fitting)
            .arg(result["expected_fit"].as_str().unwrap())
            .arg(if cv { "cv" } else { "no_cv" })
            .current_dir(&t.0)
            .output()
            .unwrap();
        assert!(
            process.status.success(),
            "{}: {}",
            path.display(),
            String::from_utf8_lossy(&process.stderr)
        );
        let numbers = |file: &str| -> Vec<f64> {
            fs::read_to_string(t.0.join(file))
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect()
        };
        near(
            &numeric(model.predict(&data.rows).unwrap()),
            &numbers("oracle-predictions.txt"),
            2e-10,
        );
        if cv {
            let scores = &model.report.pruning_scores;
            assert_eq!(scores.len(), 2);
            near(
                &[
                    scores[0].mean_loss,
                    scores[1].mean_loss,
                    scores[0].standard_error,
                    scores[1].standard_error,
                ],
                &numbers("oracle-cv.txt"),
                2e-10,
            );
        }
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn extreme_chi_square_tails_and_inverse_tails_agree_with_r_log_probabilities() {
    let t = Temp::new();
    let dfs = [1, 2, 3, 10, 100, 999, 2000, 100000];
    let mut csv = "x,df\n".to_owned();
    let mut actual = vec![];
    for df in dfs {
        for factor in [1e-6, 0.2, 0.8, 1.0, 1.2, 3.0, 10.0] {
            let x = factor * df as f64;
            csv.push_str(&format!("{x},{df}\n"));
            actual.push(stats::chi_square_log_survival(x, df));
        }
    }
    fs::write(t.0.join("tails.csv"), csv).unwrap();
    let expected = r(
        &t,
        "d<-read.csv('tails.csv');cat(format(pchisq(d$x,d$df,lower.tail=FALSE,log.p=TRUE),digits=17))",
    );
    near(&actual, &expected, 2e-9);
    let mut csv = "log_p,df\n".to_owned();
    let mut actual = vec![];
    for df in dfs {
        for log_p in [-1e-10, -0.01, -0.693, -20.0, -700.0, -1000.0, -1e6] {
            csv.push_str(&format!("{log_p},{df}\n"));
            actual.push(stats::chi_square_log_upper_quantile(log_p, df));
        }
    }
    fs::write(t.0.join("quantiles.csv"), csv).unwrap();
    let expected = r(
        &t,
        "d<-read.csv('quantiles.csv');cat(format(qchisq(d$log_p,d$df,lower.tail=FALSE,log.p=TRUE),digits=17))",
    );
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(&expected) {
        assert!((a - b).abs() <= 3e-8 * b.abs().max(1e-300), "{a} != {b}");
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn weighted_poisson_with_exposure_agrees_with_r_glm() {
    let t = Temp::new();
    let cfg = CsvOptions {
        target: "y".into(),
        weight: Some("w".into()),
        offset: Some("e".into()),
        features: vec![FeatureSchema {
            name: "x".into(),
            feature_type: FeatureType::Numeric,
            role: Role::Both,
        }],
        ..Default::default()
    };
    let o = FitOptions {
        task: Task::Regression {
            loss: Loss::Poisson,
        },
        node_model: NodeModel::Linear,
        max_depth: 0,
        pruning: Pruning::None,
        ..Default::default()
    };
    for exposure_scale in [1.0, 1e40, 1e-40] {
        let mut csv = "x,y,w,e\n".to_owned();
        for i in 0..90 {
            let x = i as f64 / 30.0 - 1.5;
            let exposure = (1 + i % 5) as f64 / 3.0;
            let y = ((0.4 + 0.7 * x).exp() * exposure + ((i * 7) % 5) as f64 / 4.0).floor();
            csv.push_str(&format!(
                "{x},{y},{},{}\n",
                1 + i % 3,
                exposure * exposure_scale
            ));
        }
        fs::write(t.0.join("data.csv"), &csv).unwrap();
        let d = dataset_from_csv(&csv, &cfg, &o.task).unwrap();
        let m = FittedModel::fit(&d, &o).unwrap();
        let expected = r(
            &t,
            "d <- read.csv('data.csv'); f <- glm(y~x+offset(log(e)),weights=w,data=d,family=poisson(),control=glm.control(epsilon=1e-12,maxit=100)); cat(format(predict(f,d,type='response'),digits=17))",
        );
        let actual = numeric(m.predict_with_offsets(&d.rows, Some(&d.offsets)).unwrap());
        near(&actual, &expected, 1e-7);
        assert!(m.report.warnings.is_empty(), "{:?}", m.report.warnings);
    }
}

#[test]
#[ignore = "requires Rscript and rpart; run cargo test --test interoperability -- --ignored"]
fn one_predictor_cost_complexity_pruning_agrees_with_rpart() {
    let t = Temp::new();
    let mut csv = "x,y\n".to_owned();
    for i in 0..40 {
        csv.push_str(&format!("{i},{}\n", [0.0, 2.0, 10.0, 20.0][i / 10]));
    }
    fs::write(t.0.join("data.csv"), &csv).unwrap();
    let cfg = CsvOptions {
        target: "y".into(),
        ..Default::default()
    };
    let d = dataset_from_csv(&csv, &cfg, &Task::default()).unwrap();
    let alphas = [0.0, 0.1, 0.7, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
    // With only one predictor, both algorithms optimize the same split and
    // squared-error objective. This does not compare variable selection rules.
    let expected = r(
        &t,
        "d <- read.csv('data.csv'); f <- rpart::rpart(y~x,data=d,method='anova',control=rpart::rpart.control(cp=0,xval=0,minsplit=4,minbucket=2,maxdepth=3)); a <- c(0,0.1,0.7,2,4,8,16,32,64,128); for (v in a) {g <- rpart::prune(f,cp=v*nrow(d)/f$frame$dev[1]); cat(sum(g$frame$var=='<leaf>'),format(mean((d$y-predict(g,d))^2),digits=17),'\\n')}",
    );
    for (i, alpha) in alphas.into_iter().enumerate() {
        let o = FitOptions {
            max_depth: 3,
            min_gain: 0.0,
            max_split_candidates: 0,
            pruning: Pruning::CostComplexity { alpha },
            ..Default::default()
        };
        let m = FittedModel::fit(&d, &o).unwrap();
        assert_eq!(
            m.trees[0].root.leaves(),
            expected[2 * i] as usize,
            "alpha={alpha}"
        );
        let mse = numeric(m.predict(&d.rows).unwrap())
            .iter()
            .zip(&d.targets)
            .map(|(p, y)| (p - y.number().unwrap()).powi(2))
            .sum::<f64>()
            / d.rows.len() as f64;
        near(&[mse], &[expected[2 * i + 1]], 1e-12);
    }
}

#[test]
#[ignore = "requires Rscript and quantreg; run cargo test --test interoperability -- --ignored"]
fn quantile_nodes_agree_with_r_quantreg() {
    let t = Temp::new();
    let mut csv = "x,z,y,w\n".to_owned();
    for i in 0..101 {
        let x = i as f64 / 17.0 - 3.0;
        let z = ((i * 13) % 29) as f64 / 7.0;
        let y = 2.0 + 0.7 * x - 0.4 * z + 0.2 * x * x + ((i * 31) % 47) as f64 / 23.0;
        csv.push_str(&format!("{x},{z},{y},{}\n", 1 + i % 4));
    }
    fs::write(t.0.join("data.csv"), &csv).unwrap();
    let mut cfg = CsvOptions {
        target: "y".into(),
        weight: Some("w".into()),
        features: ["x", "z"]
            .into_iter()
            .map(|name| FeatureSchema {
                name: name.into(),
                feature_type: FeatureType::Numeric,
                role: Role::Both,
            })
            .collect(),
        ..Default::default()
    };
    for polynomial in [false, true] {
        if polynomial {
            cfg.features[1].role = Role::Excluded;
        }
        for probability in [0.1, 0.5, 0.9] {
            let options = FitOptions {
                task: Task::Regression {
                    loss: Loss::Quantile { probability },
                },
                node_model: if polynomial {
                    NodeModel::Polynomial { degree: 2 }
                } else {
                    NodeModel::Linear
                },
                max_depth: 0,
                pruning: Pruning::None,
                ..Default::default()
            };
            let d = dataset_from_csv(&csv, &cfg, &options.task).unwrap();
            let m = FittedModel::fit(&d, &options).unwrap();
            assert!(m.report.warnings.is_empty(), "{:?}", m.report.warnings);
            let formula = if polynomial {
                "y ~ x + I(x*x)"
            } else {
                "y ~ x + z"
            };
            // Barrodale-Roberts uses a different optimization algorithm.
            let expected = r(
                &t,
                &format!(
                    "d <- read.csv('data.csv'); fit <- quantreg::rq({formula},data=d,weights=w,tau={probability},method='br'); cat(format(predict(fit,d),digits=17))"
                ),
            );
            let actual = numeric(m.predict(&d.rows).unwrap());
            near(&actual, &expected, 2e-6);
            let restored = FittedModel::from_json(&m.to_json().unwrap()).unwrap();
            assert_eq!(
                m.predict(&d.rows).unwrap(),
                restored.predict(&d.rows).unwrap()
            );
        }
    }
}

#[test]
#[ignore = "requires Rscript; run cargo test --test interoperability -- --ignored"]
fn qr_and_irls_agree_with_r_stats() {
    let t = Temp::new();
    let mut csv = "x,y,w,e\n".to_owned();
    for i in 0..60 {
        let x = i as f64 / 10.0 - 3.0;
        let y = 2.0 + 0.7 * x + (i % 7) as f64 / 10.0;
        csv.push_str(&format!("{x},{y},{},{}\n", 1 + i % 3, 1 + i % 2));
    }
    fs::write(t.0.join("data.csv"), &csv).unwrap();
    let cfg = CsvOptions {
        target: "y".into(),
        weight: Some("w".into()),
        features: vec![FeatureSchema {
            name: "x".into(),
            feature_type: FeatureType::Numeric,
            role: Role::Both,
        }],
        ..Default::default()
    };
    let mut options = FitOptions {
        max_depth: 0,
        node_model: NodeModel::Linear,
        pruning: Pruning::None,
        ..Default::default()
    };
    let d = dataset_from_csv(&csv, &cfg, &options.task).unwrap();
    let m = FittedModel::fit(&d, &options).unwrap();
    let expected = r(
        &t,
        "d <- read.csv('data.csv'); fit <- lm(y ~ x, data=d, weights=w); cat(format(predict(fit,d),digits=17))",
    );
    near(&numeric(m.predict(&d.rows).unwrap()), &expected, 1e-9);
    let mut csv = "x,y\n".to_owned();
    for x in -3..=3 {
        for j in 0..20 {
            let successes = 2 * (x + 3) + 3;
            csv.push_str(&format!("{x},{}\n", usize::from(j < successes)));
        }
    }
    fs::write(t.0.join("data.csv"), &csv).unwrap();
    options.task = Task::Regression {
        loss: Loss::Logistic,
    };
    let cfg = CsvOptions {
        target: "y".into(),
        features: cfg.features,
        ..Default::default()
    };
    let d = dataset_from_csv(&csv, &cfg, &options.task).unwrap();
    let m = FittedModel::fit(&d, &options).unwrap();
    let expected = r(
        &t,
        "d <- read.csv('data.csv'); fit <- glm(y ~ x, data=d, family=binomial()); cat(format(predict(fit,d,type='response'),digits=17))",
    );
    near(&numeric(m.predict(&d.rows).unwrap()), &expected, 1e-7);
}
