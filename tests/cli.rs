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
            "guide-rs-test-{}-{}",
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
fn run(temp: &Temp, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_guide-rs"))
        .current_dir(&temp.0)
        .args(args)
        .output()
        .unwrap()
}
fn success(out: std::process::Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn cli_fit_predict_export_and_refuse_overwrite() {
    let t = Temp::new();
    fs::write(t.0.join("train.csv"), "x,y\n1,2\n2,2\n3,8\n4,8\n").unwrap();
    fs::write(
        t.0.join("config.json"),
        // Permit this four-row smoke case under the native numerical rank
        // bounds (two rows on the right requires an explicit minimum of one).
        r#"{"data":{"target":"y"},"fit":{"min_samples_leaf":1,"pruning":{"kind":"none"}}}"#,
    )
    .unwrap();
    success(run(
        &t,
        &[
            "fit",
            "--data",
            "train.csv",
            "--config",
            "config.json",
            "--model",
            "model.json",
        ],
    ));
    let p = success(run(
        &t,
        &["predict", "--data", "train.csv", "--model", "model.json"],
    ));
    assert_eq!(p, "prediction\n2\n2\n8\n8\n");
    assert!(success(run(&t, &["inspect", "--model", "model.json"])).contains("2 fitted leaves"));
    for format in ["text", "dot", "latex"] {
        assert!(
            !success(run(
                &t,
                &["export", "--model", "model.json", "--format", format]
            ))
            .is_empty()
        );
    }
    let removed = run(&t, &["export", "--model", "model.json", "--format", "r"]);
    assert!(!removed.status.success());
    let fail = run(
        &t,
        &[
            "fit",
            "--data",
            "train.csv",
            "--config",
            "config.json",
            "--model",
            "model.json",
        ],
    );
    assert!(!fail.status.success());
    assert!(String::from_utf8_lossy(&fail.stderr).contains("exists"));
    assert!(
        !run(
            &t,
            &[
                "predict",
                "--model",
                "model.json",
                "--data",
                "train.csv",
                "--typo",
                "1"
            ]
        )
        .status
        .success()
    );
}

#[test]
fn cli_errors_name_the_offending_file() {
    let t = Temp::new();
    fs::write(t.0.join("config.json"), r#"{"data":{"target":"y"}}"#).unwrap();
    let out = run(
        &t,
        &[
            "fit",
            "--data",
            "missing.csv",
            "--config",
            "config.json",
            "--model",
            "model.json",
        ],
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read missing.csv"));
    fs::write(t.0.join("bad.json"), "{").unwrap();
    let out = run(&t, &["inspect", "--model", "bad.json"]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("invalid model bad.json"));
}

#[test]
fn cli_roundtrip_preserves_prediction_truncation() {
    let t = Temp::new();
    let mut csv = "x,y\n".to_owned();
    for i in 0..20 {
        csv.push_str(&format!("{i},{}\n", 3 + 2 * i));
    }
    fs::write(t.0.join("train.csv"), csv).unwrap();
    fs::write(t.0.join("new.csv"), "x\n-10\n30\n").unwrap();
    fs::write(t.0.join("config.json"),r#"{"data":{"target":"y","features":[{"name":"x","feature_type":{"kind":"numeric"},"role":"fit"}]},"fit":{"node_model":{"kind":"linear"},"prediction_truncation":"global_range","pruning":{"kind":"none"}}}"#).unwrap();
    success(run(
        &t,
        &[
            "fit",
            "--data",
            "train.csv",
            "--config",
            "config.json",
            "--model",
            "model.json",
        ],
    ));
    let predictions = success(run(
        &t,
        &["predict", "--data", "new.csv", "--model", "model.json"],
    ));
    assert_eq!(predictions, "prediction\n3\n41\n");
    let model = fs::read_to_string(t.0.join("model.json")).unwrap();
    assert!(model.contains("prediction_bounds"));
    assert!(
        success(run(
            &t,
            &["export", "--model", "model.json", "--format", "text"]
        ))
        .contains("prediction limits")
    );
}
#[test]
fn prediction_reorders_headers_and_preserves_missing_tokens() {
    let t = Temp::new();
    fs::write(t.0.join("train.csv"), "x,z,y\n1,0,2\n1,0,2\n?,0,8\n?,0,8\n").unwrap();
    fs::write(
        t.0.join("config.json"),
        r#"{"data":{"target":"y","missing_values":["?"]},"fit":{"pruning":{"kind":"none"}}}"#,
    )
    .unwrap();
    success(run(
        &t,
        &[
            "fit",
            "--data",
            "train.csv",
            "--config",
            "config.json",
            "--model",
            "model.json",
        ],
    ));
    fs::write(t.0.join("test.csv"), "z,x\n0,?\n0,1\n").unwrap();
    assert_eq!(
        success(run(
            &t,
            &["predict", "--data", "test.csv", "--model", "model.json"]
        )),
        "prediction\n8\n2\n"
    );
}
#[test]
fn cli_routes_missing_values_by_flag_code() {
    let t = Temp::new();
    let mut csv = "x,x_reason,y\n".to_owned();
    for i in 0..60 {
        match i % 3 {
            0 => csv.push_str("NA,refused,0\n"),
            1 => csv.push_str("NA,skipped,10\n"),
            _ => csv.push_str(&format!("{i},,10\n")),
        }
    }
    fs::write(t.0.join("train.csv"), csv).unwrap();
    fs::write(
        t.0.join("config.json"),
        r#"{"data":{"target":"y","features":[
            {"name":"x","feature_type":{"kind":"numeric"}},
            {"name":"x_reason","feature_type":{"kind":"categorical"},"role":"flag","flags":"x"}]},
          "fit":{"max_depth":1,"pruning":{"kind":"none"}}}"#,
    )
    .unwrap();
    success(run(
        &t,
        &[
            "fit",
            "--data",
            "train.csv",
            "--config",
            "config.json",
            "--model",
            "model.json",
        ],
    ));
    // Refused goes left; skipped, unseen and uncoded missing values go right.
    fs::write(
        t.0.join("new.csv"),
        "x_reason,x\nrefused,NA\nskipped,NA\nunknown,NA\n,NA\n,5\n",
    )
    .unwrap();
    assert_eq!(
        success(run(
            &t,
            &["predict", "--data", "new.csv", "--model", "model.json"]
        )),
        "prediction\n0\n10\n10\n10\n10\n"
    );
    assert!(
        success(run(&t, &["inspect", "--model", "model.json"]))
            .contains("x_reason in [\"refused\"]")
    );
}

#[test]
fn bundled_iris_has_useful_classification_accuracy() {
    let input = include_str!("fixtures/datasets/iris.csv");
    let cfg = CsvOptions {
        target: "Species".into(),
        ..Default::default()
    };
    let task = Task::Classification;
    let d = dataset_from_csv(input, &cfg, &task).unwrap();
    let options = FitOptions {
        task,
        seed: 123,
        ..Default::default()
    };
    let m = FittedModel::fit(&d, &options).unwrap();
    let correct=m.predict(&d.rows).unwrap().iter().zip(&d.targets).filter(|(p,y)|matches!(p,Prediction::Classification{label,..} if Some(label.clone())==y.label())).count();
    assert!(correct >= 140, "correct={correct}");
    assert!(m.trees[0].root.leaves() > 1);
}
#[test]
fn cli_thread_count_does_not_change_results() {
    let t = Temp::new();
    let rows: String = (0..60)
        .map(|i| {
            format!(
                "{},{},{}\n",
                i * 7 % 60,
                i % 4,
                f64::from(i * 7 % 60 > 25) * 4.0
            )
        })
        .collect();
    fs::write(t.0.join("train.csv"), format!("x,z,y\n{rows}")).unwrap();
    fs::write(
        t.0.join("config.json"),
        r#"{"data":{"target":"y"},"fit":{"pruning":{"kind":"none"}},
            "ensemble":{"kind":"bagging","trees":12},
            "importance":{"permutations":20}}"#,
    )
    .unwrap();
    for threads in ["1", "3"] {
        for (command, output) in [("fit", "--model"), ("importance", "--output")] {
            success(run(
                &t,
                &[
                    command,
                    "--threads",
                    threads,
                    "--data",
                    "train.csv",
                    "--config",
                    "config.json",
                    output,
                    &format!("{command}-{threads}.json"),
                ],
            ));
        }
    }
    for command in ["fit", "importance"] {
        assert_eq!(
            fs::read(t.0.join(format!("{command}-1.json"))).unwrap(),
            fs::read(t.0.join(format!("{command}-3.json"))).unwrap()
        );
    }
    let out = run(&t, &["--threads", "0", "config"]);
    assert!(!out.status.success());
}
