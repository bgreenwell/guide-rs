use guide_rs::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RunConfig {
    data: CsvOptions,
    fit: FitOptions,
    ensemble: Option<EnsembleOptions>,
    importance: ImportanceOptions,
}

const HELP:&str="guide-rs — experimental native GUIDE-style statistical trees

Commands:
  fit        --data training.csv --config config.json --model model.json [--validation validation.csv]
  predict    --model model.json --data new.csv [--output predictions.csv]
  importance --data training.csv --config config.json [--output importance.json]
  inspect    --model model.json [--output summary.txt]
  export     --model model.json --format text|dot|latex|r [--output file]
  convert    --data training.csv --config config.json --output dataset.json
  config     [--output config.json]   Print a complete default configuration

Existing output files are not overwritten unless --force is supplied.
CSV headers identify columns. Prediction uses the saved schema and missing-value tokens.
See docs/coverage.md: this release does not yet implement all GUIDE 46.3 features.
";
fn error(s: impl Into<String>) -> Error {
    Error(s.into())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("guide-rs: {e}");
        std::process::exit(2);
    }
}
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "--help".into());
    if matches!(command.as_str(), "--help" | "-h" | "help") {
        print!("{HELP}");
        return Ok(());
    }
    if command == "--version" {
        println!(
            "guide-rs {} (experimental; GUIDE 46.3 target)",
            env!("CARGO_PKG_VERSION")
        );
        return Ok(());
    }
    let mut flags = BTreeMap::new();
    let mut force = false;
    while let Some(key) = args.next() {
        if key == "--help" {
            print!("{HELP}");
            return Ok(());
        }
        if key == "--force" {
            force = true;
            continue;
        }
        if !key.starts_with("--") {
            return Err(error(format!("unexpected argument {key}")));
        }
        let value = args
            .next()
            .ok_or_else(|| error(format!("{key} requires a value")))?;
        if value.starts_with("--") {
            return Err(error(format!("{key} requires a value")));
        }
        if flags.insert(key.clone(), value).is_some() {
            return Err(error(format!("duplicate argument {key}")));
        }
    }
    let allowed: &[&str] = match command.as_str() {
        "fit" => &["--data", "--config", "--model", "--validation"],
        "predict" => &["--data", "--model", "--output"],
        "importance" | "convert" => &["--data", "--config", "--output"],
        "inspect" => &["--model", "--output"],
        "export" => &["--model", "--format", "--output"],
        "config" => &["--output"],
        _ => return Err(error(format!("unknown command {command}; use --help"))),
    };
    for key in flags.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(error(format!("unexpected option {key} for {command}")));
        }
    }
    let required = |key: &str| {
        flags
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| error(format!("missing {key}")))
    };
    let output = flags.get("--output").map(String::as_str);
    if let Some(path) = if command == "fit" {
        Some(required("--model")?)
    } else {
        output
    } && Path::new(path).exists()
        && !force
    {
        return Err(error(format!("{path} exists; use --force to overwrite")));
    }
    match command.as_str() {
        "config" => write_output(
            output,
            &serde_json::to_string_pretty(&RunConfig::default())?,
            force,
        )?,
        "fit" | "importance" | "convert" => {
            let config: RunConfig =
                serde_json::from_str(&fs::read_to_string(required("--config")?)?)?;
            let data = dataset_from_csv(
                &fs::read_to_string(required("--data")?)?,
                &config.data,
                &config.fit.task,
            )?;
            if command == "convert" {
                write_output(
                    Some(required("--output")?),
                    &serde_json::to_string_pretty(&data)?,
                    force,
                )?;
            } else if command == "importance" {
                let result = importance(&data, &config.fit, &config.importance)?;
                write_output(output, &serde_json::to_string_pretty(&result)?, force)?;
            } else {
                let validation = flags
                    .get("--validation")
                    .map(|p| {
                        let mut options = config.data.clone();
                        options.features = data.schema.clone();
                        dataset_from_csv(&fs::read_to_string(p)?, &options, &config.fit.task)
                    })
                    .transpose()?;
                if validation.is_some() && config.fit.pruning != Pruning::Validation {
                    return Err(error("--validation requires validation pruning"));
                }
                if validation.is_some() && config.ensemble.is_some() {
                    return Err(error("ensemble validation pruning is not implemented"));
                }
                let mut model = if let Some(e) = &config.ensemble {
                    FittedModel::fit_ensemble(&data, &config.fit, e)?
                } else {
                    FittedModel::fit_with_validation(&data, &config.fit, validation.as_ref())?
                };
                model.input = Some(config.data);
                write_output(Some(required("--model")?), &model.to_json()?, force)?;
                eprintln!(
                    "Fitted {} tree(s), {} leaves, {} training rows; excluded {}.",
                    model.trees.len(),
                    model.report.fitted_leaves,
                    model.report.training_rows,
                    model.report.excluded_rows
                );
                for w in &model.report.warnings {
                    eprintln!("Warning: {w}");
                }
            }
        }
        "predict" => {
            let model = FittedModel::from_json(&fs::read_to_string(required("--model")?)?)?;
            let options = model.input.clone().unwrap_or_default();
            let input = fs::read_to_string(required("--data")?)?;
            let rows = prediction_rows_from_csv(
                &input,
                &model.schema,
                &options.missing_values,
                options.delimiter,
            )?;
            let offsets = if model.requires_exposure {
                let name = options.offset.as_ref().ok_or_else(|| {
                    error("saved model has no CSV exposure column; use the Rust API")
                })?;
                let csv = parse_csv(&input, options.delimiter)?;
                let j = csv[0]
                    .iter()
                    .position(|h| h == name)
                    .ok_or_else(|| error(format!("missing exposure column {name}")))?;
                Some(
                    csv[1..]
                        .iter()
                        .map(|r| {
                            r[j].parse::<f64>()
                                .map_err(|_| error("invalid prediction exposure"))
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            };
            let predictions = model.predict_with_offsets(&rows, offsets.as_deref())?;
            let mut out = String::from(
                if matches!(
                    model.options.task,
                    Task::Regression {
                        loss: Loss::DualQuantile { .. }
                    }
                ) {
                    "lower,upper"
                } else {
                    "prediction"
                },
            );
            for c in &model.classes {
                out.push(',');
                out.push_str(&csv_field(&format!("probability:{c}")));
            }
            out.push('\n');
            for p in predictions {
                match p {
                    Prediction::Quantiles { lower, upper } => {
                        out.push_str(&format!("{lower},{upper}"))
                    }
                    Prediction::Regression { value } => out.push_str(&value.to_string()),
                    Prediction::Classification {
                        label,
                        probabilities,
                    } => {
                        out.push_str(&csv_field(&label));
                        for (_, p) in probabilities {
                            out.push(',');
                            out.push_str(&p.to_string());
                        }
                    }
                }
                out.push('\n');
            }
            write_output(output, &out, force)?;
        }
        "inspect" | "export" => {
            let model = FittedModel::from_json(&fs::read_to_string(required("--model")?)?)?;
            let format = if command == "inspect" {
                "text"
            } else {
                required("--format")?
            };
            let content = match format {
                "text" => model.to_text()?,
                "dot" => model.to_dot()?,
                "latex" => model.to_latex()?,
                "r" => model.to_r()?,
                _ => return Err(error("format must be text, dot, latex, or r")),
            };
            write_output(output, &content, force)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
fn write_output(path: Option<&str>, content: &str, force: bool) -> Result<()> {
    if let Some(path) = path {
        use std::io::Write;
        let mut options = fs::OpenOptions::new();
        options.write(true);
        if force {
            options.create(true).truncate(true);
        } else {
            options.create_new(true);
        }
        let mut f = options.open(path)?;
        f.write_all(content.as_bytes())?;
        if !content.ends_with('\n') {
            f.write_all(b"\n")?;
        }
    } else {
        print!("{content}");
        if !content.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}
