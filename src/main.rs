use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use guide_rs::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RunConfig {
    data: CsvOptions,
    fit: FitOptions,
    ensemble: Option<EnsembleOptions>,
    importance: ImportanceOptions,
}

/// Experimental native GUIDE-style statistical trees.
///
/// CSV headers identify columns. Prediction uses the saved schema and
/// missing-value tokens. See docs/coverage.md: this release does not yet
/// implement all GUIDE 46.3 features.
#[derive(Parser)]
#[command(
    name = "guide-rs",
    version,
    long_version = concat!(env!("CARGO_PKG_VERSION"), " (experimental; GUIDE 46.3 target)")
)]
struct Cli {
    /// Overwrite existing output files
    #[arg(long, global = true)]
    force: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fit a tree or ensemble and save it as JSON
    Fit {
        /// CSV file with a header row
        #[arg(long)]
        data: PathBuf,
        /// JSON run configuration (see `guide-rs config`)
        #[arg(long)]
        config: PathBuf,
        /// Saved model JSON
        #[arg(long)]
        model: PathBuf,
        /// Separate pruning sample (requires validation pruning)
        #[arg(long)]
        validation: Option<PathBuf>,
    },
    /// Predict new rows with a saved model
    Predict {
        /// Saved model JSON
        #[arg(long)]
        model: PathBuf,
        /// CSV file with a header row
        #[arg(long)]
        data: PathBuf,
        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compute variable importance scores
    Importance {
        /// CSV file with a header row
        #[arg(long)]
        data: PathBuf,
        /// JSON run configuration (see `guide-rs config`)
        #[arg(long)]
        config: PathBuf,
        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Summarize a saved model as text
    Inspect {
        /// Saved model JSON
        #[arg(long)]
        model: PathBuf,
        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Export the fitted trees in another format
    Export {
        /// Saved model JSON
        #[arg(long)]
        model: PathBuf,
        /// Output format
        #[arg(long, value_enum)]
        format: Format,
        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Convert a CSV file to the JSON dataset format
    Convert {
        /// CSV file with a header row
        #[arg(long)]
        data: PathBuf,
        /// JSON run configuration (see `guide-rs config`)
        #[arg(long)]
        config: PathBuf,
        /// Output file
        #[arg(long)]
        output: PathBuf,
    },
    /// Print a complete default configuration
    Config {
        /// Output file (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Text,
    Dot,
    Latex,
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("guide-rs: {e:#}");
        std::process::exit(2);
    }
}

fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))
}
fn read_config(path: &Path) -> Result<RunConfig> {
    serde_json::from_str(&read(path)?)
        .with_context(|| format!("invalid configuration {}", path.display()))
}
fn read_model(path: &Path) -> Result<FittedModel> {
    FittedModel::from_json(&read(path)?)
        .with_context(|| format!("invalid model {}", path.display()))
}
fn read_dataset(path: &Path, options: &CsvOptions, task: &Task) -> Result<Dataset> {
    dataset_from_csv(&read(path)?, options, task).with_context(|| path.display().to_string())
}

fn run(cli: Cli) -> Result<()> {
    let force = cli.force;
    let target = match &cli.command {
        Command::Fit { model, .. } => Some(model),
        Command::Convert { output, .. } => Some(output),
        Command::Predict { output, .. }
        | Command::Importance { output, .. }
        | Command::Inspect { output, .. }
        | Command::Export { output, .. }
        | Command::Config { output } => output.as_ref(),
    };
    if let Some(path) = target
        && path.exists()
        && !force
    {
        bail!("{} exists; use --force to overwrite", path.display());
    }
    match cli.command {
        Command::Config { output } => write_output(
            output.as_deref(),
            &serde_json::to_string_pretty(&RunConfig::default())?,
            force,
        ),
        Command::Convert {
            data,
            config,
            output,
        } => {
            let config = read_config(&config)?;
            let data = read_dataset(&data, &config.data, &config.fit.task)?;
            write_output(Some(&output), &serde_json::to_string_pretty(&data)?, force)
        }
        Command::Importance {
            data,
            config,
            output,
        } => {
            let config = read_config(&config)?;
            let data = read_dataset(&data, &config.data, &config.fit.task)?;
            let result = importance(&data, &config.fit, &config.importance)?;
            write_output(
                output.as_deref(),
                &serde_json::to_string_pretty(&result)?,
                force,
            )
        }
        Command::Fit {
            data,
            config,
            model,
            validation,
        } => {
            let config = read_config(&config)?;
            let training = read_dataset(&data, &config.data, &config.fit.task)?;
            let validation = validation
                .map(|p| {
                    let mut options = config.data.clone();
                    options.features = training.schema.clone();
                    read_dataset(&p, &options, &config.fit.task)
                })
                .transpose()?;
            if validation.is_some() && config.fit.pruning != Pruning::Validation {
                bail!("--validation requires validation pruning");
            }
            if validation.is_some() && config.ensemble.is_some() {
                bail!("ensemble validation pruning is not implemented");
            }
            let mut fitted = if let Some(e) = &config.ensemble {
                FittedModel::fit_ensemble(&training, &config.fit, e)?
            } else {
                FittedModel::fit_with_validation(&training, &config.fit, validation.as_ref())?
            };
            fitted.input = Some(config.data);
            write_output(Some(&model), &fitted.to_json()?, force)?;
            eprintln!(
                "Fitted {} tree(s), {} leaves, {} training rows; excluded {}.",
                fitted.trees.len(),
                fitted.report.fitted_leaves,
                fitted.report.training_rows,
                fitted.report.excluded_rows
            );
            for w in &fitted.report.warnings {
                eprintln!("Warning: {w}");
            }
            Ok(())
        }
        Command::Predict {
            model,
            data,
            output,
        } => {
            let model = read_model(&model)?;
            let options = model.input.clone().unwrap_or_default();
            let input = read(&data)?;
            let rows = prediction_rows_from_csv(
                &input,
                &model.schema,
                &options.missing_values,
                options.delimiter,
            )
            .with_context(|| data.display().to_string())?;
            let offsets = if model.requires_exposure {
                let name = options
                    .offset
                    .as_ref()
                    .context("saved model has no CSV exposure column; use the Rust API")?;
                Some(
                    exposures_from_csv(&input, name, options.delimiter)
                        .with_context(|| data.display().to_string())?,
                )
            } else {
                None
            };
            let predictions = model.predict_with_offsets(&rows, offsets.as_deref())?;
            write_output(
                output.as_deref(),
                &predictions_csv(&model, predictions),
                force,
            )
        }
        Command::Inspect { model, output } => {
            write_output(output.as_deref(), &read_model(&model)?.to_text()?, force)
        }
        Command::Export {
            model,
            format,
            output,
        } => {
            let model = read_model(&model)?;
            let content = match format {
                Format::Text => model.to_text()?,
                Format::Dot => model.to_dot()?,
                Format::Latex => model.to_latex()?,
            };
            write_output(output.as_deref(), &content, force)
        }
    }
}

fn predictions_csv(model: &FittedModel, predictions: Vec<Prediction>) -> String {
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
            Prediction::Quantiles { lower, upper } => out.push_str(&format!("{lower},{upper}")),
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
    out
}

fn write_output(path: Option<&Path>, content: &str, force: bool) -> Result<()> {
    let Some(path) = path else {
        print!("{content}");
        if !content.ends_with('\n') {
            println!();
        }
        return Ok(());
    };
    let mut options = fs::OpenOptions::new();
    options.write(true);
    if force {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    let mut f = options
        .open(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    f.write_all(content.as_bytes())
        .and_then(|()| {
            if content.ends_with('\n') {
                Ok(())
            } else {
                f.write_all(b"\n")
            }
        })
        .with_context(|| format!("cannot write {}", path.display()))
}
