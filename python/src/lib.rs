//! Python bindings for guide-rs. The `guide_rs` package wraps these in
//! scikit-learn estimators; options and schemas cross the boundary as JSON so
//! the core crate's validation applies unchanged.
use guide_rs::{
    Dataset, EnsembleOptions, FeatureSchema, FeatureType, FitOptions, FittedModel,
    ImportanceOptions, Prediction, Task, Value,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn number(x: f64) -> Value {
    if x.is_nan() {
        Value::Missing
    } else {
        Value::Number(x)
    }
}

fn label(x: Option<String>) -> Value {
    x.map_or(Value::Missing, Value::Text)
}

/// Converts column-major input to rows. Numeric and periodic columns arrive
/// as floats with NaN for missing; categorical columns as str or None.
fn rows(schema: &[FeatureSchema], columns: &[Bound<'_, PyAny>]) -> PyResult<Vec<Vec<Value>>> {
    if columns.len() != schema.len() {
        return Err(value_error(format!(
            "expected {} columns, got {}",
            schema.len(),
            columns.len()
        )));
    }
    let mut values = Vec::with_capacity(columns.len());
    for (f, column) in schema.iter().zip(columns) {
        let converted: Vec<Value> = match f.feature_type {
            FeatureType::Categorical => column
                .extract::<Vec<Option<String>>>()?
                .into_iter()
                .map(label)
                .collect(),
            _ => column
                .extract::<Vec<f64>>()?
                .into_iter()
                .map(number)
                .collect(),
        };
        values.push(converted);
    }
    let n = values.first().map_or(0, Vec::len);
    if values.iter().any(|c| c.len() != n) {
        return Err(value_error("columns must have equal length"));
    }
    let mut columns: Vec<_> = values.into_iter().map(Vec::into_iter).collect();
    Ok((0..n)
        .map(|_| {
            columns
                .iter_mut()
                // Every column was checked to have n values.
                .map(|c| c.next().expect("column length"))
                .collect()
        })
        .collect())
}

fn targets(task: &Task, target: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    Ok(match task {
        Task::Classification => target
            .extract::<Vec<Option<String>>>()?
            .into_iter()
            .map(label)
            .collect(),
        Task::Regression { .. } => target
            .extract::<Vec<f64>>()?
            .into_iter()
            .map(number)
            .collect(),
    })
}

fn dataset(
    schema: &str,
    columns: &[Bound<'_, PyAny>],
    target: &Bound<'_, PyAny>,
    options: &FitOptions,
    weights: Option<Vec<f64>>,
    exposure: Option<Vec<f64>>,
) -> PyResult<Dataset> {
    let schema: Vec<FeatureSchema> = serde_json::from_str(schema).map_err(value_error)?;
    Ok(Dataset {
        rows: rows(&schema, columns)?,
        targets: targets(&options.task, target)?,
        schema,
        weights: weights.unwrap_or_default(),
        offsets: exposure.unwrap_or_default(),
    })
}

/// Runs `f` without the GIL, on `threads` threads or the global pool (all
/// cores) when None. Results do not depend on the thread count.
fn parallel<T: Send>(
    py: Python<'_>,
    threads: Option<usize>,
    f: impl FnOnce() -> T + Send,
) -> PyResult<T> {
    match threads {
        None => Ok(py.detach(f)),
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(value_error)?;
            Ok(py.detach(|| pool.install(f)))
        }
    }
}

/// Importance scores as JSON, in the format of `guide-rs importance`.
#[pyfunction]
#[pyo3(signature = (schema, columns, target, options, importance, weights=None, threads=None))]
#[allow(clippy::too_many_arguments)]
fn importance(
    py: Python<'_>,
    schema: &str,
    columns: Vec<Bound<'_, PyAny>>,
    target: Bound<'_, PyAny>,
    options: &str,
    importance: &str,
    weights: Option<Vec<f64>>,
    threads: Option<usize>,
) -> PyResult<String> {
    let options: FitOptions = serde_json::from_str(options).map_err(value_error)?;
    let importance: ImportanceOptions = serde_json::from_str(importance).map_err(value_error)?;
    let data = dataset(schema, &columns, &target, &options, weights, None)?;
    let result = parallel(py, threads, || {
        guide_rs::importance(&data, &options, &importance)
    })?
    .map_err(value_error)?;
    serde_json::to_string(&result).map_err(value_error)
}

/// Validation columns, target, weights and exposure.
type ValidationInput<'py> = (
    Vec<Bound<'py, PyAny>>,
    Bound<'py, PyAny>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
);

/// A fitted tree or ensemble.
#[pyclass(module = "guide_rs._core", frozen)]
struct Model {
    inner: FittedModel,
}

#[pymethods]
impl Model {
    /// Fits a model. `schema`, `options` and `ensemble` are JSON in the core
    /// crate's configuration format.
    #[staticmethod]
    /// `validation` is `(columns, target, weights, exposure)` for validation
    /// pruning, with the same schema as the training data.
    #[pyo3(signature = (schema, columns, target, options, weights=None, exposure=None, ensemble=None, threads=None, validation=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        schema: &str,
        columns: Vec<Bound<'_, PyAny>>,
        target: Bound<'_, PyAny>,
        options: &str,
        weights: Option<Vec<f64>>,
        exposure: Option<Vec<f64>>,
        ensemble: Option<&str>,
        threads: Option<usize>,
        validation: Option<ValidationInput<'_>>,
    ) -> PyResult<Self> {
        let options: FitOptions = serde_json::from_str(options).map_err(value_error)?;
        let ensemble: Option<EnsembleOptions> = ensemble
            .map(serde_json::from_str)
            .transpose()
            .map_err(value_error)?;
        let data = dataset(schema, &columns, &target, &options, weights, exposure)?;
        let validation = validation
            .map(|(columns, target, weights, exposure)| {
                dataset(schema, &columns, &target, &options, weights, exposure)
            })
            .transpose()?;
        if validation.is_some() && ensemble.is_some() {
            return Err(value_error(
                "ensemble validation pruning is not implemented",
            ));
        }
        let inner = parallel(py, threads, || match &ensemble {
            Some(e) => FittedModel::fit_ensemble(&data, &options, e),
            None => FittedModel::fit_with_validation(&data, &options, validation.as_ref()),
        })?
        .map_err(value_error)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_json(input: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FittedModel::from_json(input).map_err(value_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    fn to_text(&self) -> PyResult<String> {
        self.inner.to_text().map_err(value_error)
    }

    fn to_dot(&self) -> PyResult<String> {
        self.inner.to_dot().map_err(value_error)
    }

    fn to_latex(&self) -> PyResult<String> {
        self.inner.to_latex().map_err(value_error)
    }

    /// Class labels in the model's (lexicographic) order; empty for regression.
    #[getter]
    fn classes(&self) -> Vec<String> {
        self.inner.classes.clone()
    }

    /// The fit report as JSON.
    #[getter]
    fn report(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.report).map_err(value_error)
    }

    /// Leaf count and depth of each tree.
    #[getter]
    fn shapes(&self) -> Vec<(usize, usize)> {
        self.inner
            .trees
            .iter()
            .map(|t| (t.root.leaves(), t.root.depth()))
            .collect()
    }

    /// Regression predictions.
    #[pyo3(signature = (columns, exposure=None))]
    fn predict_values(
        &self,
        py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
        exposure: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        let rows = rows(&self.inner.schema, &columns)?;
        let predictions = py
            .detach(|| self.inner.predict_with_offsets(&rows, exposure.as_deref()))
            .map_err(value_error)?;
        predictions
            .into_iter()
            .map(|p| match p {
                Prediction::Regression { value } => Ok(value),
                _ => Err(value_error("model does not predict a single value")),
            })
            .collect()
    }

    /// Lower and upper quantile predictions of a dual-quantile model.
    fn predict_intervals(
        &self,
        py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(f64, f64)>> {
        let rows = rows(&self.inner.schema, &columns)?;
        let predictions = py
            .detach(|| self.inner.predict(&rows))
            .map_err(value_error)?;
        predictions
            .into_iter()
            .map(|p| match p {
                Prediction::Quantiles { lower, upper } => Ok((lower, upper)),
                _ => Err(value_error("model does not predict quantile intervals")),
            })
            .collect()
    }

    /// Predicted labels and class probabilities in the order of `classes`.
    fn predict_classes(
        &self,
        py: Python<'_>,
        columns: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<String>, Vec<Vec<f64>>)> {
        let rows = rows(&self.inner.schema, &columns)?;
        let predictions = py
            .detach(|| self.inner.predict(&rows))
            .map_err(value_error)?;
        predictions
            .into_iter()
            .map(|p| match p {
                Prediction::Classification {
                    label,
                    probabilities,
                } => Ok((label, probabilities.into_iter().map(|(_, p)| p).collect())),
                _ => Err(value_error("model is not a classifier")),
            })
            .collect()
    }

    /// Terminal node id of each row, one list per tree.
    fn apply(&self, columns: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Vec<usize>>> {
        let rows = rows(&self.inner.schema, &columns)?;
        Ok(self
            .inner
            .trees
            .iter()
            .map(|t| rows.iter().map(|r| t.root.leaf(r).id).collect())
            .collect())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(importance, m)?)?;
    m.add("ALGORITHM_VERSION", guide_rs::ALGORITHM_VERSION)?;
    Ok(())
}
