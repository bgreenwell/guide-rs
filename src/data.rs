use crate::{Error, Loss, Result, Task, invalid};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Number(f64),
    Text(String),
    Missing,
}
impl Value {
    pub fn number(&self) -> Option<f64> {
        if let Self::Number(x) = self {
            Some(*x)
        } else {
            None
        }
    }
    pub fn label(&self) -> Option<String> {
        match self {
            Self::Number(x) => Some(x.to_string()),
            Self::Text(s) => Some(s.clone()),
            Self::Missing => None,
        }
    }
    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureType {
    Numeric,
    Categorical,
    Periodic { period: f64 },
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[default]
    Split,
    Fit,
    Both,
    Excluded,
}
impl Role {
    pub fn splits(self) -> bool {
        matches!(self, Self::Split | Self::Both)
    }
    pub fn fits(self) -> bool {
        matches!(self, Self::Fit | Self::Both)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeatureSchema {
    pub name: String,
    pub feature_type: FeatureType,
    #[serde(default)]
    pub role: Role,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub schema: Vec<FeatureSchema>,
    pub rows: Vec<Vec<Value>>,
    pub targets: Vec<Value>,
    #[serde(default)]
    pub weights: Vec<f64>,
    /// Exposure (not its logarithm) for Poisson regression.
    #[serde(default)]
    pub offsets: Vec<f64>,
}
impl Dataset {
    pub fn weight(&self, i: usize) -> f64 {
        self.weights.get(i).copied().unwrap_or(1.0)
    }
    pub fn offset(&self, i: usize) -> f64 {
        self.offsets.get(i).copied().unwrap_or(1.0)
    }
    pub fn validate(&self, task: &Task) -> Result<()> {
        validate_schema(&self.schema)?;
        if self.rows.is_empty() {
            return Err(invalid("dataset is empty"));
        }
        if self.targets.len() != self.rows.len()
            || (!self.weights.is_empty() && self.weights.len() != self.rows.len())
            || (!self.offsets.is_empty() && self.offsets.len() != self.rows.len())
        {
            return Err(invalid(
                "targets, weights and offsets must have one entry per row",
            ));
        }
        if !self.offsets.is_empty()
            && !matches!(
                task,
                Task::Regression {
                    loss: Loss::Poisson
                }
            )
        {
            return Err(invalid("offsets require Poisson regression"));
        }
        for (i, row) in self.rows.iter().enumerate() {
            validate_row(&self.schema, row).map_err(|e| invalid(format!("row {}: {e}", i + 1)))?;
            let w = self.weight(i);
            if !w.is_finite() || w < 0.0 {
                return Err(invalid(format!("row {}: invalid weight", i + 1)));
            }
            if !self.offset(i).is_finite() || self.offset(i) <= 0.0 {
                return Err(invalid(format!(
                    "row {}: exposure must be positive and finite",
                    i + 1
                )));
            }
            if let Value::Number(y) = self.targets[i]
                && !y.is_finite()
            {
                return Err(invalid("nonfinite response"));
            }
            if let Task::Regression { loss } = task {
                if self.targets[i].is_missing() || w == 0.0 {
                    continue;
                }
                let y = self.targets[i]
                    .number()
                    .ok_or_else(|| invalid("regression requires numeric responses"))?;
                if matches!(loss, Loss::Poisson) && y < 0.0 {
                    return Err(invalid("Poisson response must be nonnegative"));
                }
                if matches!(loss, Loss::Logistic) && y != 0.0 && y != 1.0 {
                    return Err(invalid("logistic response must be 0 or 1"));
                }
            }
        }
        if self.active_rows().is_empty() {
            return Err(invalid(
                "no rows with observed response and positive weight",
            ));
        }
        Ok(())
    }
    pub fn active_rows(&self) -> Vec<usize> {
        (0..self.rows.len())
            .filter(|&i| self.weight(i) > 0.0 && !self.targets[i].is_missing())
            .collect()
    }
    pub fn subset(&self, ids: &[usize]) -> Self {
        Self {
            schema: self.schema.clone(),
            rows: ids.iter().map(|&i| self.rows[i].clone()).collect(),
            targets: ids.iter().map(|&i| self.targets[i].clone()).collect(),
            weights: ids.iter().map(|&i| self.weight(i)).collect(),
            offsets: if self.offsets.is_empty() {
                vec![]
            } else {
                ids.iter().map(|&i| self.offset(i)).collect()
            },
        }
    }
}
pub(crate) fn validate_schema(schema: &[FeatureSchema]) -> Result<()> {
    let mut names = BTreeSet::new();
    for f in schema {
        if f.name.is_empty() || !names.insert(&f.name) {
            return Err(invalid("feature names must be nonempty and unique"));
        }
        if let FeatureType::Periodic { period } = f.feature_type
            && (!period.is_finite() || period <= 0.0)
        {
            return Err(invalid("period must be positive and finite"));
        }
    }
    Ok(())
}
pub(crate) fn validate_row(schema: &[FeatureSchema], row: &[Value]) -> Result<()> {
    if row.len() != schema.len() {
        return Err(invalid("row width does not match model schema"));
    }
    for (f, x) in schema.iter().zip(row) {
        if f.role == Role::Excluded || x.is_missing() {
            continue;
        }
        match (&f.feature_type, x) {
            (_, Value::Number(n)) if !n.is_finite() => {
                return Err(invalid(format!("{} is not finite", f.name)));
            }
            (FeatureType::Numeric | FeatureType::Periodic { .. }, Value::Text(_)) => {
                return Err(invalid(format!("{} must be numeric", f.name)));
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CsvOptions {
    pub target: String,
    pub weight: Option<String>,
    pub offset: Option<String>,
    /// If empty, infer numeric versus categorical from all nonmissing values.
    pub features: Vec<FeatureSchema>,
    pub missing_values: Vec<String>,
    pub delimiter: char,
}
impl Default for CsvOptions {
    fn default() -> Self {
        Self {
            target: "target".into(),
            weight: None,
            offset: None,
            features: vec![],
            missing_values: vec!["".into(), "NA".into()],
            delimiter: ',',
        }
    }
}

/// Strict CSV parser supporting quoted delimiters, escaped quotes, and multiline fields.
///
/// Blank lines are skipped, as in R's `read.csv`.
pub fn parse_csv(input: &str, delimiter: char) -> Result<Vec<Vec<String>>> {
    Ok(parse_csv_lines(input, delimiter)?
        .into_iter()
        .map(|(_, r)| r)
        .collect())
}
/// Parses CSV records paired with the 1-based line on which each record starts.
fn parse_csv_lines(input: &str, delimiter: char) -> Result<Vec<(usize, Vec<String>)>> {
    if matches!(delimiter, '"' | '\r' | '\n') {
        return Err(invalid("invalid CSV delimiter"));
    }
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = input.chars().peekable();
    let mut quoted = false;
    let mut after_quote = false;
    let mut pending = false;
    let mut line = 1;
    let mut start = 1;
    while let Some(c) = chars.next() {
        if !pending {
            start = line;
        }
        pending = true;
        if c == '\n' || (c == '\r' && chars.peek() != Some(&'\n')) {
            line += 1;
        }
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                    after_quote = true;
                }
            } else {
                field.push(c);
            }
        } else if c == delimiter {
            row.push(std::mem::take(&mut field));
            after_quote = false;
        } else if c == '\n' || c == '\r' {
            if c == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
                line += 1;
            }
            if row.is_empty() && field.is_empty() && !after_quote {
                pending = false;
                continue;
            }
            row.push(std::mem::take(&mut field));
            records.push((start, std::mem::take(&mut row)));
            after_quote = false;
            pending = false;
        } else if after_quote {
            return Err(invalid(format!(
                "CSV line {line}: unexpected character after closing quote"
            )));
        } else if c == '"' {
            if !field.is_empty() {
                return Err(invalid(format!(
                    "CSV line {line}: quote inside unquoted field"
                )));
            }
            quoted = true;
        } else {
            field.push(c);
        }
    }
    if quoted {
        return Err(invalid(format!(
            "CSV line {start}: unterminated quoted field"
        )));
    }
    if pending {
        row.push(field);
        records.push((start, row));
    }
    if let Some((_, header)) = records.first() {
        let mut seen = BTreeSet::new();
        if header.iter().any(|s| s.is_empty() || !seen.insert(s)) {
            return Err(invalid("CSV header names must be nonempty and unique"));
        }
        for (line, r) in records.iter().skip(1) {
            if r.len() != header.len() {
                return Err(invalid(format!(
                    "CSV line {line} has {} fields; expected {}",
                    r.len(),
                    header.len()
                )));
            }
        }
    }
    Ok(records)
}
fn located(line: usize, column: &str, e: Error) -> Error {
    invalid(format!("CSV line {line}, column {column:?}: {}", e.0))
}
pub fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}
fn cell(s: &str, numeric: bool, missing: &[String]) -> Result<Value> {
    if missing.iter().any(|m| m == s) {
        return Ok(Value::Missing);
    }
    if numeric {
        let n: f64 = s
            .parse()
            .map_err(|_| invalid(format!("expected number, got {s:?}")))?;
        if !n.is_finite() {
            return Err(invalid("nonfinite CSV number"));
        }
        Ok(Value::Number(n))
    } else {
        Ok(Value::Text(s.to_owned()))
    }
}
pub fn dataset_from_csv(input: &str, options: &CsvOptions, task: &Task) -> Result<Dataset> {
    let lines = parse_csv_lines(input, options.delimiter)?;
    let headers = &lines.first().ok_or_else(|| invalid("empty CSV"))?.1;
    let records = &lines[1..];
    let index: BTreeMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();
    let col = |name: &str| {
        index
            .get(name)
            .copied()
            .ok_or_else(|| invalid(format!("missing CSV column {name:?}")))
    };
    let target = col(&options.target)?;
    let weight = options.weight.as_deref().map(col).transpose()?;
    let offset = options.offset.as_deref().map(col).transpose()?;
    if weight == Some(target) || offset == Some(target) || (weight.is_some() && weight == offset) {
        return Err(invalid(
            "target, weight and offset must be different columns",
        ));
    }
    let mut schema = options.features.clone();
    if schema.is_empty() {
        for (j, h) in headers.iter().enumerate() {
            if j == target || Some(j) == weight || Some(j) == offset {
                continue;
            }
            let numeric = records.iter().all(|(_, r)| {
                options.missing_values.contains(&r[j]) || r[j].parse::<f64>().is_ok()
            });
            schema.push(FeatureSchema {
                name: h.clone(),
                feature_type: if numeric {
                    FeatureType::Numeric
                } else {
                    FeatureType::Categorical
                },
                role: Role::Split,
            });
        }
    }
    let columns: Vec<usize> = schema.iter().map(|s| col(&s.name)).collect::<Result<_>>()?;
    if columns
        .iter()
        .any(|&j| j == target || Some(j) == weight || Some(j) == offset)
    {
        return Err(invalid(
            "response, weight and offset cannot also be features",
        ));
    }
    let mut data = Dataset {
        schema,
        rows: vec![],
        targets: vec![],
        weights: vec![],
        offsets: vec![],
    };
    for &(line, ref r) in records {
        let at = |j: usize, numeric: bool| {
            cell(&r[j], numeric, &options.missing_values).map_err(|e| located(line, &headers[j], e))
        };
        let row = data
            .schema
            .iter()
            .zip(&columns)
            .map(|(s, &j)| at(j, !matches!(s.feature_type, FeatureType::Categorical)))
            .collect::<Result<_>>()?;
        data.rows.push(row);
        data.targets
            .push(at(target, !matches!(task, Task::Classification))?);
        if let Some(j) = weight {
            data.weights.push(at(j, true)?.number().unwrap_or(0.0));
        }
        if let Some(j) = offset {
            data.offsets.push(
                at(j, true)?
                    .number()
                    .ok_or_else(|| located(line, &headers[j], invalid("missing exposure")))?,
            );
        }
    }
    data.validate(task)?;
    Ok(data)
}
pub fn prediction_rows_from_csv(
    input: &str,
    schema: &[FeatureSchema],
    missing: &[String],
    delimiter: char,
) -> Result<Vec<Vec<Value>>> {
    let lines = parse_csv_lines(input, delimiter)?;
    let headers = &lines.first().ok_or_else(|| invalid("empty CSV"))?.1;
    let positions: Vec<usize> = schema
        .iter()
        .map(|s| {
            headers
                .iter()
                .position(|h| h == &s.name)
                .ok_or_else(|| invalid(format!("missing feature {}", s.name)))
        })
        .collect::<Result<_>>()?;
    lines[1..]
        .iter()
        .map(|(line, r)| {
            schema
                .iter()
                .zip(&positions)
                .map(|(s, &j)| {
                    cell(
                        &r[j],
                        !matches!(s.feature_type, FeatureType::Categorical),
                        missing,
                    )
                    .map_err(|e| located(*line, &headers[j], e))
                })
                .collect()
        })
        .collect()
}
/// Reads positive prediction exposures from the named CSV column.
pub fn exposures_from_csv(input: &str, column: &str, delimiter: char) -> Result<Vec<f64>> {
    let lines = parse_csv_lines(input, delimiter)?;
    let headers = &lines.first().ok_or_else(|| invalid("empty CSV"))?.1;
    let j = headers
        .iter()
        .position(|h| h == column)
        .ok_or_else(|| invalid(format!("missing exposure column {column:?}")))?;
    lines[1..]
        .iter()
        .map(|(line, r)| {
            cell(&r[j], true, &[])?
                .number()
                .filter(|e| *e > 0.0)
                .ok_or_else(|| invalid("exposure must be positive and finite"))
                .map_err(|e| located(*line, column, e))
        })
        .collect()
}
