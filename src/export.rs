use crate::*;
use std::fmt::Write;

fn quote(s: &str) -> String {
    serde_json::to_string(s).expect("string serialization")
}
impl FittedModel {
    pub fn to_text(&self) -> Result<String> {
        self.validate()?;
        let mut out = format!(
            "{}\n{} tree(s), {} fitted leaves\n",
            self.algorithm_version,
            self.trees.len(),
            self.report.fitted_leaves
        );
        fn walk(model: &FittedModel, n: &Node, depth: usize, out: &mut String) {
            let label = if let Some(s) = &n.split {
                model.split_label(s)
            } else {
                model.leaf_label(&n.model)
            };
            let _ = writeln!(
                out,
                "{}node {} (n={}): {}",
                "  ".repeat(depth),
                n.id,
                n.sample_count,
                label
            );
            if let Some(l) = &n.left {
                walk(model, l, depth + 1, out);
            }
            if let Some(r) = &n.right {
                walk(model, r, depth + 1, out);
            }
        }
        for (i, t) in self.trees.iter().enumerate() {
            let _ = writeln!(out, "Tree {}", i + 1);
            walk(self, &t.root, 0, &mut out);
        }
        Ok(out)
    }
    pub fn to_dot(&self) -> Result<String> {
        self.validate()?;
        let mut out = "digraph GUIDE {\n  node [shape=box];\n".to_owned();
        fn walk(m: &FittedModel, n: &Node, t: usize, out: &mut String) {
            let text = if let Some(s) = &n.split {
                m.split_label(s)
            } else {
                m.leaf_label(&n.model)
            };
            let label = quote(&format!("{text}\nn = {}", n.sample_count));
            let _ = writeln!(out, "  t{t}n{} [label={label}];", n.id);
            for (child, side) in [(&n.left, "yes"), (&n.right, "no")] {
                if let Some(c) = child {
                    let _ = writeln!(out, "  t{t}n{} -> t{t}n{} [label=\"{side}\"];", n.id, c.id);
                    walk(m, c, t, out);
                }
            }
        }
        for (i, t) in self.trees.iter().enumerate() {
            walk(self, &t.root, i, &mut out);
        }
        out.push_str("}\n");
        Ok(out)
    }
    pub fn to_latex(&self) -> Result<String> {
        self.validate()?;
        fn escape(s: &str) -> String {
            s.chars()
                .map(|c| match c {
                    '\\' => "\\textbackslash{}".into(),
                    '{' | '}' | '$' | '&' | '#' | '%' | '_' => format!("\\{c}"),
                    '^' => "\\textasciicircum{}".into(),
                    '~' => "\\textasciitilde{}".into(),
                    _ => c.to_string(),
                })
                .collect()
        }
        fn walk(m: &FittedModel, n: &Node, out: &mut String) {
            let label = if let Some(s) = &n.split {
                m.split_label(s)
            } else {
                m.leaf_label(&n.model)
            };
            let _ = write!(out, "[{{{} (n={})}}", escape(&label), n.sample_count);
            if let Some(l) = &n.left {
                walk(m, l, out);
            }
            if let Some(r) = &n.right {
                walk(m, r, out);
            }
            out.push(']');
        }
        let mut out =
            "\\documentclass{article}\n\\usepackage{forest}\n\\begin{document}\n".to_owned();
        for t in &self.trees {
            out.push_str("\\begin{forest} for tree={draw,align=center,font=\\scriptsize}\n");
            walk(self, &t.root, &mut out);
            out.push_str("\n\\end{forest}\n\\clearpage\n");
        }
        out.push_str("\\end{document}\n");
        Ok(out)
    }
    /// Legacy R export, outside supported product scope. Truncated models are rejected.
    pub fn to_r(&self) -> Result<String> {
        self.validate()?;
        if !self.options.prediction_truncation.is_none() {
            return Err(invalid(
                "legacy R export does not support prediction_truncation; R export is outside product scope",
            ));
        }
        fn term(t: &Term, m: &FittedModel) -> String {
            let v = format!("data[[{}]][i]", quote(&m.schema[t.feature].name));
            if t.missing_indicator {
                format!("as.numeric(is.na({v}))")
            } else if let Some(c) = &t.category {
                format!(
                    "as.numeric(!is.na({v}) && as.character({v}) == {})",
                    quote(c)
                )
            } else {
                format!(
                    "(((if (is.na({v})) {:.17e} else {v}) - {:.17e}) / {:.17e})^{}",
                    t.impute, t.center, t.scale, t.power
                )
            }
        }
        fn leaf(n: &LeafModel, m: &FittedModel) -> String {
            match n {
                LeafModel::Quantiles { lower, upper } => format!("c({lower:.17e},{upper:.17e})"),
                LeafModel::Classification {
                    probabilities,
                    predicted_class,
                } => {
                    let p = format!(
                        "c({})",
                        probabilities
                            .iter()
                            .map(|p| format!("{p:.17e}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    if m.ensemble.is_none() {
                        format!(
                            "if (type == \"response\") {} else {p}",
                            quote(&m.classes[*predicted_class])
                        )
                    } else {
                        p
                    }
                }
                LeafModel::Constant { value } => format!(
                    "{value:.17e}{}",
                    if matches!(
                        m.options.task,
                        Task::Regression {
                            loss: Loss::Poisson
                        }
                    ) {
                        " * exposure[i]"
                    } else {
                        ""
                    }
                ),
                LeafModel::Linear {
                    intercept,
                    coefficients,
                    terms,
                    loss,
                    ..
                } => {
                    let eta = format!(
                        "({intercept:.17e}{})",
                        coefficients
                            .iter()
                            .zip(terms)
                            .map(|(b, t)| format!(" + {b:.17e} * ({})", term(t, m)))
                            .collect::<String>()
                    );
                    match loss {
                        Loss::Logistic => format!("plogis({eta})"),
                        Loss::Poisson => format!("exp({eta} + log(exposure[i]))"),
                        _ => eta,
                    }
                }
            }
        }
        fn expr(n: &Node, m: &FittedModel) -> String {
            if let (Some(s), Some(l), Some(r)) = (&n.split, &n.left, &n.right) {
                let v = format!("data[[{}]][i]", quote(&m.schema[s.feature()].name));
                let boolean = |b: bool| if b { "TRUE" } else { "FALSE" };
                let condition = match s {
                    Split::Linear {
                        features,
                        coefficients,
                        threshold,
                        missing_left,
                    } => {
                        let b = format!("data[[{}]][i]", quote(&m.schema[features[1]].name));
                        format!(
                            "if (is.na({v}) || is.na({b})) {} else ({v} * {:.17e} + {b} * {:.17e}) <= {threshold:.17e}",
                            boolean(*missing_left),
                            coefficients[0],
                            coefficients[1]
                        )
                    }
                    Split::Missing { .. } => format!("is.na({v})"),
                    Split::Numeric {
                        threshold,
                        missing_left,
                        ..
                    } => format!(
                        "if (is.na({v})) {} else {v} <= {threshold:.17e}",
                        boolean(*missing_left)
                    ),
                    Split::Categorical {
                        left_categories,
                        known_categories,
                        missing_left,
                        unknown_left,
                        ..
                    } => format!(
                        "if (is.na({v})) {} else if (!(as.character({v}) %in% c({}))) {} else as.character({v}) %in% c({})",
                        boolean(*missing_left),
                        known_categories
                            .iter()
                            .map(|s| quote(s))
                            .collect::<Vec<_>>()
                            .join(","),
                        boolean(*unknown_left),
                        left_categories
                            .iter()
                            .map(|s| quote(s))
                            .collect::<Vec<_>>()
                            .join(",")
                    ),
                    Split::Periodic {
                        start,
                        end,
                        period,
                        missing_left,
                        ..
                    } => format!(
                        "if (is.na({v})) {} else (({v} %% {period:.17e}) > {start:.17e} {} ({v} %% {period:.17e}) <= {end:.17e})",
                        boolean(*missing_left),
                        if start <= end { "&&" } else { "||" }
                    ),
                };
                format!(
                    "if ({condition}) {{ {} }} else {{ {} }}",
                    expr(l, m),
                    expr(r, m)
                )
            } else {
                leaf(&n.model, m)
            }
        }
        let mut out="# Generated by guide-rs. Missing values must be R NA.\npredict_guide <- function(data, exposure = NULL, type = c(\"response\", \"probability\")) {\n  type <- match.arg(type)\n  n <- nrow(data)\n".to_owned();
        for f in &self.schema {
            let _ = writeln!(
                out,
                "  if (!({} %in% names(data))) stop({})",
                quote(&f.name),
                quote(&format!("missing feature {}", f.name))
            );
            if f.role != Role::Excluded && !matches!(f.feature_type, FeatureType::Categorical) {
                let _ = writeln!(
                    out,
                    "  if (!is.numeric(data[[{}]]) && !all(is.na(data[[{}]]))) stop(\"numeric feature required\")\n  if (any(!is.finite(data[[{}]]) & !is.na(data[[{}]]))) stop(\"nonfinite feature\")",
                    quote(&f.name),
                    quote(&f.name),
                    quote(&f.name),
                    quote(&f.name)
                );
            }
        }
        if self.requires_exposure {
            out.push_str("  if (is.null(exposure)) stop(\"exposure required\")\n");
        }
        out.push_str("  if (is.null(exposure)) exposure <- rep(1, n)\n  if (length(exposure) != n || any(!is.finite(exposure) | exposure <= 0)) stop(\"invalid exposure\")\n");
        let dual_quantile = matches!(
            self.options.task,
            Task::Regression {
                loss: Loss::DualQuantile { .. }
            }
        );
        let single_classifier = !self.classes.is_empty() && self.ensemble.is_none();
        if dual_quantile {
            out.push_str(
                "  result <- matrix(0, n, 2)\n  colnames(result) <- c(\"lower\", \"upper\")\n",
            );
        } else if self.classes.is_empty() {
            out.push_str("  result <- numeric(n)\n");
        } else {
            let _ = writeln!(
                out,
                "  result <- matrix(0, n, {})\n  colnames(result) <- c({})",
                self.classes.len(),
                self.classes
                    .iter()
                    .map(|s| quote(s))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            if single_classifier {
                out.push_str("  if (type == \"response\") result <- character(n)\n");
            }
        }
        out.push_str("  for (i in seq_len(n)) {\n");
        for t in &self.trees {
            if single_classifier {
                let _ = writeln!(
                    out,
                    "    value <- {}\n    if (type == \"response\") result[i] <- value else result[i,] <- value",
                    expr(&t.root, self)
                );
                continue;
            }
            let target = if self.classes.is_empty() && !dual_quantile {
                "result[i]"
            } else {
                "result[i,]"
            };
            let _ = writeln!(
                out,
                "    {target} <- {target} + ( {} ) / {}",
                expr(&t.root, self),
                self.trees.len()
            );
        }
        out.push_str("  }\n");
        if single_classifier {
            out.push_str("  return(result)\n");
        } else if !self.classes.is_empty() {
            let k = self.classes.len();
            let costs: Vec<f64> = (0..k)
                .flat_map(|i| {
                    (0..k).map(move |j| {
                        self.options
                            .class_costs
                            .as_ref()
                            .map(|c| c[i][j])
                            .unwrap_or(if i == j { 0.0 } else { 1.0 })
                    })
                })
                .collect();
            let _ = writeln!(
                out,
                "  if (type == \"probability\") return(result)\n  if (n == 0) return(character())\n  costs <- matrix(c({}), nrow = {k}, byrow = TRUE)\n  return(colnames(result)[max.col(-(result %*% costs), ties.method = \"first\")])",
                costs
                    .iter()
                    .map(|c| format!("{c:.17e}"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        } else {
            out.push_str("  return(result)\n");
        }
        out.push_str("}\n");
        Ok(out)
    }
    fn split_label(&self, s: &Split) -> String {
        let f = &self.schema[s.feature()].name;
        match s {
            Split::Linear {
                features,
                coefficients,
                threshold,
                missing_left,
            } => format!(
                "{:.6} * {} + {:.6} * {} <= {threshold:.6}; missing -> {}",
                coefficients[0],
                self.schema[features[0]].name,
                coefficients[1],
                self.schema[features[1]].name,
                if *missing_left { "yes" } else { "no" }
            ),
            Split::Missing { .. } => format!("{f} is missing"),
            Split::Numeric {
                threshold,
                missing_left,
                ..
            } => format!(
                "{f} <= {threshold:.6}; missing -> {}",
                if *missing_left { "yes" } else { "no" }
            ),
            Split::Categorical {
                left_categories,
                missing_left,
                unknown_left,
                ..
            } => format!(
                "{f} in {:?}; missing -> {}, unseen -> {}",
                left_categories,
                if *missing_left { "yes" } else { "no" },
                if *unknown_left { "yes" } else { "no" }
            ),
            Split::Periodic {
                start,
                end,
                period,
                missing_left,
                ..
            } => format!(
                "{f} mod {period} in ({start:.6}, {end:.6}]; missing -> {}",
                if *missing_left { "yes" } else { "no" }
            ),
        }
    }
    fn leaf_label(&self, l: &LeafModel) -> String {
        match l {
            LeafModel::Quantiles { lower, upper } => format!("quantiles ({lower:.6}, {upper:.6})"),
            LeafModel::Classification {
                predicted_class,
                probabilities,
            } => format!(
                "class {:?}; p={:?}",
                self.classes[*predicted_class], probabilities
            ),
            LeafModel::Constant { value } => format!("prediction {value:.6}"),
            LeafModel::Linear {
                coefficients,
                converged,
                prediction_bounds,
                ..
            } => {
                let limits = prediction_bounds
                    .as_ref()
                    .map(|b| format!("; prediction limits [{:.6e}, {:.6e}]", b.lower, b.upper))
                    .unwrap_or_default();
                format!(
                    "linear model ({} terms; converged={converged}{limits})",
                    coefficients.len()
                )
            }
        }
    }
}
