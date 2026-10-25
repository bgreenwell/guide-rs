use crate::*;
use std::fmt::Write;

fn quote(s: &str) -> String {
    // Serializing a plain &str to JSON cannot fail.
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
