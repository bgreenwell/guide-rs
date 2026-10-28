# guide-rs manuscript

Working draft for a **Journal of Statistical Software (JSS)** article.
`manuscript/` is independent of Cargo and excluded from the Rust package.
The abstract and introduction establish the contribution and scope; later
sections retain explicit TODOs. Empirical analyses remain to be completed.
The paper leads with the Rust implementation, library API, and CLI. The
scikit-learn-compatible Python wrapper follows as a convenient interface to
the same core.

## Files and building

- `article.tex`: editable manuscript, metadata, and section prompts.
- `refs.bib`: starter bibliography; extend it as the paper develops.
- `jss.cls`, `jss.bst`, `jsslogo.jpg`: unchanged official typesetting files.
- `jss.pdf`: official class manual; see `UPSTREAM.md` for provenance and licenses.
- `build/`: ignored PDF and intermediate files, created when building.

Install a TeX distribution with pdfLaTeX, BibTeX, and latexmk (for example,
TeX Live, MacTeX, or TinyTeX). From the repository root:

```sh
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build article.tex
```

Open `build/article.pdf`. This builds the manuscript only; there is no paper
analysis/replication script yet. The skeleton needs no Rust build or datasets.
If your minimal TeX installation reports a missing package, install that package
through its package manager. Keep the class defaults rather than overriding
fonts, margins, or citation formatting.

Without latexmk, run from `manuscript/`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error article.tex
bibtex article
pdflatex -interaction=nonstopmode -halt-on-error article.tex
pdflatex -interaction=nonstopmode -halt-on-error article.tex
```

That alternative writes `article.pdf` and intermediates here; they are ignored.

## JSS requirements

Checked against the official guidance on **2026-10-27**; recheck before submission.

[Mission](https://www.jstatsoft.org/mission): explain the statistical concepts,
implementation design, related software, and a substantive application.
Reproducibility is mandatory. New methodology should be published elsewhere
first; focus this article on the software contribution.

[Author instructions](https://www.jstatsoft.org/authors) and
[submission checklist](https://www.jstatsoft.org/about/submissions):

- Submit an English PDF in JSS style, software source, and replication material
  for every result, preferably a single commented standalone script.
- Compare software of similar scope, including other languages; explain
  advantages and limitations. Extensive simulations are discouraged.
- Supply readable, commented source, installation packaging, and documentation.
  A standard package repository is encouraged where possible.
- Include a GPL-compatible software license. Preserve third-party notices.
- At final preparation, provide the manuscript sources and graphics as well.

## Writing and formatting

Follow the [style guide](https://www.jstatsoft.org/style), the bundled manual,
and the [official article template](https://www.jstatsoft.org/public/journals/1/jss-article-tex.zip).

- Use pdfLaTeX and `\documentclass[article]{jss}` with minimal extra packages.
- Mark languages with `\proglang{Rust}`, packages with `\pkg{guide-rs}`, and
  identifiers with `\code{GuideRegressor}`, including in references.
- Use title case for the title and bibliography titles; sentence case for
  headings, captions, and table headers.
- Maintain markup-free `\Plainauthor`, `\Plaintitle`, and `\Plainkeywords`.
  Give marked-up section headings a plain optional bookmark title.
- Put figure and table captions below the object; end them with a period.
  Put table notes in captions and use readable figure labels.
- Label and cross-reference figures, tables, and equations with `\label`/`\ref`.
- Use `Code` for non-executed listings, `CodeInput`/`CodeOutput` for examples,
  and `\code` inline. Keep lines within the text width and explain code in prose.
- Cite software using its recommended references. Keep citations in BibTeX;
  use `\citet`/`\citep`, complete journal titles, and available DOIs.
- Put optional, descriptively titled appendices after the bibliography.

## Contribution and scope

The central contribution is **open, reusable statistical computing
infrastructure for GUIDE-style trees**. Present the contributions in this order:

1. An independent, MIT-licensed implementation that users can inspect, build,
   modify, redistribute, and embed in other applications.
2. A reusable Rust library and explicit CLI for scripted fitting, inspection,
   model persistence, and prediction.
3. Native deployment on tested Linux, macOS, and Windows targets, without the
   original GUIDE executable, a Python interpreter for CLI use, or external
   BLAS/LAPACK dependencies.
4. A complementary scikit-learn-compatible Python wrapper using the same core.

Native comparisons and independent numerical checks support these claims;
their scope and known differences must be reported. Demonstrate convenience
through complete workflows and concrete integration steps.

The methodological treatment and detailed evaluation center on variable
selection, split search, constant classification/least-squares trees,
multiple-linear least-squares leaves, pruning, and associated predictor and
missing-value handling. Describe other implemented losses, node models,
ensembles, and importance with their validation status. Identify boosting as
an experimental extension. Survival, treatment-effect, longitudinal, and
multiresponse methods are outside the implemented scope.

Credit original GUIDE methods and distinguish a subset implementation from
full compatibility. Original GUIDE already supports batch operation and
provides executables for the three major desktop/server operating systems;
the distinction is source availability, buildability, reusable interfaces,
and integration. Avoid claims of universal OS support, superior speed or
accuracy, or measured usability improvements without corresponding evidence.

## Writing workflow

These are project choices, not additional journal rules:

1. Confirm author information and pin the software release. Audit the agreed
   scope above against that release and document its known differences.
2. Develop the statistical motivation and software design together. Explain why
   separating variable selection from split search helps users and what the
   Rust library and CLI make practical. Present those interfaces first, then
   show the Python wrapper's convenience for existing estimator workflows.
3. Choose a substantive real-data application with distributable inputs. Record
   provenance, preprocessing, evaluation splits, and interpretation. Lead the
   worked analysis with the CLI and follow with a concise Python equivalent.
4. Plan fair capability and empirical comparisons. Pin comparator versions,
   tuning budgets, seeds, metrics, and hardware/thread settings. Cite original
   GUIDE work separately from this implementation.
5. Add a commented replication entry point that runs the CLI analysis and
   Python wrapper examples, with locked dependencies when analyses exist.
   Have it generate figures, tables,
   and example output; include those outputs in LaTeX instead of transcribing
   results. State expected runtime and commands in this README.
6. Archive the eventual software and replication release, then fill in
   Computational details. Do not invent a DOI or publication metadata.

Use [coverage](../docs/coverage.md), [algorithm documentation](../docs/algorithms.md),
[validation](../docs/validation.md), and the
[selection-bias study](../docs/selection-bias-study.md) as evidence inventories.
Audit their dated claims against the selected release before using numbers.
Native agreement is not itself proof of correctness; retain independently
verified QR and pruning behavior and explain known differences. Do not claim
complete GUIDE parity or universal unbiasedness from finite experiments.
Replication must not silently depend on ignored local binaries or papers.

## Before sharing or submitting

- Search `article.tex` for `TODO` and confirm author metadata and scope.
- Rebuild; resolve missing citations/references and overflowing text, then
  inspect every PDF page, including figures and code listings.
- Run the replication workflow from a clean environment and reconcile every
  reported result with its generated output.
- Include data provenance, permitted redistribution, dependencies, versions,
  seeds, installation instructions, and known limitations in the archive.
- Check the live JSS instructions and assemble the PDF, software, replication
  materials, and final manuscript sources as required.

The current draft has no empirical results, so a successful LaTeX build
does not establish replication readiness. Generated build files stay out of Git.
