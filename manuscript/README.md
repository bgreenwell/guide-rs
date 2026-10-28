# guide-rs manuscript

Working skeleton for a **Journal of Statistical Software (JSS)** article.
`manuscript/` is independent of Cargo and excluded from the Rust package.
The draft contains explicit TODOs, not completed analyses or submission claims.
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

## Proposed paper scope and workflow

These are project choices, not additional journal rules:

1. Confirm author information and the paper's supported release and feature set.
   Start with constant classification/regression and multiple-linear trees.
   Treat wider coverage and experimental boosting as scope decisions.
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

The initial skeleton has no empirical results, so a successful LaTeX build
does not establish replication readiness. Generated build files stay out of Git.
