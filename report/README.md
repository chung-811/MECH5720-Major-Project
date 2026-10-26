# Report

Two-column ICCP paper, 6 pages + references.

- `main.tex`: title, authors, section order. The preamble is a stand-in: paste in the official ICCP
  template preamble (Canvas: iccp2026_latex_template.zip).
- `sections/`: one file per section. Each starts with what the brief asks for.
- `figures/make_figures.py`: redraws every figure and writes `figures/numbers.tex`, the numbers the
  text quotes (never type a result by hand). Run from `software/deconvolution`:
  `uv run python ../../report/figures/make_figures.py`. Its docstring lists the inputs.
- Grey boxes (`\placeholder{...}`) are figures still to make; swap in `\includegraphics[width=\linewidth]{name}`.
- `refs.bib`: papers only.

Build: `tectonic main.tex` (or `latexmk -pdf main.tex`).

Markers to fill in: `\TODO{...}` (red text) and `\X` (a number still to measure). Before submitting,
set `\showtodosfalse` in `main.tex` and check that no grey placeholder boxes are left.

## Before submitting (from the brief)

- [ ] 6 pages max, not counting references; ICCP template
- [ ] Every figure and table is mentioned in the text; captions make sense on their own
- [ ] At least one plot with axes, units and uncertainty (mean ± std over ≥10 captures)
- [ ] Abstract: no citations, no acronyms, key numbers included
- [ ] Repo public, named `mech5720-2026-<short name>`, GitHub topic `mech5720`, licence file
- [ ] Repo README starts with unit code, year and project title; parts list, build steps with photos,
      capture procedure, pitfalls
- [ ] One script that regenerates the key figure from data in the repo (or a stable public link)
