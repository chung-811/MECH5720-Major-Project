# Report

The ICCP-format report (two columns, 6 pages plus references). It was written on Overleaf;
this folder holds a copy of its source.

- `main.tex`, `sections/`: the text, one file per section.
- `figures/make_figures.py`: redraws every figure and writes `figures/numbers.tex`, the numbers the text
  quotes. Run it from `software/deconvolution`:
  `uv run python ../../report/figures/make_figures.py`. Its docstring lists the inputs. The rig figures
  need only `figures/data/` (in this repo); the simulation figures also need the Blender renders.
- `refs.bib`: references.

Build: `latexmk -pdf main.tex` (or upload the folder to Overleaf).
