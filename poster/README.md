# Poster (A1) — LaTeX

Source: `poster/main.tex`

## Build
Recommended:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

If `latexmk` is not available:
- `pdflatex -interaction=nonstopmode -halt-on-error main.tex` (run 2–3 times)

## Figures
Put poster figures in `poster/figs/` (or update paths in `main.tex` to point into `outputs/eda/`).

