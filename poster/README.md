# Poster (A1, Portrait) — LaTeX

Source: `poster/main.tex`

## Build
Recommended:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- `./build.sh` (builds `main.pdf` and a PNG preview `main.png`)

If `latexmk` is not available:
- `pdflatex -interaction=nonstopmode -halt-on-error main.tex` (run 2–3 times)

## Template
Default poster: `poster/main.tex` (A1 portrait).

Build with PNG preview:
```bash
cd poster
./build.sh
```

Alternate template (kept for comparison): `poster/main_modernposter.tex` (uses `poster/modernposter.cls`).

## Figures
Put poster figures in `poster/figs/` (or update paths in `main.tex` to point into `outputs/eda/`).
