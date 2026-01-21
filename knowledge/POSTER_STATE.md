# Poster State (A1, LaTeX) — AMR Kaggle Project

**Authors**: MohammadErfan Jabbari$^{1,2}$, Ana Mirza$^{1}$
**Affiliations**: $^{1}$Universidad Carlos III de Madrid, $^{2}$IMDEA Networks Institute
**Poster source**: `poster/main.tex`
**Printing target**: A1 portrait (594mm × 841mm)

---

## Current Poster Structure (as of 2026-01-21)

**Layout**: 2-column beamerposter, A1 portrait, 6 numbered sections

```
┌─────────────────────────────────────────────────────┐
│              HEADER (Title + Authors)               │
├────────────────────────┬────────────────────────────┤
│                        │                            │
│  1. THE CHALLENGES     │  4. WHAT WORKED            │
│  - Species shift chart │  - Rank avg formula        │
│  - Sparsity            │  - Species weighting       │
│  - Missing labels      │  - Intrinsic resistance    │
│  [Figure 1]            │  - Pipeline diagram        │
│                        │  [Figure 3]                │
├────────────────────────┼────────────────────────────┤
│                        │                            │
│  2. WHAT WE EXPLORED   │  5. WHAT FAILED & WHY      │
│  - Color-coded grid    │  - Stacking, Species-spec  │
│  - Legend              │  - DR, K.pn optimization   │
│  [Figure 2]            │  - Quote box               │
│                        │                            │
├────────────────────────┼────────────────────────────┤
│                        │                            │
│  3. RESULTS            │  6. REFERENCES             │
│  - 5-row table         │  - DRIAMS paper            │
│  - Lesson line         │  - LightGBM paper          │
│  [Table 1]             │                            │
│                        │                            │
└────────────────────────┴────────────────────────────┘
│                     FOOTER                          │
└─────────────────────────────────────────────────────┘
```

---

## Technical Parameters

### Current Settings (as of 2026-01-21 Session 2)
- **Scale**: 1.15
- **Title font**: `\fontsize{58}{66}` (fixed, doesn't scale with beamerposter)
- **Header height**: 7cm, accent line at -8.0cm
- **Footer height**: 2.5cm with red accent line
- **Section headers**: `\LARGE`, bold, sans-serif family
- **Body text**: `\large`
- **Captions**: `\normalsize` with "Figure X:" / "Table X:" labels, centered
- **Table**: `\tabcolsep{1.2em}` for connected row colors
- **References**: 4 total, compact format

### Figure 2 (What We Explored) Box Settings
- Box minimum width: 3.6cm
- Box positions: x = 2.0, 6.0, 10.0, 14.0 (4cm spacing)
- Category labels at x = -1.2
- Colors: DarkBlue!85 (worked), orange!80!black (partial), Accent!85 (failed)

### Colors
- DarkBlue: #1a365d (header, titles, worked items)
- Accent: #e53e3e (red lines, failed items)
- LightGray: #f7fafc (block backgrounds)
- orange!80!black (partial success items)

---

## Style Decisions

- **No em-dashes**: Rewrite sentences to avoid them
- **Captions**: Centered, "Figure X:" or "Table X:" prefix, normalsize font
- **Section numbering**: 1-6 in section titles
- **Figures**: TikZ/pgfplots for consistency (vector, matches poster style)
- **Tone**: Technical but with conceptual insight sentences

---

## Results Cited in Table 1

| Submission | Public | Private | OOF CV |
|------------|--------|---------|--------|
| Mega-blend (rank avg) | **0.8386** | **0.8131** | 0.8174 |
| Self-training blend | 0.8366 | 0.8094 | 0.8052 |
| LightGBM + PLS | 0.8323 | 0.8012 | 0.7986 |
| Stacking (overfit) | 0.8269 | 0.7840 | *0.9545* |
| Baseline | 0.8022 | 0.7856 | 0.7823 |

---

## Build Commands

```bash
cd poster
pdflatex -interaction=nonstopmode main.tex
pdftoppm -png -r 150 -singlefile main.pdf main
# Outputs: main.pdf, main.png
```

---

## Next Session Priority

**Font scaling**: Change `scale=1.12` to `scale=1.2` (or 1.25) in:
```latex
\usepackage[size=a1,orientation=portrait,scale=1.2]{beamerposter}
```

This single change will proportionally increase ALL font sizes across the poster.

---

## Professor's Guidelines (from email)

**Key points:**
- Focus on **exploration and methodology**, not just best solution
- **No need to describe problem or dataset** - everyone is familiar
- Show **alternative approaches even if they failed**
- Poster boards: **50 × 70 cm** (width × height) - NOTE: current A1 may need resizing
- Evaluation: methodology, design/clarity, team contribution
- Worth 10% of grade, possible +5% for strong work

**Do NOT add:**
- Dataset statistics (not needed)
- Problem description (everyone knows it)

**DO emphasize:**
- Different approaches explored
- Why things worked or failed
- Methodology and process

---

## Session Notes Index

- `knowledge/sessions/2026-01-20_poster.md` (initial scaffold)
- `knowledge/sessions/2026-01-21_poster.md` (content completion, styling, captions, section numbering)
