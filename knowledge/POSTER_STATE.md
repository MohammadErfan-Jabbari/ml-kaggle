# Poster State (A1, LaTeX) — AMR Kaggle Project

**Authors**: MohammadErfan Jabbari$^{1,2}$, Ana Mirza$^{1}$
**Affiliations**: $^{1}$Universidad Carlos III de Madrid, $^{2}$IMDEA Networks Institute
**Poster source**: `poster/main.tex`
**Printing target**: A1 portrait (594mm × 841mm)

---

## Current Poster Structure (as of 2026-01-21)

**Two versions exist:**
1. **A1 version**: `poster/main.tex` (59.4×84.1cm) - original
2. **50×70cm version**: `poster/main_50x70.tex` - **FINAL FOR PRINT**

**Layout**: 2-column beamerposter, 6 numbered sections

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

### 50×70cm Version Settings (FINAL - as of 2026-01-21 Session 3)
- **Size**: `width=50, height=70` (custom size in beamerposter)
- **Scale**: 0.72
- **Title font**: `\fontsize{44}{52}` (scaled down from A1)
- **Header**: TikZ overlay, 5.5cm height, accent line at -5.7cm
- **Footer**: TikZ overlay, 1.5cm height, accent line 0.15cm thick
- **Footer text**: `\large` font
- **Section headers**: `\LARGE`, bold, sans-serif family
- **Body text**: `\large`
- **Block spacing**: `\vskip1.0ex` between blocks
- **References header**: `\large` bold, red accent line 1.5pt
- **References content**: `\large` font
- **Captions**: `\normalsize` with "Figure X:" / "Table X:" labels, centered

### A1 Version Settings (original)
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

### 50×70cm Version (FINAL FOR PRINT)
```bash
cd poster
pdflatex main_50x70.tex
# Optional: generate PNG preview
pdftoppm -png -r 150 main_50x70.pdf main_50x70 && mv main_50x70-1.png main_50x70.png
# Outputs: main_50x70.pdf, main_50x70.png
```

### A1 Version (original)
```bash
cd poster
pdflatex -interaction=nonstopmode main.tex
pdftoppm -png -r 150 -singlefile main.pdf main
# Outputs: main.pdf, main.png
```

---

## Next Session Priority

**50×70cm version is READY FOR PRINT** - `poster/main_50x70.pdf`

If any changes needed:
- Minor text edits can be made directly
- Orphan lines have been fixed in this session
- Footer/header are TikZ overlays positioned at page edges

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
- `knowledge/sessions/2026-01-21_poster_50x70.md` (50×70cm version created, finalized for print)
