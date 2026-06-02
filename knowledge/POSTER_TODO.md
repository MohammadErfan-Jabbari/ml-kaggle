# Poster TODO

## Completed
- [x] Section 1: The Challenges (species shift chart, sparsity, missing labels)
- [x] Header with title, authors, affiliations (superscript notation)
- [x] Basic 2-column A1 beamerposter structure
- [x] Section 2: What We Explored (color-coded exploration grid)
- [x] Section 3: Results Table (5 submissions with Public/Private/OOF)
- [x] Section 4: What Worked (rank avg, species weighting, intrinsic resistance + pipeline diagram)
- [x] Section 5: What Failed & Why (stacking, species-specific, DR, K.pn optimization + quote)
- [x] Footer with TikZ overlay (anchored to page bottom)
- [x] Header/footer accent lines (full-width red lines using TikZ)
- [x] Block padding improved (top/bottom breathing room)

## Completed (2026-01-21 Earlier)
- [x] Rank averaging formula added to Section 4
- [x] Dataset stats added to Section 1 (3,360/1,000/6,000/93%)
- [x] Training details added (5-fold CV, LightGBM hyperparams, PLS components)
- [x] References section added (DRIAMS, LightGBM citations)
- [x] Reduced header-to-content spacing
- [x] Tightened block spacing for denser layout
- [x] Balanced column heights with \vfill

## Completed (2026-01-21 This Session)
- [x] Removed Takeaway section (redundant content)
- [x] Moved References to right column (was awkward full-width)
- [x] Added condensed lesson to Results section
- [x] Removed dataset info line from Challenges (redundant)
- [x] Added section numbering (1-6)
- [x] Increased section header font (\Large → \LARGE with \sffamily)
- [x] Fixed "What We Explored" box overlaps (minimum width=3.6cm, positions spread to 2.0/6.0/10.0/14.0)
- [x] Fixed legend spacing and alignment
- [x] Added proper figure/table captions (Figure 1, Figure 2, Table 1, Figure 3)
- [x] Centered all captions
- [x] Increased caption font size (\small → \normalsize)

## Completed (2026-01-21 Session 2)
- [x] Font scale increased to 1.15 (balance between readability and fit)
- [x] Fixed header title font (58pt fixed size, doesn't scale with beamerposter)
- [x] Fixed header accent line position (-8.0cm)
- [x] Increased footer height (2.5cm) for better visibility
- [x] Fixed table column spacing with connected row colors
- [x] Added "Why PLS?" explanation to Section 4
- [x] Added 2 more references (XGBoost, CatBoost) - now 4 total
- [x] Shortened references to compact format

## Completed (2026-01-21 Session 3 - 50×70cm Version)
- [x] Created `main_50x70.tex` for 50×70cm poster board size
- [x] Scaled content to fit (scale=0.72)
- [x] Header extends edge-to-edge (TikZ overlay)
- [x] Footer extends edge-to-edge (TikZ overlay, 1.5cm height)
- [x] All 3 figures scaled down proportionally
- [x] Table overflow fixed (tabcolsep reduced)
- [x] Author name updated to Ana-Maria Mirza
- [x] References section: `\large` font, proper spacing
- [x] Footer text: `\large` font, centered
- [x] Fixed orphan lines in right column:
  - "clinical rules apply." → "resistance rules."
  - "peaks." → removed "but" for better flow
  - "Underfitting dominated." → "Models underfit."
  - "macro-averaged." → "lower the macro-averaged score."
- [x] Fixed orphan lines in left column:
  - "that matter." / "features." → "compressing too hard discards the signal."
- [x] Reduced space above references section (-0.8em)
- [x] Reduced space between sections (1.0ex)
- [x] Figure 2 caption spacing tightened (0.2em)

## READY FOR PRINT
- **File**: `poster/main_50x70.pdf`
- **Size**: 50cm × 70cm (matches poster board)
- **Build**: `cd poster && pdflatex main_50x70.tex`

## If Changes Needed
- [ ] Any final text corrections
- [ ] Verify embedded fonts for print shop

## Skipped (low value or redundant)
- AUC definition (too generic)
- BCE loss formula (not used explicitly)
- Training time (32 min CPU, not impressive)
- Experiment count (not well documented)
- PLS citation (too old, 2001)
- Top 10% ranking mention (AUC already shown)
