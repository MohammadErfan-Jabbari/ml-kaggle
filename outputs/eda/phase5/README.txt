Phase 5 EDA: Dimensionality Reduction & Clustering
===================================================

Generated: 2026-01-07

Files Generated:
--------------
1. pca_scree_plot.png (4168x1471, 300 DPI)
   - Explained variance per principal component
   - Cumulative explained variance curve
   - Shows 80/90/95% variance thresholds

2. pca_species_resistance.png (4117x1771, 300 DPI)
   - PC1 vs PC2 colored by species (4 species)
   - PC1 vs PC2 colored by Ertapenem resistance
   - Visualizes species separation in PCA space

3. clustering_kmeans.png (4168x1771, 300 DPI)
   - True species labels in PC space
   - K-means clusters (k=4) in PC space
   - Shows alignment between clusters and species

4. species_linear_separability.png (2968x1761, 300 DPI)
   - Logistic regression accuracy vs PCA dimensions
   - Cross-validation accuracy curve
   - Shows species are highly linearly separable

5. eda_phase5_dimensionality.md
   - Comprehensive markdown report
   - Located in /knowledge/insights/
   - Contains all numerical results and interpretations

Key Findings:
------------
1. HIGH INTRINSIC DIMENSIONALITY
   - PC1: 6.27%, PC2: 5.24%, PC3: 3.69%
   - Need 202 PCs for 80% variance
   - Need 433 PCs for 90% variance
   - No dominant low-dimensional structure

2. SPECIES ARE HIGHLY SEPARABLE
   - 99.7% accuracy with 10 PCs (logistic regression)
   - Silhouette scores peak at 10 PCs (0.699)
   - Species is the strongest signal in the data

3. CLUSTERS ALIGN WITH SPECIES
   - k=4 ARI=0.785 (strong alignment)
   - K-means recovers species structure
   - Contingency table shows clean separation

4. RESISTANCE IS DISTRIBUTED
   - Low silhouette scores (< 0.3 for all antibiotics)
   - Imipenem and Ertapenem show highest structure
   - No clean R vs S clusters
   - Resistance requires complex, non-linear models

Implications for Modeling:
-------------------------
✓ Use species as a strong feature (embedding)
✓ Deep learning justified (high-dimensional patterns)
✓ Don't aggressively reduce dimensions
✓ Consider species-specific models
✓ Non-linear models (NN, GBM) required for resistance
✗ PCA for feature engineering (loses signal)

Recommendations:
---------------
1. Include species embedding in model architecture
2. Use full 5635 features (after constant removal)
3. Try hybrid: global + species-specific heads
4. Species-stratified cross-validation
5. Monitor per-species AUC (avoid P. aeruginosa bias)

Script Location:
--------------
/sata_disk/users/erfan/ml_kaggle/scripts/eda/phase5_dimensionality_reduction.py

Run with:
--------
uv run python scripts/eda/phase5_dimensionality_reduction.py

Optional: Install UMAP for additional visualization
---------------------------------------------------
uv add umap-learn
