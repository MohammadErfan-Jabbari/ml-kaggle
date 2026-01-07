# ML Course Methods Reference
> **Source**: Professor's course notebooks (13 Jupyter notebooks analyzed)
> **Purpose**: Tools and methods expected for AMR Prediction competition

---

## Quick Reference: Methods by Category

| Category | Methods | Key Libraries | AMR Relevance |
|----------|---------|---------------|---------------|
| **Dimensionality Reduction** | PCA, Probabilistic PCA, Kernel PCA, PLS, CCA, KPCA | `sklearn.decomposition`, `sklearn.cross_decomposition` | Handle 6000 MALDI features |
| **Classification** | SVM (Linear/RBF), One-Class SVM, Logistic Regression, K-NN | `sklearn.svm`, `sklearn.neighbors` | Multi-label (8 antibiotics) |
| **Clustering** | K-Means, Silhouette Analysis, Elbow Method | `sklearn.cluster` | Semi-supervised labeling, data exploration |
| **Ensemble Methods** | Bagging, AdaBoost, Gradient Boosting, XGBoost, Stacking | `sklearn.ensemble`, `xgboost` | Baseline model (LightGBM/GBM) |
| **Bayesian Methods** | Bayesian Regression, Gaussian Processes, ARD | `GPy`, `sklearn.gaussian_process`, `torch` | Uncertainty quantification, small data |
| **Novelty Detection** | One-Class SVM, SVDD, Outlier Analysis | `sklearn.svm.OneClassSVM`, `cvxopt` | Species shift, OOD detection |
| **Kernel Methods** | Kernel Ridge Regression, Kernel Logistic Regression, Nyström, RFF | `sklearn.kernel_ridge`, `sklearn.kernel_approximation` | Non-linear relationships |
| **Feature Engineering** | Standardization, Supervised FE (PLS/CCA), Kernel Approximation | `sklearn.preprocessing`, `sklearn.pipeline` | Leakage-safe pipelines |

---

## Detailed Notebook Breakdown

### 1. NoveltyDetection_1SVM.ipynb
**Topics**: Novelty Detection, One-Class SVM, Outlier Analysis, PCA + SVM Pipeline
**Methods**:
- One-Class SVM (OCSVM) with `nu` parameter (controls outlier fraction)
- PCA for dimensionality reduction before classification
- StandardScaler for normalization
- GridSearchCV for hyperparameter tuning

**AMR Application**:
- Detect OOD samples due to species shift (P. aeruginosa 43%→3%)
- Pipeline: `StandardScaler → PCA → OneClassSVM` for filtering unusual spectra
- Reject samples before main classifier to prevent degradation

**Key Parameters**: `nu` (outlier tolerance), `gamma` (RBF width), `n_components` (PCA)

---

### 2. KernelMethods_student.ipynb
**Topics**: Kernel Trick, SVM, Kernel Ridge Regression, Kernel Logistic Regression
**Methods**:
- SVC with RBF kernel (`gamma`, `C`)
- KernelRidge (KRR) for regression
- Custom Kernel Logistic Regression pipeline
- Polynomial kernel extensions

**AMR Application**:
- KLR for multi-label probabilities (better than hard SVM boundaries)
- Heavily regularized RBF (low `gamma`, high `alpha`) to prevent memorizing species noise
- Kernel methods efficient for sparse data (93% zeros in MALDI)

**Key Parameters**: `gamma` (RBF reach), `C` (SVM regularization), `alpha` (KRR regularization)

---

### 3. GP_regression_professor.ipynb
**Topics**: Gaussian Processes for Regression, Kernel Composition, Log-Marginal Likelihood
**Methods**:
- GaussianProcessRegressor with scikit-learn
- Kernel composition: `RBF + WhiteKernel`, `ConstantKernel * RBF`
- LML optimization with `n_restarts_optimizer`
- Uncertainty quantification (95% confidence intervals)

**AMR Application**:
- Small data robustness (3360 samples favor GP over deep NN)
- Uncertainty estimates flag low-confidence predictions
- Custom kernels for sparse MALDI features

**Key Parameters**: `length_scale` (smoothness), `noise_level`, `n_restarts_optimizer`

---

### 4. Clustering_Kmeans.ipynb
**Topics**: K-Means, Elbow Method, Silhouette Analysis
**Methods**:
- KMeans clustering with `n_init` for stability
- Silhouette score for cluster validation
- Elbow method for K selection

**AMR Application**:
- **Semi-supervised labeling**: Cluster similar samples to infer missing labels (Amox/Clav 42.8% missing)
- Data exploration: Check if MALDI features group by species or resistance patterns
- Feature engineering: Use cluster assignments as meta-features

**Key Parameters**: `n_clusters`, `n_init`, `max_iter`

---

### 5. FE_homework_FashionMNIST_sol_part1.ipynb
**Topics**: Supervised Feature Extraction, Data Leakage Prevention, Kernel Approximations
**Methods**:
- PCA (unsupervised), PLS (supervised), CCA (supervised linear)
- KCCA (Kernel CCA), PCA + CCA pipeline
- Nyström method for kernel approximation
- RBF Sampler (Random Fourier Features)
- GridSearchCV with leakage-safe pipelines

**AMR Application**:
- **PLS/CCA**: Use antibiotic labels to find resistance-specific spectral peaks (better than PCA)
- **PCA pre-processing**: Reduce 6000→200-500 features before CCA to prevent rank deficiency
- **Leakage-safe CV**: Critical for small dataset (3360 samples)
- **Kernel approximations**: Scale non-linear methods without O(N³) cost

**Key Parameters**: `n_components` (K), `gamma` (RBF), `m` (landmarks for Nyström)

---

### 6. Introduction_to_SVMs_student.ipynb
**Topics**: SVM Theory, Primal vs Dual, Soft Margin, Hinge Loss, Multiclass
**Methods**:
- LinearSVC (primal) vs SVC(kernel='linear') (dual)
- One-vs-Rest and One-vs-One multiclass strategies
- Soft margin with slack variables
- StandardScaler for SVM preprocessing

**AMR Application**:
- **One-vs-Rest**: Train 8 separate SVMs (one per antibiotic)
- Sparse SVM support vectors robust in high-dimensional MALDI space
- Tune `C` critical: high C risks overfitting to 6000 features

**Key Parameters**: `C` (regularization), `kernel`, `class_weight`

---

### 7. GP_withGPy_v2.ipynb
**Topics**: Advanced GP with GPy, Kernel Engineering, Sparse GPs, ARD
**Methods**:
- GPy library for advanced GP features
- Kernel engineering: sum (+) and multiply (*) kernels
- ARD (Automatic Relevance Determination) for feature selection
- Sparse GP with inducing points (O(NM²) vs O(N³))

**AMR Application**:
- **ARD**: Automatically learn which MALDI peaks matter (feature selection for 6000 features)
- **Sparse GPs**: Necessary for full dataset (3360 samples)
- Kernel sums: Combine MALDI kernel + species kernel

**Key Parameters**: `variance`, `lengthscale`, `ARD=True`, `num_inducing`

---

### 8. Introduction_Ensembles_students.ipynb
**Topics**: Bagging, Boosting, Stacking, Variance Reduction
**Methods**:
- BaggingClassifier with DecisionTreeClassifier
- AdaBoost (SAMME, SAMME.R)
- GradientBoostingClassifier
- XGBoost with early stopping
- StackingClassifier
- Out-of-Bag (OOB) estimation

**AMR Application**:
- **Bagging**: Variance reduction for small data (3360 samples)
- **max_features sampling**: Increase diversity, prevent overfitting to P. aeruginosa
- **OOB score**: Validation without separate set
- **Feature importance**: Average across bagged trees for robust feature selection

**Key Parameters**: `n_estimators`, `max_depth`, `max_samples`, `max_features`, `learning_rate`

---

### 9. Feature_extraction_professor.ipynb
**Topics**: Supervised vs Unsupervised FE, Kernel Methods, Data Leakage
**Methods**:
- PLS (maximizes X-Y covariance)
- CCA (maximizes X-Y correlation)
- PLSSVD, PLSCanonical, PLSRegression
- Kernel PCA, Kernel PLS, Kernel CCA
- Multi-label handling with `label_binarize`

**AMR Application**:
- **PLSRegression**: Find MALDI peaks predictive of each antibiotic
- **Multi-label**: Use `label_binarize` for 8 antibiotics
- **Regularization**: Ridge-type or PCA pre-whitening for sparse features
- **Validation**: Fit extractors only on training folds (no leakage)

**Key Parameters**: `n_components`, `gamma` (RBF), `max_iter`

---

### 10. Homework_mnist_red_students.ipynb
**Topics**: Dimensionality Reduction for Classification
**Methods**:
- StandardScaler + PCA + Classifier pipeline
- GridSearchCV for component selection
- K-NN for evaluating reduced features

**AMR Application**:
- Template pipeline: `StandardScaler → PCA → LightGBM/SVM`
- Scree plot to find optimal compression (aim for 80-95% variance)
- Use PCA components as inputs to reduce parameter count

**Key Parameters**: `n_components`, CV strategy

---

### 11. Prob_PCA.ipynb
**Topics**: Probabilistic PCA, Generative Modeling, Reconstruction Error
**Methods**:
- FactorAnalysis for PPCA
- Eigen-decomposition vs SVD
- Generative sampling from latent space
- Reconstruction error curves
- Scree plots for component selection

**AMR Application**:
- **Noise reduction**: Discard low-eigenvalue components (noise in MALDI spectra)
- **Feature engineering**: PCA components as inputs to GBM/MLP
- **Visualization**: Project to 2D/3D to check species clustering
- **Generative**: Sample synthetic MALDI spectra for data augmentation

**Key Parameters**: `n_components`, cumulative variance threshold (80-95%)

---

### 12. Bayesian_regression_professor.ipynb
**Topics**: Bayesian Linear Regression, Kernel Bayes, Evidence Maximization
**Methods**:
- Maximum Likelihood vs Bayesian estimation
- Gaussian prior + Gaussian likelihood = Gaussian posterior
- Predictive distribution (mean + variance)
- Kernel trick for non-linear Bayesian regression
- PyTorch optimization for evidence maximization

**AMR Application**:
- **Uncertainty quantification**: High variance = OOD (species shift)
- **Small data**: Prior prevents overfitting (equivalent to L2 regularization)
- **Kernel methods**: RBF kernels capture non-linear MALDI patterns
- **Evidence maximization**: Learn regularization strength from data

**Key Parameters**: `sigma_n` (noise), `Sigma_p` (prior covariance), `gamma` (kernel)

---

### 13. Bayesian_regression_student.ipynb
**Topics**: Gaussian Theory, Bayesian Updating, Non-linear Bayes
**Methods**:
- Multivariate Gaussians, marginals, conditionals
- Sequential Bayesian updating
- Kernel predictive distribution
- PyTorch MarginalLikelihood module

**AMR Application**:
- **Sequential learning**: Update model as new data arrives
- **Uncertainty signaling**: Flag low-confidence predictions on shifted species
- **Evidence framework**: Alternative to CV for hyperparameter tuning

**Key Parameters**: Same as professor version

---

## Recommended Toolkit for AMR Competition

### Must-Have Libraries
```python
# Core ML
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Bayesian (optional)
import GPy

# Gradient Boosting (baseline)
import lightgbm as lgb
```

### Pipeline Template from Course
```python
# Leakage-safe feature extraction + classification
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('clf', GradientBoostingClassifier())
])

# Use within cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

---

## Semi-Supervised Strategies (from course methods)

### For Missing Labels (Amox/Clav 42.8% missing)
1. **K-Means Clustering**: Group similar samples, propagate labels within clusters
2. **PLS/CCA**: Use labeled samples to learn feature-label relationships, predict missing
3. **Kernel Methods**: Use kernel similarity to weighted-average labels from neighbors

### For Species Distribution Shift
1. **One-Class SVM**: Train on each species, detect OOD test samples
2. **Sample Reweighting**: Down-weight overrepresented species (P. aeruginosa)
3. **Bayesian Uncertainty**: Use predictive variance to detect unreliable predictions

---

## Dimensionality Reduction Strategy

| Method | When to Use | AMR Application |
|--------|-------------|-----------------|
| **PCA** | Baseline, unsupervised | Reduce 6000→200-500, noise removal |
| **PLS** | Have labels, want predictive features | Find resistance-specific MALDI peaks |
| **CCA** | Multi-label, maximize correlation | One CCA per antibiotic or multi-output CCA |
| **KPCA** | Non-linear relationships | Capture complex spectral patterns |
| **Sparse GPs with ARD** | Feature selection + uncertainty | Learn which peaks matter per antibiotic |

---

## Key Takeaways for AMR Competition

### From EDA + Course Methods:

1. **Use LightGBM/GBM as baseline** (from Ensembles notebook)
   - Bagging for variance reduction
   - `max_features` sampling to increase diversity
   - OOB score for validation

2. **Supervised dimensionality reduction** (from Feature Extraction notebooks)
   - PLS > PCA for predictive tasks
   - CCA for multi-label (8 antibiotics)
   - Leakage-safe pipelines (fit extractors in CV folds only)

3. **Handle species shift** (from Novelty Detection + Bayesian notebooks)
   - One-Class SVM per species
   - Bayesian uncertainty for OOD detection
   - Sample reweighting (0.3x for P. aeruginosa)

4. **Semi-supervised labeling** (from Clustering + Kernel Methods)
   - K-Means to group samples
   - Kernel similarity to propagate labels
   - PLS regression to predict missing labels

5. **Feature selection** (from GP + Feature Engineering)
   - ARD to identify important MALDI peaks
   - Bagging feature importance (robust)
   - Scree plots to determine PCA components

---

## Parameters to Tune (Priority Order)

### High Priority
1. **`n_components`** (PCA/PLS/CCA): Critical for 6000→reduced features
2. **`C`** (SVM) or `learning_rate`** (GBM): Regularization strength
3. **`max_depth`** (GBM trees): Control base learner complexity
4. **`nu`** (OneClassSVM): Outlier tolerance for OOD detection

### Medium Priority
5. **`gamma`** (RBF kernels): Controls non-linearity
6. **`max_features`** (Bagging): Diversity for ensemble
7. **`n_estimators`** (Ensembles): More = better until diminishing returns

### Low Priority (if using Bayesian methods)
8. **`length_scale`** (GP kernels): Smoothness
9. **`noise_level`** (GP): Data uncertainty
10. **`num_inducing`** (Sparse GP): Scalability

---

## References in Course

| Concept | Notebook | Section |
|---------|----------|---------|
| Species shift handling | NoveltyDetection_1SVM | OCSVM pipeline |
| Multi-label classification | Feature_extraction_professor | label_binarize |
| Small data regularization | Bayesian_regression_* | Prior as L2 |
| Semi-supervised learning | Clustering_Kmeans | Cluster-based labeling |
| Feature selection | GP_withGPy_v2 | ARD section |
| Ensemble methods | Introduction_Ensembles_students | Bagging/Boosting |
| Leakage prevention | FE_homework_FashionMNIST | GridSearchCV pipeline |
| Dimensionality reduction | Prob_PCA, Feature_extraction | PCA/PLS/CCA |
| Uncertainty quantification | GP_regression_professor | Confidence intervals |
| Kernel methods | KernelMethods_student | KLR/KRR |

---

**Last Updated**: 2026-01-07 (Course notebook analysis complete)
