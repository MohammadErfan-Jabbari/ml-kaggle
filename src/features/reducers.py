"""
Dimensionality reduction and feature selection transformers for Phase 1+.

This module contains sklearn-compatible transformers for the DR research pipeline.

Available Transformers:
- MultiTargetFeatureUnion: Union of top-k features across all 8 antibiotics using f_classif
- LGBImportanceSelector: Select top-k features by LightGBM importance
- MultiStageSelector: Multi-stage pipeline (Variance -> f_classif -> LGB)
- PerSpeciesPLS: Train separate PLS models per species (requires species as last feature)
- NMFScaler: NMF with min-shifting for stable non-negative DR across CV folds

Note: Many DR methods can be used directly from sklearn:
- TruncatedSVD: Native sparse support, no scaling needed
- PCA: Requires StandardScaler, dense data
- PLSRegression: Auto-detected by ControlledExperiment, uses fully-labeled rows
- PLSCanonical: Multi-target variant (use instead of CCA)
- KernelPCA: Cosine kernel recommended for sparse MALDI data
- NMF: Native non-negative, good for MALDI spectra
"""

import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import NMF


class MultiTargetFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Union of top-k features across all 8 antibiotics.

    For each antibiotic target (ignoring NaN labels), selects top-k features
    using the specified score function, then takes the union across all targets.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Scoring function for univariate feature selection.
        Must accept (X, y) and return (scores, pvalues).
    k : int, default=500
        Number of features to select per target.

    Attributes
    ----------
    selected_features_ : ndarray of shape (n_selected,)
        Indices of selected features (union across all targets).
    n_features_per_target_ : dict
        Number of features selected per target.
    """

    def __init__(self, score_func=f_classif, k=500):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        """
        Fit the feature selector on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values (may contain NaN).

        Returns
        -------
        self : object
            Fitted transformer.
        """
        n_samples, n_features = X.shape
        n_targets = y.shape[1] if y.ndim > 1 else 1

        selected = set()
        self.n_features_per_target_ = {}

        for col_idx in range(n_targets):
            # Get valid samples (non-NaN labels)
            if y.ndim > 1:
                target = y[:, col_idx]
            else:
                target = y

            mask = ~np.isnan(target)
            n_valid = mask.sum()

            if n_valid < 100:
                print(f"  WARNING: Target {col_idx} has only {n_valid} valid samples, skipping")
                continue

            # Select top-k features for this target
            k_actual = min(self.k, n_features)
            selector = SelectKBest(self.score_func, k=k_actual)
            selector.fit(X[mask], target[mask])

            # Get selected feature indices
            target_features = np.where(selector.get_support())[0]
            selected.update(target_features)
            self.n_features_per_target_[col_idx] = len(target_features)

        self.selected_features_ = np.array(sorted(selected))
        return self

    def transform(self, X):
        """
        Transform data by selecting the union of features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_selected)
            Transformed data.
        """
        return X[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        """Get feature names for transformed output."""
        if input_features is None:
            return np.array([f"feature_{i}" for i in self.selected_features_])
        return np.array(input_features)[self.selected_features_]


class LGBImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features by LightGBM importance (sum across all targets).

    Trains a LightGBM classifier for each target (ignoring NaN labels),
    accumulates feature importances, and selects the top-k.

    Parameters
    ----------
    k : int, default=500
        Number of features to select.
    importance_type : str, default='gain'
        Type of importance to use ('gain', 'split', or 'weight').
    n_estimators : int, default=100
        Number of boosting rounds for importance estimation.

    Attributes
    ----------
    selected_features_ : ndarray of shape (k,)
        Indices of selected features (sorted by index for consistent ordering).
    feature_importances_ : ndarray of shape (n_features,)
        Aggregated importance scores.
    """

    def __init__(self, k=500, importance_type='gain', n_estimators=100):
        self.k = k
        self.importance_type = importance_type
        self.n_estimators = n_estimators

    def fit(self, X, y):
        """
        Fit the feature selector on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values (may contain NaN).

        Returns
        -------
        self : object
            Fitted transformer.
        """
        n_samples, n_features = X.shape
        n_targets = y.shape[1] if y.ndim > 1 else 1

        importances = np.zeros(n_features)

        for col_idx in range(n_targets):
            # Get valid samples (non-NaN labels)
            if y.ndim > 1:
                target = y[:, col_idx]
            else:
                target = y

            mask = ~np.isnan(target)
            n_valid = mask.sum()

            if n_valid < 100:
                continue

            # Train LightGBM for importance estimation
            lgb_model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                importance_type=self.importance_type
            )
            lgb_model.fit(X[mask], target[mask].astype(int))
            importances += lgb_model.feature_importances_

        self.feature_importances_ = importances

        # Select top-k by importance (store indices sorted for consistent transform)
        top_indices = np.argsort(importances)[-self.k:]
        self.selected_features_ = np.sort(top_indices)

        return self

    def transform(self, X):
        """
        Transform data by selecting top-k features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : array-like of shape (n_samples, k)
            Transformed data.
        """
        return X[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        """Get feature names for transformed output."""
        if input_features is None:
            return np.array([f"feature_{i}" for i in self.selected_features_])
        return np.array(input_features)[self.selected_features_]


class MultiStageSelector(BaseEstimator, TransformerMixin):
    """
    Multi-stage feature selection: VarianceThreshold -> f_classif -> LGB importance.

    Three-stage pipeline that progressively filters features:
    1. Remove low-variance features
    2. Univariate selection (f_classif) with union across targets
    3. LightGBM-based importance selection

    Parameters
    ----------
    var_threshold : float, default=0.001
        Variance threshold for first stage.
    univariate_k : int, default=1000
        Number of features to select per target in second stage.
    lgb_k : int, default=500
        Number of features to select in final stage.

    Attributes
    ----------
    var_selector_ : VarianceThreshold
        First stage selector.
    univariate_selector_ : MultiTargetFeatureUnion
        Second stage selector.
    lgb_selector_ : LGBImportanceSelector
        Third stage selector.
    n_features_stage1_ : int
        Number of features after variance threshold.
    n_features_stage2_ : int
        Number of features after univariate selection.
    n_features_final_ : int
        Final number of features.
    """

    def __init__(self, var_threshold=0.001, univariate_k=1000, lgb_k=500):
        self.var_threshold = var_threshold
        self.univariate_k = univariate_k
        self.lgb_k = lgb_k

    def fit(self, X, y):
        """
        Fit the multi-stage selector on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values (may contain NaN).

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Stage 1: Variance threshold
        self.var_selector_ = VarianceThreshold(threshold=self.var_threshold)
        X1 = self.var_selector_.fit_transform(X)
        self.n_features_stage1_ = X1.shape[1]

        # Stage 2: f_classif union across targets
        self.univariate_selector_ = MultiTargetFeatureUnion(f_classif, k=self.univariate_k)
        X2 = self.univariate_selector_.fit_transform(X1, y)
        self.n_features_stage2_ = X2.shape[1]

        # Stage 3: LGB importance
        # Ensure we don't try to select more than we have
        actual_lgb_k = min(self.lgb_k, X2.shape[1])
        self.lgb_selector_ = LGBImportanceSelector(k=actual_lgb_k)
        self.lgb_selector_.fit(X2, y)
        self.n_features_final_ = actual_lgb_k

        return self

    def transform(self, X):
        """
        Transform data through all three stages.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features_final)
            Transformed data.
        """
        X1 = self.var_selector_.transform(X)
        X2 = self.univariate_selector_.transform(X1)
        return self.lgb_selector_.transform(X2)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'var_threshold': self.var_threshold,
            'univariate_k': self.univariate_k,
            'lgb_k': self.lgb_k
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class PerSpeciesPLS(BaseEstimator, TransformerMixin):
    """
    Train separate PLSRegression models per species.

    For each species, fits a PLSRegression on only that species' data.
    At transform time, routes each sample to its species-specific model.

    This transformer requires that X has species_id as the LAST column.
    During fit/transform, species is extracted from X[:, -1] and the
    actual features are X[:, :-1].

    Parameters
    ----------
    n_components : int, default=20
        Number of PLS components (same for all species).
    min_samples : int, default=20
        Minimum fully-labeled samples required to train species-specific PLS.

    Attributes
    ----------
    pls_models_ : dict
        {species_id: PLSRegression} fitted models.
    n_features_in_ : int
        Number of features seen during fit (including species column).
    species_stats_ : dict
        Statistics about samples used per species for debugging.

    Notes
    -----
    - PLSRegression requires fully-labeled y (no NaNs in any target column).
    - Species with insufficient fully-labeled samples are skipped.
    - During transform, samples from species with no model get zeros.
    """

    def __init__(self, n_components=20, min_samples=20):
        self.n_components = n_components
        self.min_samples = min_samples
        self.pls_models_ = {}
        self.species_stats_ = {}

    def fit(self, X, y):
        """
        Fit per-species PLS models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features + 1)
            Features with species_id as the LAST column.
        y : array-like of shape (n_samples, n_targets)
            Target values (may contain NaN).

        Returns
        -------
        self : object
            Fitted transformer.
        """
        from sklearn.cross_decomposition import PLSRegression

        # Extract species from last column
        species = X[:, -1].astype(int)
        X_features = X[:, :-1]

        self.n_features_in_ = X.shape[1]

        # Fit PLS for each species
        for species_id in [0, 1, 2, 3]:
            species_mask = (species == species_id)
            n_samples = species_mask.sum()

            if n_samples < self.min_samples:
                self.species_stats_[species_id] = {
                    'n_total': n_samples,
                    'n_fully_labeled': 0,
                    'trained': False,
                    'reason': 'Insufficient total samples'
                }
                continue

            # Get fully-labeled subset (PLSRegression requires complete y)
            y_sp = y[species_mask]
            full_label_mask = ~np.isnan(y_sp).any(axis=1)

            n_fully_labeled = full_label_mask.sum()
            if n_fully_labeled < self.min_samples:
                self.species_stats_[species_id] = {
                    'n_total': n_samples,
                    'n_fully_labeled': n_fully_labeled,
                    'trained': False,
                    'reason': 'Insufficient fully-labeled samples'
                }
                continue

            # Fit species-specific PLS
            pls = PLSRegression(n_components=self.n_components)
            try:
                pls.fit(
                    X_features[species_mask][full_label_mask],
                    y[species_mask][full_label_mask]
                )
                self.pls_models_[species_id] = pls
                self.species_stats_[species_id] = {
                    'n_total': n_samples,
                    'n_fully_labeled': n_fully_labeled,
                    'n_components': self.n_components,
                    'trained': True
                }
            except Exception as e:
                self.species_stats_[species_id] = {
                    'n_total': n_samples,
                    'n_fully_labeled': n_fully_labeled,
                    'trained': False,
                    'reason': f'Fit failed: {e}'
                }

        return self

    def transform(self, X):
        """
        Transform X using per-species PLS models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features + 1)
            Features with species_id as the LAST column.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Transformed data.
        """
        X_features = X[:, :-1]
        species = X[:, -1].astype(int)

        # Initialize output
        X_transformed = np.zeros((len(X), self.n_components))

        # Transform each species with its model
        for species_id, pls in self.pls_models_.items():
            species_mask = (species == species_id)
            if species_mask.sum() > 0:
                X_transformed[species_mask] = pls.transform(X_features[species_mask])

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array([f"pls_component_{i}" for i in range(self.n_components)])


class NMFScaler(BaseEstimator, TransformerMixin):
    """
    Non-negative Matrix Factorization with min-shifting for stable CV.

    NMF requires non-negative input (X >= 0). Standard scaling approaches
    like MinMaxScaler cause issues across CV folds because different folds
    have different min/max ranges, making NMF components non-comparable.

    This transformer uses min-shifting (X - X.min(axis=0)) instead of scaling:
    - Deterministic (same shift for all folds)
    - Stable across CV folds
    - NMF only cares about non-negativity, not absolute scale

    Parameters
    ----------
    n_components : int, default=50
        Number of NMF components to extract.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    nmf : NMF
        Fitted sklearn NMF transformer.
    shift_ : ndarray of shape (n_features,)
        Feature minimums stored during fit for stable transform.
    n_components : int
        Number of components (for get_feature_names_out).

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ('nmf', NMFScaler(n_components=50)),
    ...     ('pls', PLSRegression(n_components=20))
    ... ])
    """

    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.nmf = NMF(n_components=n_components, random_state=random_state)
        self.shift_ = None

    def fit(self, X, y=None):
        """
        Fit NMF on min-shifted data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for sklearn compatibility.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Store minimums for each feature
        self.shift_ = X.min(axis=0)
        # Shift to non-negative space (ensure X >= 0)
        X_shifted = X - self.shift_
        # Fit NMF on shifted data
        self.nmf.fit(X_shifted)
        return self

    def transform(self, X):
        """
        Transform X using fitted NMF with same min-shift.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data (NMF components).
        """
        # Apply same shift as fit
        X_shifted = X - self.shift_
        return self.nmf.transform(X_shifted)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array([f"nmf_component_{i}" for i in range(self.n_components)])

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'n_components': self.n_components, 'random_state': self.random_state}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
