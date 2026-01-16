"""
AMR Prediction: Round 3 Advanced Experiments
Focus: Course methods, KNN, multi-seed ensemble, aggressive ensembles
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from datetime import datetime
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Constants
ANTIBIOTICS = [
    'Ampicillin', 'Levofloxacin', 'Ciprofloxacin', 'Imipenem',
    'Amoxicillin_Clavulanic_acid', 'Ertapenem', 'Cefotaxime', 'Cefuroxime'
]
N_FOLDS = 5
RANDOM_SEED = 42


def load_data():
    """Load and prepare data."""
    train = pd.read_csv('/sata_disk/users/erfan/ml_kaggle/raw/train.csv')
    test = pd.read_csv('/sata_disk/users/erfan/ml_kaggle/raw/test.csv')

    # Features are maldi_feature_0 through maldi_feature_5999
    feature_cols = [c for c in train.columns if c.startswith('maldi_feature_')]
    X_train = train[feature_cols].values
    y_train = train[ANTIBIOTICS].values  # 8 targets
    species_train = train['species_id'].values

    X_test = test[feature_cols].values
    species_test = test['species_id'].values

    # Remove zero-variance columns
    var = X_train.var(axis=0)
    keep = var > 0
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"After filtering: Train {X_train.shape}")

    return X_train, y_train, species_train, X_test, species_test


def get_species_weights(species_train):
    """Calculate extreme species weights to fix distribution shift."""
    # P.aeruginosa: 0.05x, K.pneumoniae: 3x, others: 1.5x
    weights = np.ones(len(species_train))
    weights[species_train == 3] = 0.05  # P.aeruginosa
    weights[species_train == 1] = 3.0   # K.pneumoniae
    weights[species_train == 0] = 1.5   # E.coli
    weights[species_train == 2] = 1.5   # P.mirabilis
    return weights


def compute_species_auc(y_true, y_pred, species, ab_idx=0):
    """Compute AUC for each species."""
    species_aucs = {}
    names = ['E.coli', 'K.pneumoniae', 'P.mirabilis', 'P.aeruginosa']

    for sp in range(4):
        mask = (species == sp) & ~np.isnan(y_true[:, ab_idx])
        if mask.sum() > 10:
            try:
                auc = roc_auc_score(y_true[mask, ab_idx], y_pred[mask, ab_idx])
                species_aucs[names[sp]] = auc
            except:
                species_aucs[names[sp]] = np.nan

    return species_aucs


def experiment_knn_blend(X_train, y_train, species_train, X_test, species_test):
    """
    KNN-based predictions blended with LightGBM.
    Use K-NN in PLS space for species-aware similarity.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: KNN Blend")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros((len(X_train), 8))
    test_preds_knn = np.zeros((len(X_test), 8))
    test_preds_lgb = np.zeros((len(X_test), 8))
    weights = get_species_weights(species_train)

    for ab_idx, ab in enumerate(ANTIBIOTICS):
        print(f"  [{ab_idx+1}/8] {ab}")

        mask = ~np.isnan(y_train[:, ab_idx])
        X_ab, y_ab = X_train[mask], y_train[mask, ab_idx]
        sp_ab = species_train[mask]
        w_ab = weights[mask]
        mask_indices = np.where(mask)[0]

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_ab)
        X_test_scaled = scaler.transform(X_test)

        # PCA for KNN (reduce dimensionality)
        pca = PCA(n_components=100)
        X_pca = pca.fit_transform(X_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        fold_test_knn = []
        fold_test_lgb = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_ab, sp_ab)):
            # KNN model
            knn = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean')
            knn.fit(X_pca[train_idx], y_ab[train_idx].astype(int))

            val_orig_idx = mask_indices[val_idx]
            oof_preds[val_orig_idx, ab_idx] = knn.predict_proba(X_pca[val_idx])[:, 1] if len(knn.classes_) > 1 else np.zeros(len(val_idx))
            fold_test_knn.append(knn.predict_proba(X_test_pca)[:, 1] if len(knn.classes_) > 1 else np.zeros(len(X_test)))

            # LightGBM model
            model = lgb.LGBMClassifier(
                n_estimators=300, num_leaves=63, learning_rate=0.02,
                min_child_samples=20, random_state=RANDOM_SEED + fold,
                verbose=-1
            )
            model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=w_ab[train_idx])
            fold_test_lgb.append(model.predict_proba(X_ab[val_idx])[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_ab[val_idx]))

        test_preds_knn[:, ab_idx] = np.mean(fold_test_knn, axis=0)
        test_preds_lgb[:, ab_idx] = np.mean(fold_test_lgb, axis=0)

    # Blend KNN + LGB (30% KNN, 70% LGB)
    test_preds = 0.3 * test_preds_knn + 0.7 * test_preds_lgb

    return oof_preds, test_preds


def experiment_multi_seed(X_train, y_train, species_train, X_test, species_test, n_seeds=5):
    """
    Multi-seed ensembling: Train same architecture with different seeds.
    Reduces variance and improves generalization.
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT: Multi-Seed Ensemble ({n_seeds} seeds)")
    print("="*60)

    all_oof_preds = []
    all_test_preds = []
    weights = get_species_weights(species_train)

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED + seed * 100)

        oof = np.zeros((len(X_train), 8))
        test_preds = np.zeros((len(X_test), 8))

        for ab_idx, ab in enumerate(ANTIBIOTICS):
            mask = ~np.isnan(y_train[:, ab_idx])
            X_ab, y_ab = X_train[mask], y_train[mask, ab_idx]
            sp_ab = species_train[mask]
            w_ab = weights[mask]

            # Track positions in original array
            mask_indices = np.where(mask)[0]

            fold_test = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_ab, sp_ab)):
                model = lgb.LGBMClassifier(
                    n_estimators=350, num_leaves=127, learning_rate=0.02,
                    min_child_samples=25, random_state=seed * 1000 + fold,
                    verbose=-1, reg_alpha=0.1, reg_lambda=0.1
                )
                model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=w_ab[train_idx])

                # Use mask_indices to properly assign to oof
                val_orig_idx = mask_indices[val_idx]
                oof[val_orig_idx, ab_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
                fold_test.append(model.predict_proba(X_test)[:, 1])

            test_preds[:, ab_idx] = np.mean(fold_test, axis=0)

        all_oof_preds.append(oof)
        all_test_preds.append(test_preds)

    # Average across seeds
    final_oof = np.mean(all_oof_preds, axis=0)
    final_test = np.mean(all_test_preds, axis=0)

    return final_oof, final_test


def experiment_pls_knn(X_train, y_train, species_train, X_test, species_test):
    """
    PLS features + KNN: Use supervised dimensionality reduction with KNN.
    From course: PLS captures resistance-specific patterns.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: PLS + KNN")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros((len(X_train), 8))
    test_preds = np.zeros((len(X_test), 8))

    for ab_idx, ab in enumerate(ANTIBIOTICS):
        print(f"  [{ab_idx+1}/8] {ab}")

        mask = ~np.isnan(y_train[:, ab_idx])
        X_ab, y_ab = X_train[mask], y_train[mask, ab_idx]
        sp_ab = species_train[mask]
        mask_indices = np.where(mask)[0]

        fold_test = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_ab, sp_ab)):
            # PLS on training fold only (no leakage)
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_ab[train_idx])
            X_val_fold = scaler.transform(X_ab[val_idx])
            X_test_scaled = scaler.transform(X_test)

            pls = PLSRegression(n_components=50)
            pls.fit(X_train_fold, y_ab[train_idx])

            X_train_pls = pls.transform(X_train_fold)
            X_val_pls = pls.transform(X_val_fold)
            X_test_pls = pls.transform(X_test_scaled)

            # KNN in PLS space
            knn = KNeighborsClassifier(n_neighbors=11, weights='distance')
            knn.fit(X_train_pls, y_ab[train_idx].astype(int))

            val_orig_idx = mask_indices[val_idx]
            if len(knn.classes_) > 1:
                oof_preds[val_orig_idx, ab_idx] = knn.predict_proba(X_val_pls)[:, 1]
                fold_test.append(knn.predict_proba(X_test_pls)[:, 1])
            else:
                oof_preds[val_orig_idx, ab_idx] = 0.0
                fold_test.append(np.zeros(len(X_test)))

        test_preds[:, ab_idx] = np.mean(fold_test, axis=0)

    return oof_preds, test_preds


def experiment_svm_ensemble(X_train, y_train, species_train, X_test, species_test):
    """
    SVM ensemble: Use SVM with RBF kernel (good for sparse data).
    From course: Kernel methods robust with 93% zeros.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: SVM Ensemble (RBF)")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros((len(X_train), 8))
    test_preds = np.zeros((len(X_test), 8))
    weights = get_species_weights(species_train)

    for ab_idx, ab in enumerate(ANTIBIOTICS):
        print(f"  [{ab_idx+1}/8] {ab}")

        mask = ~np.isnan(y_train[:, ab_idx])
        X_ab, y_ab = X_train[mask], y_train[mask, ab_idx]
        sp_ab = species_train[mask]
        w_ab = weights[mask]
        mask_indices = np.where(mask)[0]

        fold_test = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_ab, sp_ab)):
            # Scale data
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_ab[train_idx])
            X_val_fold = scaler.transform(X_ab[val_idx])
            X_test_scaled = scaler.transform(X_test)

            # PCA to speed up SVM
            pca = PCA(n_components=100)
            X_train_pca = pca.fit_transform(X_train_fold)
            X_val_pca = pca.transform(X_val_fold)
            X_test_pca = pca.transform(X_test_scaled)

            # SVM with RBF
            svm = SVC(C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_SEED)
            svm.fit(X_train_pca, y_ab[train_idx].astype(int))

            val_orig_idx = mask_indices[val_idx]
            oof_preds[val_orig_idx, ab_idx] = svm.predict_proba(X_val_pca)[:, 1]
            fold_test.append(svm.predict_proba(X_test_pca)[:, 1])

        test_preds[:, ab_idx] = np.mean(fold_test, axis=0)

    return oof_preds, test_preds


def experiment_extreme_ensemble(X_train, y_train, species_train, X_test, species_test):
    """
    Extreme ensemble: Combine multiple diverse models with aggressive K.pn weighting.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Extreme Ensemble (5 models)")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    weights = get_species_weights(species_train)

    # We'll train 5 different model types
    models_oof = {name: np.zeros((len(X_train), 8)) for name in ['lgb1', 'lgb2', 'lgb3', 'lgb4', 'lgb5']}
    models_test = {name: np.zeros((len(X_test), 8)) for name in ['lgb1', 'lgb2', 'lgb3', 'lgb4', 'lgb5']}

    model_configs = [
        {'n_estimators': 350, 'num_leaves': 127, 'learning_rate': 0.02, 'min_child_samples': 25},
        {'n_estimators': 400, 'num_leaves': 63, 'learning_rate': 0.03, 'min_child_samples': 15},
        {'n_estimators': 500, 'num_leaves': 31, 'learning_rate': 0.01, 'min_child_samples': 30},
        {'n_estimators': 300, 'num_leaves': 255, 'learning_rate': 0.025, 'min_child_samples': 20},
        {'n_estimators': 450, 'num_leaves': 95, 'learning_rate': 0.015, 'min_child_samples': 10},
    ]

    for ab_idx, ab in enumerate(ANTIBIOTICS):
        print(f"  [{ab_idx+1}/8] {ab}")

        mask = ~np.isnan(y_train[:, ab_idx])
        X_ab, y_ab = X_train[mask], y_train[mask, ab_idx]
        sp_ab = species_train[mask]
        w_ab = weights[mask]
        mask_indices = np.where(mask)[0]

        for model_idx, (name, config) in enumerate(zip(models_oof.keys(), model_configs)):
            fold_test = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_ab, sp_ab)):
                model = lgb.LGBMClassifier(
                    **config,
                    random_state=RANDOM_SEED + model_idx * 100 + fold,
                    verbose=-1,
                    reg_alpha=0.1,
                    reg_lambda=0.1
                )
                model.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=w_ab[train_idx])

                val_orig_idx = mask_indices[val_idx]
                models_oof[name][val_orig_idx, ab_idx] = model.predict_proba(X_ab[val_idx])[:, 1]
                fold_test.append(model.predict_proba(X_test)[:, 1])

            models_test[name][:, ab_idx] = np.mean(fold_test, axis=0)

    # Rank averaging across models
    print("  Rank averaging...")
    final_oof = np.zeros((len(X_train), 8))
    final_test = np.zeros((len(X_test), 8))

    for ab_idx in range(8):
        ranks_oof = np.zeros((len(X_train), len(models_oof)))
        ranks_test = np.zeros((len(X_test), len(models_test)))

        for i, name in enumerate(models_oof.keys()):
            ranks_oof[:, i] = models_oof[name][:, ab_idx].argsort().argsort()
            ranks_test[:, i] = models_test[name][:, ab_idx].argsort().argsort()

        final_oof[:, ab_idx] = ranks_oof.mean(axis=1) / len(X_train)
        final_test[:, ab_idx] = ranks_test.mean(axis=1) / len(X_test)

    return final_oof, final_test


def experiment_species_weighted_knn(X_train, y_train, species_train, X_test, species_test):
    """
    Species-weighted KNN: Use species information in prediction.
    Weight neighbors by species similarity.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Species-Weighted KNN")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros((len(X_train), 8))
    test_preds = np.zeros((len(X_test), 8))

    for ab_idx, ab in enumerate(ANTIBIOTICS):
        print(f"  [{ab_idx+1}/8] {ab}")

        mask = ~np.isnan(y_train[:, ab_idx])
        X_ab, y_ab = X_train[mask], y_train[mask, ab_idx]
        sp_ab = species_train[mask]
        mask_indices = np.where(mask)[0]

        fold_test = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_ab, sp_ab)):
            # Scale + PCA
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_ab[train_idx])
            X_val_fold = scaler.transform(X_ab[val_idx])
            X_test_scaled = scaler.transform(X_test)

            pca = PCA(n_components=50)
            X_train_pca = pca.fit_transform(X_train_fold)
            X_val_pca = pca.transform(X_val_fold)
            X_test_pca = pca.transform(X_test_scaled)

            # Add species as feature (scaled)
            sp_train = sp_ab[train_idx].reshape(-1, 1) / 3.0
            sp_val = sp_ab[val_idx].reshape(-1, 1) / 3.0
            sp_test_scaled = species_test.reshape(-1, 1) / 3.0

            X_train_sp = np.hstack([X_train_pca, sp_train * 5])  # Weight species heavily
            X_val_sp = np.hstack([X_val_pca, sp_val * 5])
            X_test_sp = np.hstack([X_test_pca, sp_test_scaled * 5])

            # KNN
            knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
            knn.fit(X_train_sp, y_ab[train_idx].astype(int))

            val_orig_idx = mask_indices[val_idx]
            if len(knn.classes_) > 1:
                oof_preds[val_orig_idx, ab_idx] = knn.predict_proba(X_val_sp)[:, 1]
                fold_test.append(knn.predict_proba(X_test_sp)[:, 1])
            else:
                oof_preds[val_orig_idx, ab_idx] = 0.0
                fold_test.append(np.zeros(len(X_test)))

        test_preds[:, ab_idx] = np.mean(fold_test, axis=0)

    return oof_preds, test_preds


def mega_ensemble_v2(X_train, y_train, species_train, X_test, species_test):
    """
    Mega ensemble v2: Combine best methods from R1-R3 using rank averaging.
    """
    print("\n" + "="*60)
    print("MEGA ENSEMBLE V2")
    print("="*60)

    results = {}

    # 1. Multi-seed LGB
    print("\n  [1/4] Multi-seed ensemble...")
    oof1, test1 = experiment_multi_seed(X_train, y_train, species_train, X_test, species_test, n_seeds=3)
    results['multi_seed'] = {'oof': oof1, 'test': test1}

    # 2. Extreme ensemble (5 LGB variants)
    print("\n  [2/4] Extreme ensemble...")
    oof2, test2 = experiment_extreme_ensemble(X_train, y_train, species_train, X_test, species_test)
    results['extreme'] = {'oof': oof2, 'test': test2}

    # 3. PLS-KNN
    print("\n  [3/4] PLS + KNN...")
    oof3, test3 = experiment_pls_knn(X_train, y_train, species_train, X_test, species_test)
    results['pls_knn'] = {'oof': oof3, 'test': test3}

    # 4. Species-global blend from previous best
    print("\n  [4/4] Species-global LGB...")
    weights = get_species_weights(species_train)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof4 = np.zeros((len(X_train), 8))
    test4 = np.zeros((len(X_test), 8))

    for ab_idx, ab in enumerate(ANTIBIOTICS):
        mask = ~np.isnan(y_train[:, ab_idx])
        X_ab, y_ab = X_train[mask], y_train[mask, ab_idx]
        sp_ab = species_train[mask]
        w_ab = weights[mask]
        mask_indices = np.where(mask)[0]

        fold_global = []
        fold_species = {sp: [] for sp in range(4)}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_ab, sp_ab)):
            # Global model
            model_g = lgb.LGBMClassifier(n_estimators=350, num_leaves=127, learning_rate=0.02,
                                        min_child_samples=25, verbose=-1, random_state=RANDOM_SEED + fold)
            model_g.fit(X_ab[train_idx], y_ab[train_idx], sample_weight=w_ab[train_idx])

            global_pred_val = model_g.predict_proba(X_ab[val_idx])[:, 1]
            fold_global.append(model_g.predict_proba(X_test)[:, 1])

            # Species-specific models
            sp_pred_val = np.zeros(len(val_idx))
            sp_pred_test = np.zeros(len(X_test))

            for sp in range(4):
                sp_mask_train = sp_ab[train_idx] == sp
                sp_mask_val = sp_ab[val_idx] == sp
                sp_mask_test = species_test == sp

                if sp_mask_train.sum() > 50:
                    model_sp = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.03,
                                                 min_child_samples=10, verbose=-1, random_state=RANDOM_SEED + fold)
                    model_sp.fit(X_ab[train_idx][sp_mask_train], y_ab[train_idx][sp_mask_train])

                    if sp_mask_val.sum() > 0:
                        sp_pred_val[sp_mask_val] = model_sp.predict_proba(X_ab[val_idx][sp_mask_val])[:, 1]
                    if sp_mask_test.sum() > 0:
                        sp_pred_test[sp_mask_test] = model_sp.predict_proba(X_test[sp_mask_test])[:, 1]
                else:
                    if sp_mask_val.sum() > 0:
                        sp_pred_val[sp_mask_val] = global_pred_val[sp_mask_val]
                    if sp_mask_test.sum() > 0:
                        sp_pred_test[sp_mask_test] = model_g.predict_proba(X_test[sp_mask_test])[:, 1]

            # Blend 60% global + 40% species
            blend_val = 0.6 * global_pred_val + 0.4 * sp_pred_val
            val_orig_idx = mask_indices[val_idx]
            oof4[val_orig_idx, ab_idx] = blend_val

            fold_species[0].append(sp_pred_test)

        # Average test predictions (using global only for simplicity)
        test4[:, ab_idx] = np.mean(fold_global, axis=0)

    results['species_global'] = {'oof': oof4, 'test': test4}

    # Rank averaging across all 4 methods
    print("\n  Final rank averaging...")
    final_oof = np.zeros((len(X_train), 8))
    final_test = np.zeros((len(X_test), 8))

    for ab_idx in range(8):
        ranks_oof = np.zeros((len(X_train), 4))
        ranks_test = np.zeros((len(X_test), 4))

        for i, name in enumerate(['multi_seed', 'extreme', 'pls_knn', 'species_global']):
            ranks_oof[:, i] = results[name]['oof'][:, ab_idx].argsort().argsort()
            ranks_test[:, i] = results[name]['test'][:, ab_idx].argsort().argsort()

        final_oof[:, ab_idx] = ranks_oof.mean(axis=1) / len(X_train)
        final_test[:, ab_idx] = ranks_test.mean(axis=1) / len(X_test)

    return final_oof, final_test, results


def evaluate(oof, y_true, species, method_name):
    """Evaluate predictions and print results."""
    per_ab = {}
    per_species = {}
    names = ['E.coli', 'K.pneumoniae', 'P.mirabilis', 'P.aeruginosa']

    for ab_idx, ab in enumerate(ANTIBIOTICS):
        mask = ~np.isnan(y_true[:, ab_idx])
        if mask.sum() > 0:
            try:
                auc = roc_auc_score(y_true[mask, ab_idx], oof[mask, ab_idx])
                per_ab[ab] = auc
            except:
                per_ab[ab] = np.nan

    # Species AUCs (average across antibiotics)
    for sp in range(4):
        sp_aucs = []
        for ab_idx in range(8):
            mask = (species == sp) & ~np.isnan(y_true[:, ab_idx])
            if mask.sum() > 10:
                try:
                    auc = roc_auc_score(y_true[mask, ab_idx], oof[mask, ab_idx])
                    sp_aucs.append(auc)
                except:
                    pass
        per_species[names[sp]] = np.mean(sp_aucs) if sp_aucs else np.nan

    mean_auc = np.mean([v for v in per_ab.values() if not np.isnan(v)])

    print(f"\n{'='*60}")
    print(f"Results: {method_name}")
    print(f"{'='*60}")
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"K.pneumoniae: {per_species['K.pneumoniae']:.4f}")

    return {
        'method': method_name,
        'mean_auc': mean_auc,
        'per_antibiotic': per_ab,
        'per_species': per_species,
        'k_pn_auc': per_species['K.pneumoniae']
    }


def save_submission(test_preds, species_test, suffix):
    """Save submission file."""
    test = pd.read_csv('/sata_disk/users/erfan/ml_kaggle/raw/test.csv')

    submission = pd.DataFrame({
        'sample_id': test['sample_id']
    })

    for i, ab in enumerate(ANTIBIOTICS):
        submission[ab] = test_preds[:, i]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"/sata_disk/users/erfan/ml_kaggle/outputs/submissions/sub_r3_{suffix}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    print(f"\nSubmission saved: {filename}")
    return filename


def run_round3():
    """Run Round 3 experiments."""
    print("\n" + "="*80)
    print("AMR Prediction: Round 3 Experiments")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[Loading Data]")
    X_train, y_train, species_train, X_test, species_test = load_data()

    all_results = []

    # Individual experiments
    experiments = [
        ("Multi-Seed", lambda: experiment_multi_seed(X_train, y_train, species_train, X_test, species_test, n_seeds=5)),
        ("PLS-KNN", lambda: experiment_pls_knn(X_train, y_train, species_train, X_test, species_test)),
        ("SVM-Ensemble", lambda: experiment_svm_ensemble(X_train, y_train, species_train, X_test, species_test)),
        ("Extreme-Ensemble", lambda: experiment_extreme_ensemble(X_train, y_train, species_train, X_test, species_test)),
        ("Species-Weighted-KNN", lambda: experiment_species_weighted_knn(X_train, y_train, species_train, X_test, species_test)),
    ]

    best_result = None
    best_test = None
    best_mean = 0

    for name, fn in experiments:
        try:
            oof, test = fn()
            result = evaluate(oof, y_train, species_train, name)
            result['test_preds'] = test
            all_results.append(result)

            if result['mean_auc'] > best_mean:
                best_mean = result['mean_auc']
                best_result = result
                best_test = test
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # Mega Ensemble V2
    print("\n" + "="*80)
    print("MEGA ENSEMBLE V2: Combining Best Methods")
    print("="*80)

    try:
        oof_mega, test_mega, _ = mega_ensemble_v2(X_train, y_train, species_train, X_test, species_test)
        result_mega = evaluate(oof_mega, y_train, species_train, "Mega-Ensemble-V2")
        result_mega['test_preds'] = test_mega
        all_results.append(result_mega)

        if result_mega['mean_auc'] > best_mean:
            best_mean = result_mega['mean_auc']
            best_result = result_mega
            best_test = test_mega
    except Exception as e:
        print(f"\nERROR in Mega Ensemble V2: {e}")
        import traceback
        traceback.print_exc()

    # Print comparison
    print("\n" + "="*80)
    print("ROUND 3 COMPARISON")
    print("="*80)
    print("\nRanked by Mean AUC (primary metric - matches leaderboard scoring):")
    print("-" * 60)
    print(f"{'Method':<30} {'Mean':>8} {'K.pn':>8} {'E.coli':>8}")
    print("-" * 60)

    sorted_results = sorted(all_results, key=lambda x: x['mean_auc'], reverse=True)
    for r in sorted_results:
        print(f"{r['method']:<30} {r['mean_auc']:>8.4f} {r['k_pn_auc']:>8.4f} {r['per_species'].get('E.coli', 0):>8.4f}")

    # Save best submission
    if best_test is not None:
        filename = save_submission(best_test, species_test, best_result['method'].lower().replace(' ', '_'))
        print(f"\n{'='*80}")
        print(f"BEST ROUND 3: {best_result['method']} (K.pn AUC: {best_kpn:.4f})")
        print(f"{'='*80}")
        print(f"Best submission: {filename}")

    # Save results
    results_file = f"/sata_disk/users/erfan/ml_kaggle/outputs/experiments/round3_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    save_results = [{k: v for k, v in r.items() if k != 'test_preds'} for r in all_results]
    for r in save_results:
        r['per_antibiotic'] = {k: float(v) for k, v in r['per_antibiotic'].items()}
        r['per_species'] = {k: float(v) if not np.isnan(v) else None for k, v in r['per_species'].items()}
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == "__main__":
    run_round3()
