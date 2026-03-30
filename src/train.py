import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from data_loader import load_all_data
from features import extract_features

def get_ensemble_models():
    """Returns a dictionary of uninitialized base models for the ensemble."""
    return {
        'xgb': xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1  # Suppress LightGBM output
        ),
        'rf': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }

def fit_ensemble(models_dict, X, y):
    """Fits all models in the dictionary."""
    fitted_models = {}
    for name, model in models_dict.items():
        model.fit(X, y)
        fitted_models[name] = model
    return fitted_models

def predict_proba_ensemble(fitted_models, X):
    """Returns the average predicted probability (Soft Voting) across all models."""
    probs = []
    for model in fitted_models.values():
        probs.append(model.predict_proba(X)[:, 1])
    # Average the probabilities across columns (models)
    return np.mean(np.column_stack(probs), axis=1)

def train_model():
    """
    V3: Trains a Soft Voting Ensemble (XGBoost + LightGBM + Random Forest)
    with cross-validated threshold (#6) and full retrain (#7).
    """
    # 1. Load and Prepare
    users_df, posts_df = load_all_data()
    X, y, _ = extract_features(users_df, posts_df)
    
    print(f"\nFeatures: {X.columns.tolist()}")
    print(f"Shape: {X.shape}")
    
    # --- IMPROVEMENT #6: Cross-Validated Threshold via Soft Voting Ensemble ---
    print("\n=== Cross-Validation (5-Fold) for Robust Threshold (Ensemble) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_thresholds = []
    fold_results = []
    
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize and fit models for this fold
        models_dict = get_ensemble_models()
        fitted_models = fit_ensemble(models_dict, X_train, y_train)
        
        # Get ensemble soft voting probabilities
        y_probs = predict_proba_ensemble(fitted_models, X_test)
        
        # Find minimum threshold for zero FP in this fold
        best_t = 0.99
        for t in np.arange(0.5, 1.0, 0.01):
            y_pred_t = (y_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
            if fp == 0 and tp > 0:
                best_t = t
                break
        
        # Get stats at this threshold
        y_pred_final = (y_probs >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
        
        fold_thresholds.append(best_t)
        fold_results.append({'fold': fold_i+1, 'threshold': best_t, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn})
        print(f"  Fold {fold_i+1}: Threshold={best_t:.2f} | TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Take the MAX threshold (most conservative = safest)
    safe_threshold = max(fold_thresholds)
    print(f"\n  >>> Safe Ensemble Threshold (max across folds): {safe_threshold:.2f}")
    
    # --- IMPROVEMENT #7: Retrain on 100% of data ---
    print("\n=== Retraining Ensemble on 100% of data ===")
    final_models_dict = get_ensemble_models()
    final_fitted_models = fit_ensemble(final_models_dict, X, y)
    
    # Verify on training data (sanity check)
    y_probs_all = predict_proba_ensemble(final_fitted_models, X)
    y_pred_all = (y_probs_all >= safe_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_all).ravel()
    print(f"  Full training set at threshold {safe_threshold:.2f}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Standard eval for comparison
    y_pred_05 = (y_probs_all >= 0.5).astype(int)
    print("\n--- Full Data Eval (Ensemble Threshold 0.5, for reference) ---")
    print(classification_report(y, y_pred_05))
    
    # Save models and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_fitted_models, 'models/bot_detector_ensemble.pkl')
    joblib.dump({
        'threshold': safe_threshold,
        'features': X.columns.tolist(),
        'cv_results': fold_results,
        'model_type': 'soft_voting_ensemble'
    }, 'models/metadata.pkl')
    print(f"\nEnsemble models saved. Threshold: {safe_threshold:.2f}")

if __name__ == "__main__":
    train_model()
