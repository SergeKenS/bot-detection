import pandas as pd
import numpy as np
import json
import joblib
import os
import sys
from features import extract_features

def predict_proba_ensemble(fitted_models, X):
    """Returns the average predicted probability (Soft Voting) across all models."""
    probs = []
    for model in fitted_models.values():
        probs.append(model.predict_proba(X)[:, 1])
    # Average the probabilities across columns (models)
    return np.mean(np.column_stack(probs), axis=1)

def predict_bots(input_json_path, output_txt_path='submission.txt'):
    """
    V3: Loads a new JSON dataset, uses the Ensemble model to predict bots, outputs IDs.
    """
    # 1. Load Model and Metadata
    if not os.path.exists('models/bot_detector_ensemble.pkl'):
        print("Error: Ensemble Model not found. Please run src/train.py first.")
        return
    
    models = joblib.load('models/bot_detector_ensemble.pkl')
    meta = joblib.load('models/metadata.pkl')
    threshold = meta.get('threshold', 0.5)
    required_features = meta.get('features', [])
    
    # 2. Ingest New Data
    print(f"Loading input data: {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build users with dataset metadata
    dataset_id = data.get('id', 0)
    dataset_meta = data.get('metadata', {})
    dataset_avg_posts = dataset_meta.get('users_average_amount_posts', 0)
    
    users_list = data.get('users', [])
    for u in users_list:
        u['dataset_id'] = dataset_id
        u['dataset_avg_posts'] = dataset_avg_posts
    
    users_df = pd.DataFrame(users_list)
    posts_df = pd.DataFrame(data.get('posts', []))
    
    if users_df.empty:
        print("Error: No users found in input JSON.")
        return

    # 3. Feature Engineering
    users_df['is_bot'] = 0
    X, _, full_df = extract_features(users_df, posts_df)
    
    # Ensure columns match training, add missing cols as 0
    for col in required_features:
        if col not in X.columns:
            X[col] = 0
    X = X[required_features]
    
    # 4. Predict using Ensemble Soft Voting
    print(f"Predicting with Ensemble (Soft Voting), threshold {threshold:.2f}...")
    y_probs = predict_proba_ensemble(models, X)
    y_pred = (y_probs >= threshold).astype(int)
    
    print(f"Stats on probabilities: Mean={y_probs.mean():.4f}, Max={y_probs.max():.4f}, Min={y_probs.min():.4f}")
    
    # 5. Extract Bot IDs
    bot_ids = full_df.loc[y_pred == 1, 'id'].tolist()
    
    # 6. Save to txt
    with open(output_txt_path, 'w') as f:
        for b_id in bot_ids:
            f.write(f"{b_id}\n")
            
    print(f"Done! Found {len(bot_ids)} bots out of {len(users_df)} users.")
    print(f"Results saved to {output_txt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/input.json [output.txt]")
    else:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else 'submission.txt'
        predict_bots(inp, out)
