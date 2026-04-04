import pandas as pd
import numpy as np
import re

def extract_features(users_df, posts_df):
    """
    V2: Enhanced feature extraction with all 7 agent improvements.
    """
    print("Extracting features (V2)...")
    
    # --- 1. Basic User Profile Features ---
    users_df['username_length'] = users_df['username'].str.len().fillna(0)
    users_df['name_length'] = users_df['name'].str.len().fillna(0)
    users_df['bio_length'] = users_df['description'].str.len().fillna(0)
    users_df['has_location'] = users_df['location'].apply(
        lambda x: 1 if x and str(x).strip() and str(x).lower() not in ['idk', 'nan', 'none', ''] else 0
    )
    
    # --- IMPROVEMENT #1: Interaction features with z_score ---
    users_df['abs_z_score'] = users_df['z_score'].abs()
    users_df['z_score_x_tweet_count'] = users_df['z_score'] * users_df['tweet_count']
    
    # --- IMPROVEMENT #2: Normalization per dataset ---
    if 'dataset_avg_posts' in users_df.columns:
        users_df['post_count_vs_dataset_avg'] = users_df['tweet_count'] / users_df['dataset_avg_posts'].replace(0, 1)
    else:
        users_df['post_count_vs_dataset_avg'] = 0
    
    # --- 2. Behavioral Features (from Posts) ---
    posts_df = posts_df.copy()
    posts_df['created_at'] = pd.to_datetime(posts_df['created_at'])
    posts_df = posts_df.sort_values(['author_id', 'created_at'])
    posts_df['time_diff'] = posts_df.groupby('author_id')['created_at'].diff().dt.total_seconds()
    posts_df['hour'] = posts_df['created_at'].dt.hour
    posts_df['text_len'] = posts_df['text'].str.len()
    
    # Basic aggregations
    behavior_stats = posts_df.groupby('author_id').agg(
        avg_post_length=('text_len', 'mean'),
        link_ratio=('text', lambda x: x.str.contains('https://t.co', na=False).mean()),
        hashtag_ratio=('text', lambda x: x.str.contains('#', na=False).mean()),
        mention_ratio=('text', lambda x: x.str.contains('@', na=False).mean()),
        time_diff_std=('time_diff', 'std'),
        time_diff_mean=('time_diff', 'mean'),
    ).reset_index()
    
    # --- IMPROVEMENT #5: Advanced temporal features ---
    # Coefficient of Variation (regularity indicator)
    def safe_cv(group):
        m = group['time_diff'].mean()
        s = group['time_diff'].std()
        if pd.isna(m) or m == 0:
            return pd.Series({'time_diff_cv': 0, 'night_post_ratio': 0, 'burst_count': 0})
        cv = s / m if not pd.isna(s) else 0
        night = ((group['hour'] >= 2) & (group['hour'] <= 6)).mean()
        bursts = 0
        diffs = group['time_diff'].dropna().values
        streak = 0
        for d in diffs:
            if d < 60:
                streak += 1
                if streak >= 2:
                    bursts += 1
            else:
                streak = 0
        return pd.Series({'time_diff_cv': cv, 'night_post_ratio': night, 'burst_count': bursts})
    
    temporal_advanced = posts_df.groupby('author_id').apply(safe_cv, include_groups=False).reset_index()
    
    # --- IMPROVEMENT #3: Shared text detection (swarm bots) ---
    # Normalize text to better catch variants (lowercase, no accents, no punctuation)
    import unicodedata
    def normalize_text(t):
        if not isinstance(t, str): return ""
        t = unicodedata.normalize('NFKD', t.lower()).encode('ASCII', 'ignore').decode('utf-8')
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', t)).strip()
        
    posts_df['norm_text'] = posts_df['text'].apply(normalize_text)

    # Count how many different authors posted each normalized text
    text_author_counts = posts_df.groupby('norm_text')['author_id'].nunique().reset_index()
    text_author_counts.columns = ['norm_text', 'n_authors_with_text']
    posts_with_sharing = posts_df.merge(text_author_counts, on='norm_text', how='left')
    
    shared_stats = posts_with_sharing.groupby('author_id').agg(
        n_shared_texts=('n_authors_with_text', lambda x: (x > 1).sum()),
        max_text_sharing=('n_authors_with_text', 'max'),
        shared_text_ratio=('n_authors_with_text', lambda x: (x > 1).mean()),
    ).reset_index()
    
    # --- IMPROVEMENT #4: LLM paraphrase markers + emoji analysis ---
    paraphrase_patterns = [
        r"here's a revised",
        r"here is a revised",
        r"here's a slightly modified",
        r"here is a slightly modified",
        r"here's the revised version",
    ]
    pattern_regex = '|'.join(paraphrase_patterns)
    
    def nlp_features(group):
        texts = group['text'].astype(str)
        texts_lower = texts.str.lower()
        
        has_paraphrase = texts_lower.str.contains(pattern_regex, na=False).any()
        
        # Emoji ratio (count emoji characters)
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001f900-\U0001f9FF"
            "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
            flags=re.UNICODE
        )
        total_chars = texts.str.len().sum()
        emoji_chars = texts.apply(lambda t: len(emoji_pattern.findall(str(t)))).sum()
        emoji_ratio = emoji_chars / max(total_chars, 1)
        
        return pd.Series({
            'has_paraphrase_marker': int(has_paraphrase),
            'emoji_ratio': emoji_ratio,
        })
    
    nlp_stats = posts_df.groupby('author_id').apply(nlp_features, include_groups=False).reset_index()
    
    # Lexical Diversity
    def calculate_lexical_diversity(text_series):
        all_text = " ".join(text_series.astype(str)).lower()
        words = re.findall(r'\w+', all_text)
        if not words:
            return 0
        return len(set(words)) / len(words)
    
    lexical_div = posts_df.groupby('author_id')['text'].apply(calculate_lexical_diversity).reset_index()
    lexical_div.columns = ['author_id', 'lexical_diversity']
    
    # --- Merge everything ---
    features_df = users_df.merge(behavior_stats, left_on='id', right_on='author_id', how='left')
    features_df = features_df.merge(lexical_div, on='author_id', how='left')
    features_df = features_df.merge(temporal_advanced, on='author_id', how='left')
    features_df = features_df.merge(shared_stats, on='author_id', how='left')
    features_df = features_df.merge(nlp_stats, on='author_id', how='left')
    
    # Interaction that needs time_diff_std
    features_df['z_score_x_time_std'] = features_df['z_score'] * features_df['time_diff_std'].fillna(0)
    
    # Fill NaN
    features_df = features_df.fillna(0)
    
    # Select feature columns (drop metadata/ID columns)
    drop_cols = ['id', 'author_id', 'username', 'name', 'description', 
                 'location', 'is_bot', 'dataset_id', 'dataset_avg_posts']
    drop_cols = [c for c in drop_cols if c in features_df.columns]
    
    X = features_df.drop(columns=drop_cols)
    y = features_df['is_bot']
    
    print(f"  Features extracted: {X.shape[1]} columns")
    return X, y, features_df

if __name__ == "__main__":
    from data_loader import load_all_data
    u_df, p_df = load_all_data()
    X, y, full = extract_features(u_df, p_df)
    print("Features shape:", X.shape)
    print("Columns:", X.columns.tolist())
    print(X.head())
