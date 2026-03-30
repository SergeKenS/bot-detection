import json
import pandas as pd
import glob
import os

def load_all_data(data_dir='enoncer_complet'):
    """
    Loads all JSON datasets and bot labels from the specified directory.
    Returns:
        - users_df: DataFrame of all users with metadata + dataset context.
        - posts_df: DataFrame of all posts with dataset_id.
    """
    all_users = []
    all_posts = []
    
    # 1. Load Posts and Users from JSON files
    json_files = sorted(glob.glob(os.path.join(data_dir, 'dataset.posts&users.*.json')))
    print(f"Loading {len(json_files)} JSON files...")
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset_id = data.get('id', 0)
            meta = data.get('metadata', {})
            dataset_avg_posts = meta.get('users_average_amount_posts', 0)
            
            # Extract users with dataset context
            if 'users' in data:
                for u in data['users']:
                    u['dataset_id'] = dataset_id
                    u['dataset_avg_posts'] = dataset_avg_posts
                all_users.extend(data['users'])
            
            # Extract posts with dataset context
            if 'posts' in data:
                for p in data['posts']:
                    p['dataset_id'] = dataset_id
                all_posts.extend(data['posts'])
    
    users_df = pd.DataFrame(all_users)
    posts_df = pd.DataFrame(all_posts)
    
    # Remove duplicates (users might appear in multiple chunks)
    users_df = users_df.drop_duplicates(subset=['id']).reset_index(drop=True)
    
    # 2. Load Bot Labels
    bot_files = glob.glob(os.path.join(data_dir, 'dataset.bots.*.txt'))
    print(f"Loading {len(bot_files)} label files...")
    
    bot_ids = set()
    for file_path in bot_files:
        with open(file_path, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
            bot_ids.update(ids)
            
    # 3. Label users
    users_df['is_bot'] = users_df['id'].isin(bot_ids).astype(int)
    
    print(f"Total Users: {len(users_df)} (Bots: {users_df['is_bot'].sum()}, Humans: {len(users_df) - users_df['is_bot'].sum()})")
    print(f"Total Posts: {len(posts_df)}")
    
    return users_df, posts_df

if __name__ == "__main__":
    users, posts = load_all_data()
    print(users.columns.tolist())
    print(users[['id','dataset_id','dataset_avg_posts','tweet_count','z_score','is_bot']].head(10))
