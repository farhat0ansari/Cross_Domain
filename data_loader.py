import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def load_hotels(filepath):
    """Load deceptive opinion spam dataset (Hotels)"""
    df = pd.read_csv(filepath)
    # 'deceptive' column: truthful=real, deceptive=fake
    df = df[['text', 'deceptive']].copy()
    df.columns = ['text', 'label']
    df['label'] = df['label'].map({'truthful': 0, 'deceptive': 1})
    df = df.dropna()
    print(f"Hotels dataset loaded: {len(df)} reviews")
    print(f"  Fake: {df['label'].sum()}, Real: {(df['label']==0).sum()}")
    return df

def load_online_products(filepath):
    """Load fake reviews dataset (Online Products)"""
    df = pd.read_csv(filepath)
    # 'label' column: OR=real, CG=fake
    df = df[['text_', 'label']].copy()
    df.columns = ['text', 'label']
    df['label'] = df['label'].map({'OR': 0, 'CG': 1})
    df = df.dropna()
    # Use a balanced sample of 1600 reviews for efficiency
    fake = df[df['label']==1].sample(n=min(800, len(df[df['label']==1])), random_state=42)
    real = df[df['label']==0].sample(n=min(800, len(df[df['label']==0])), random_state=42)
    df = shuffle(pd.concat([fake, real]), random_state=42).reset_index(drop=True)
    print(f"Online Products dataset loaded: {len(df)} reviews")
    print(f"  Fake: {df['label'].sum()}, Real: {(df['label']==0).sum()}")
    return df

def load_yelp(filepath):
    """Load Yelp labelled dataset (Restaurants)"""
    df = pd.read_csv(filepath)
    # 'Label' column: -1=fake, 1=real (or similar)
    df = df[['Review', 'Label']].copy()
    df.columns = ['text', 'label']
    # Map labels to 0=real, 1=fake
    unique_labels = df['label'].unique()
    print(f"  Yelp unique labels found: {unique_labels}")
    if -1 in unique_labels:
        df['label'] = df['label'].map({-1: 1, 1: 0})
    else:
        df['label'] = df['label'].map({0: 0, 1: 1})
    df = df.dropna()
    # Use a balanced sample of 1600 reviews for efficiency
    fake = df[df['label']==1].sample(n=min(800, len(df[df['label']==1])), random_state=42)
    real = df[df['label']==0].sample(n=min(800, len(df[df['label']==0])), random_state=42)
    df = shuffle(pd.concat([fake, real]), random_state=42).reset_index(drop=True)
    print(f"Yelp dataset loaded: {len(df)} reviews")
    print(f"  Fake: {df['label'].sum()}, Real: {(df['label']==0).sum()}")
    return df

def load_all_datasets(hotels_path, products_path, yelp_path):
    """Load all Group I datasets"""
    print("\n" + "="*50)
    print("Loading all datasets...")
    print("="*50)
    hotels = load_hotels(hotels_path)
    products = load_online_products(products_path)
    yelp = load_yelp(yelp_path)
    print("="*50)
    print("All datasets loaded successfully!")
    print("="*50 + "\n")
    return hotels, products, yelp
