#!/usr/bin/env python3
"""
Run Amazon Reviews Anomaly Detection Analysis
Simplified version for reliable execution
"""

import pandas as pd
import numpy as np
import gzip
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Amazon Reviews Anomaly Detection")
print("="*60)

# Configuration
OUTPUT_DIR = Path("data")
SEED = 42
np.random.seed(SEED)

print("\n[1/8] Loading data...")
def load_jsonl_gz(file_path, max_rows=None):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

# Load reviews (limit to 100K for faster execution)
print("  Loading reviews (max 100,000 for demo)...")
reviews_df = load_jsonl_gz(OUTPUT_DIR / "All_Beauty.jsonl.gz", max_rows=100000)
print(f"  Loaded: {len(reviews_df):,} reviews")

print("\n  Loading metadata...")
meta_df = load_jsonl_gz(OUTPUT_DIR / "meta_All_Beauty.jsonl.gz")
print(f"  Loaded: {len(meta_df):,} products")

print("\n[2/8] Cleaning data...")
# Clean reviews
required = ["rating", "user_id", "parent_asin", "timestamp"]
reviews_clean = reviews_df.dropna(subset=required)
has_text = (
    reviews_clean["text"].fillna("").str.strip().ne("") |
    reviews_clean["title"].fillna("").str.strip().ne("")
)
reviews_clean = reviews_clean[has_text]
reviews_clean = reviews_clean[(reviews_clean["rating"] >= 1) & (reviews_clean["rating"] <= 5)]
reviews_clean["verified_purchase"] = reviews_clean.get("verified_purchase", False).fillna(False).astype(bool)
reviews_clean["helpful_vote"] = reviews_clean.get("helpful_vote", 0).fillna(0)
reviews_clean = reviews_clean.drop_duplicates(
    subset=["user_id", "parent_asin", "text", "timestamp"], keep="first"
)
reviews_clean["review_time"] = pd.to_datetime(reviews_clean["timestamp"], unit="ms")
print(f"  After cleaning: {len(reviews_clean):,} reviews")

# Clean metadata
meta_clean = meta_df[['parent_asin', 'price', 'average_rating', 'rating_number', 'store', 'main_category']].copy()
if 'price' in meta_clean.columns:
    meta_clean['price'] = meta_clean['price'].astype(str).str.replace('$', '').str.replace(',', '')
    meta_clean['price'] = pd.to_numeric(meta_clean['price'], errors='coerce')
meta_clean = meta_clean.drop_duplicates('parent_asin')

print("\n[3/8] Merging data...")
reviews_merged = reviews_clean.merge(meta_clean, on='parent_asin', how='left', suffixes=('', '_product'))
print(f"  Merged shape: {reviews_merged.shape}")

print("\n[4/8] Feature engineering...")
# Text features
text = reviews_merged["text"].fillna("")
reviews_merged['text_length'] = text.str.len()
reviews_merged['word_count'] = text.str.split().str.len()
reviews_merged['uppercase_ratio'] = text.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
reviews_merged['punctuation_ratio'] = text.apply(lambda x: sum(c in "!?.,;:" for c in x) / max(len(x), 1))

# Product features
if 'average_rating' in reviews_merged.columns:
    reviews_merged['rating_deviation'] = reviews_merged['rating'] - reviews_merged['average_rating']
reviews_merged['is_extreme_rating'] = ((reviews_merged['rating'] == 1.0) | (reviews_merged['rating'] == 5.0)).astype(int)

print(f"  Added {len(reviews_merged.columns) - len(reviews_clean.columns)} features")

print("\n[5/8] User behavioral features...")
# User aggregations
user_stats = reviews_merged.groupby('user_id').agg({
    'rating': ['count', 'mean', 'std'],
    'verified_purchase': 'mean',
    'helpful_vote': 'sum',
    'word_count': 'mean',
    'is_extreme_rating': 'mean'
}).reset_index()

user_stats.columns = ['user_id', 'user_review_count', 'user_avg_rating', 'user_rating_std',
                      'user_verified_ratio', 'user_total_helpful', 'user_avg_word_count', 'user_extreme_ratio']
user_stats = user_stats.fillna(0)
reviews_merged = reviews_merged.merge(user_stats, on='user_id', how='left')
print(f"  Added user features")

print("\n[6/8] Preparing features for anomaly detection...")
# Select numeric features
feature_cols = [
    'rating', 'rating_deviation', 'text_length', 'word_count',
    'uppercase_ratio', 'punctuation_ratio',
    'verified_purchase', 'helpful_vote',
    'user_review_count', 'user_avg_rating', 'user_rating_std',
    'user_verified_ratio', 'user_total_helpful', 'user_avg_word_count', 'user_extreme_ratio',
    'is_extreme_rating'
]

# Filter available columns
available_cols = [c for c in feature_cols if c in reviews_merged.columns]
X = reviews_merged[available_cols].fillna(0).values

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Feature matrix shape: {X_scaled.shape}")

print("\n[7/8] Training anomaly detection models...")

# Model 1: Isolation Forest
print("  Training Isolation Forest...")
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=SEED, n_jobs=-1)
iso_predictions = iso_forest.fit_predict(X_scaled)
iso_scores = iso_forest.score_samples(X_scaled)
reviews_merged['iso_anomaly'] = (iso_predictions == -1).astype(int)
reviews_merged['iso_score'] = iso_scores
print(f"    Found {reviews_merged['iso_anomaly'].sum():,} anomalies ({reviews_merged['iso_anomaly'].mean()*100:.2f}%)")

# Model 2: HBOS
print("  Training HBOS...")
from pyod.models.hbos import HBOS
hbos = HBOS(contamination=0.05, n_bins=10)
hbos.fit(X_scaled)
hbos_predictions = hbos.labels_
hbos_scores = hbos.decision_scores_
reviews_merged['hbos_anomaly'] = hbos_predictions
reviews_merged['hbos_score'] = hbos_scores
print(f"    Found {reviews_merged['hbos_anomaly'].sum():,} anomalies ({reviews_merged['hbos_anomaly'].mean()*100:.2f}%)")

# Model 3: One-Class SVM (on sample)
print("  Training One-Class SVM (sample)...")
from sklearn.svm import OneClassSVM
sample_size = min(50000, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm.fit(X_scaled[sample_idx])
ocsvm_predictions = ocsvm.predict(X_scaled)
ocsvm_scores = ocsvm.score_samples(X_scaled)
reviews_merged['ocsvm_anomaly'] = (ocsvm_predictions == -1).astype(int)
reviews_merged['ocsvm_score'] = ocsvm_scores
print(f"    Found {reviews_merged['ocsvm_anomaly'].sum():,} anomalies ({reviews_merged['ocsvm_anomaly'].mean()*100:.2f}%)")

# Ensemble
reviews_merged['anomaly_votes'] = (
    reviews_merged['iso_anomaly'] +
    reviews_merged['hbos_anomaly'] +
    reviews_merged['ocsvm_anomaly']
)
reviews_merged['ensemble_anomaly'] = (reviews_merged['anomaly_votes'] >= 2).astype(int)
print(f"\n  Ensemble anomalies: {reviews_merged['ensemble_anomaly'].sum():,} ({reviews_merged['ensemble_anomaly'].mean()*100:.2f}%)")

print("\n[8/8] Saving results...")
# Save full results
output_path = OUTPUT_DIR / "anomaly_results.csv"
reviews_merged.to_csv(output_path, index=False)
print(f"  Saved: {output_path}")

# Save high-confidence anomalies
high_conf = reviews_merged[reviews_merged['ensemble_anomaly'] == 1].copy()
high_conf_path = OUTPUT_DIR / "high_confidence_anomalies.csv"
export_cols = ['user_id', 'parent_asin', 'rating', 'text', 'verified_purchase',
               'iso_anomaly', 'hbos_anomaly', 'ocsvm_anomaly', 'anomaly_votes']
export_cols = [c for c in export_cols if c in high_conf.columns]
high_conf[export_cols].to_csv(high_conf_path, index=False)
print(f"  Saved: {high_conf_path}")
print(f"  High-confidence anomalies: {len(high_conf):,}")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
print(f"\nSummary:")
print(f"  Total reviews analyzed: {len(reviews_merged):,}")
print(f"  Isolation Forest anomalies: {reviews_merged['iso_anomaly'].sum():,}")
print(f"  HBOS anomalies: {reviews_merged['hbos_anomaly'].sum():,}")
print(f"  One-Class SVM anomalies: {reviews_merged['ocsvm_anomaly'].sum():,}")
print(f"  Ensemble (2+ votes) anomalies: {reviews_merged['ensemble_anomaly'].sum():,}")
print(f"\nOutput files:")
print(f"  - {output_path}")
print(f"  - {high_conf_path}")
