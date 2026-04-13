"""
COAT Framework - Main execution file
Cross-Domain Fake Review Detection via Orthogonal Counterfactual Representations

Group I Datasets:
- Online Products (P)
- Hotels (H)
- Yelp Restaurant (R)

Source -> Target pairs:
P->R, H->R, R->P, H->P, P->H, R->H
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data_loader import load_all_datasets
from phase1_masking import two_step_masking
from phase2_counterfactuals import generate_counterfactuals
from phase3_classifier import train_and_evaluate

# ============================================================
# CONFIGURATION - Update these paths to match your files
# ============================================================
HOTELS_PATH = "data/group1/hotels/deceptive-opinion.csv"
PRODUCTS_PATH = "data/group1/online_products/fake reviews dataset.csv"
YELP_PATH = "data/group1/yelp/Labelled Yelp Dataset.csv"

def run_coat(source_df, target_df, source_name, target_name):
    """
    Run complete COAT framework for one source->target pair
    """
    print("\n" + "="*60)
    print(f"Running COAT: {source_name} --> {target_name}")
    print("="*60)

    source_reviews = source_df['text'].tolist()
    source_labels = source_df['label'].tolist()
    target_reviews = target_df['text'].tolist()
    target_labels = target_df['label'].tolist()

    # Phase I: Two-Step Masking
    masked_reviews = two_step_masking(
        source_reviews=source_reviews,
        target_reviews=target_reviews,
        heuristic_threshold=50,
        contextual_threshold=50
    )

    # Phase II: Generate Counterfactuals
    counterfactual_embeddings, filled_reviews, R = generate_counterfactuals(
        masked_reviews=masked_reviews,
        source_labels=source_labels,
        target_reviews=target_reviews,
        source_reviews_original=source_reviews
    )

    # Phase III: Train and Evaluate
    accuracy, f1, model = train_and_evaluate(
        counterfactual_embeddings=counterfactual_embeddings,
        source_labels=source_labels,
        target_reviews=target_reviews,
        target_labels=target_labels,
        n_folds=5,
        epochs=100,
        batch_size=32
    )

    return accuracy, f1

def main():
    print("\n" + "="*60)
    print("  COAT Framework - Cross Domain Fake Review Detection")
    print("="*60)

    # ── Load datasets ──────────────────────────────────────────
    hotels, products, yelp = load_all_datasets(
        HOTELS_PATH, PRODUCTS_PATH, YELP_PATH
    )

    datasets = {
        'Hotels (H)': hotels,
        'Online Products (P)': products,
        'Yelp Restaurant (R)': yelp
    }

    # ── Define source->target pairs (Group I) ─────────────────
    pairs = [
        ('Online Products (P)', 'Yelp Restaurant (R)', 'P->R'),
        ('Hotels (H)',          'Yelp Restaurant (R)', 'H->R'),
        ('Yelp Restaurant (R)', 'Online Products (P)', 'R->P'),
        ('Hotels (H)',          'Online Products (P)', 'H->P'),
        ('Online Products (P)', 'Hotels (H)',          'P->H'),
        ('Yelp Restaurant (R)', 'Hotels (H)',          'R->H'),
    ]

    # ── Run COAT for each pair ─────────────────────────────────
    results = []
    for source_name, target_name, pair_label in pairs:
        accuracy, f1 = run_coat(
            source_df=datasets[source_name],
            target_df=datasets[target_name],
            source_name=source_name,
            target_name=target_name
        )
        results.append({
            'Source -> Target': pair_label,
            'Accuracy': round(accuracy, 4),
            'Accuracy (%)': round(accuracy * 100, 2),
            'F1-Score': round(f1, 4),
            'F1-Score (%)': round(f1 * 100, 2)
        })

    # ── Print final results table ──────────────────────────────
    print("\n" + "="*60)
    print("  FINAL RESULTS - Group I")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    avg_acc = results_df['Accuracy'].mean()
    avg_f1  = results_df['F1-Score'].mean()
    print("\n" + "-"*60)
    print(f"  Average Accuracy : {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"  Average F1-Score : {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print("="*60)

    # Save results to CSV
    results_df.to_csv('coat_results_group1.csv', index=False)
    print("\n Results saved to: coat_results_group1.csv")

if __name__ == "__main__":
    main()
