"""
clustering.py
-------------
Determines the optimal number of clusters using the Elbow Method and
Silhouette Score, then fits a final KMeans model.

Methods
-------
Elbow Method    : Plot inertia (within-cluster sum of squares) vs k.
                  The 'elbow' point where inertia stops dropping sharply
                  is the optimal k.

Silhouette Score: Measures how similar a point is to its own cluster
                  vs other clusters. Ranges from -1 to 1; higher = better.

Both methods are run and compared to pick the best k.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
K_RANGE     = range(2, 11)   # test k from 2 to 10


# ── Elbow + Silhouette ────────────────────────────────────────────────────────

def evaluate_k(rfm_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Run KMeans for k = 2..10 and record inertia and silhouette score.

    Parameters
    ----------
    rfm_scaled : pd.DataFrame
        Scaled RFM features from rfm_scoring.scale_rfm().

    Returns
    -------
    pd.DataFrame with columns: k, inertia, silhouette_score
    """
    print("\nEvaluating k from 2 to 10...")
    features = rfm_scaled[["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]].values

    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(features)
        inertia   = km.inertia_
        sil_score = silhouette_score(features, labels)
        results.append({"k": k, "inertia": inertia, "silhouette_score": sil_score})
        print(f"  k={k:2d}  |  Inertia: {inertia:>10.1f}  |  Silhouette: {sil_score:.4f}")

    return pd.DataFrame(results)


def find_optimal_k(eval_df: pd.DataFrame) -> int:
    """
    Identify optimal k using the elbow method (second derivative of inertia)
    and confirm with the silhouette score.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Output of evaluate_k().

    Returns
    -------
    int  Recommended optimal k.
    """
    inertias = eval_df["inertia"].values
    ks       = eval_df["k"].values

    # Second derivative of inertia — peak = elbow point
    deltas       = np.diff(inertias)
    second_deriv = np.diff(deltas)
    elbow_k      = ks[np.argmax(second_deriv) + 2]  # offset for double diff

    # Best silhouette score
    sil_k = eval_df.loc[eval_df["silhouette_score"].idxmax(), "k"]

    print(f"\n  Elbow method suggests   k = {elbow_k}")
    print(f"  Silhouette score suggests k = {sil_k}")

    # If they agree, use that k; otherwise prefer silhouette
    optimal_k = elbow_k if elbow_k == sil_k else sil_k
    print(f"  ➡ Optimal k selected    : {optimal_k}")
    return int(optimal_k)


# ── Fit Final Model ───────────────────────────────────────────────────────────

def fit_kmeans(rfm_scaled: pd.DataFrame, k: int) -> tuple[KMeans, np.ndarray]:
    """
    Fit the final KMeans model with the chosen k.

    Parameters
    ----------
    rfm_scaled : pd.DataFrame  Scaled RFM features.
    k          : int           Number of clusters.

    Returns
    -------
    model  : fitted KMeans model
    labels : cluster assignment array
    """
    print(f"\nFitting final KMeans model with k={k}...")
    features = rfm_scaled[["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]].values

    model  = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = model.fit_predict(features)

    final_sil = silhouette_score(features, labels)
    print(f"  Final silhouette score: {final_sil:.4f}")
    print(f"  Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"    Cluster {cluster}: {count:,} customers ({count/len(labels):.1%})")

    return model, labels


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_clustering_pipeline(rfm_scaled: pd.DataFrame) -> tuple[pd.DataFrame, KMeans, np.ndarray, int]:
    """
    Full clustering pipeline: evaluate k → find optimal → fit model.

    Returns
    -------
    eval_df   : k evaluation results (for elbow plot)
    model     : fitted KMeans
    labels    : cluster assignments
    optimal_k : chosen k
    """
    eval_df   = evaluate_k(rfm_scaled)
    optimal_k = find_optimal_k(eval_df)
    model, labels = fit_kmeans(rfm_scaled, optimal_k)
    return eval_df, model, labels, optimal_k
