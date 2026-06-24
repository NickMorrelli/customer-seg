"""
main.py
-------
End-to-end pipeline for the Customer Segmentation analysis.

Usage
-----
    python main.py          (run from the customer_segmentation/ folder)

Steps
-----
1. Load and clean the UCI Online Retail dataset.
2. Calculate RFM metrics per customer.
3. Scale features and determine optimal k via elbow method.
4. Fit KMeans and assign cluster labels.
5. Profile segments and generate executive summary.
6. Generate and save all visualizations to /outputs/.
"""

import os
import sys

from src.data_prep      import run_prep_pipeline
from src.rfm_scoring    import run_rfm_pipeline
from src.clustering     import run_clustering_pipeline
from src.profiling      import run_profiling_pipeline
from src.visualizations import generate_all_plots

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "OnlineRetail.xlsx")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "outputs", "executive_summary.txt")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CUSTOMER SEGMENTATION — RFM + KMeans Clustering")
    print("  UCI Online Retail Dataset")
    print("=" * 65)

    # ── Step 1: Data Preparation ───────────────────────────────────────────
    print("\n[1/5] Data Preparation")
    if not os.path.exists(DATA_PATH):
        print(f"\n  ❌ Dataset not found at: {DATA_PATH}")
        print("  Please download 'OnlineRetail.xlsx' from:")
        print("  https://archive.ics.uci.edu/dataset/352/online+retail")
        print("  and place it in the /data/ folder.\n")
        sys.exit(1)

    df_clean = run_prep_pipeline(DATA_PATH)

    # ── Step 2: RFM Scoring ────────────────────────────────────────────────
    print("\n[2/5] RFM Scoring")
    rfm, rfm_log, rfm_scaled, scaler = run_rfm_pipeline(df_clean)

    # ── Step 3: Clustering ─────────────────────────────────────────────────
    print("\n[3/5] Finding Optimal Clusters")
    eval_df, model, labels, optimal_k = run_clustering_pipeline(rfm_scaled)

    # ── Step 4: Segment Profiling ──────────────────────────────────────────
    print("\n[4/5] Profiling Segments")
    rfm_labeled, cluster_summary, exec_summary = run_profiling_pipeline(rfm, labels)

    # Save executive summary to file
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(exec_summary)
    print(f"  Executive summary saved to: {SUMMARY_PATH}")

    # ── Step 5: Visualizations ─────────────────────────────────────────────
    print("\n[5/5] Generating Visualizations")
    generate_all_plots(rfm, rfm_labeled, cluster_summary, eval_df, optimal_k)

    # ── Done ───────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs saved to: {os.path.join(os.path.dirname(__file__), 'outputs')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
