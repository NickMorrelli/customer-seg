"""
profiling.py
------------
Assigns business-friendly labels to each KMeans cluster based on their
RFM profiles, and generates an executive summary with actionable
marketing recommendations for each segment.

Labeling Logic
--------------
Clusters are ranked by a composite RFM score:
  - Low Recency (recent buyers) → good
  - High Frequency               → good
  - High Monetary                → good

Segments are then mapped to intuitive business labels like:
  "Champions", "Loyal Customers", "At Risk", "Lost", etc.
"""

import pandas as pd
import numpy as np


# ── Segment Definitions ───────────────────────────────────────────────────────
# Labels and recommendations mapped by RFM tier rank (best → worst)
SEGMENT_PROFILES = [
    {
        "label"         : "Champions",
        "description"   : "Bought recently, buy often, and spend the most.",
        "color"         : "#2ECC71",
        "strategy"      : "Reward them. They're your brand advocates — offer early access, loyalty rewards, and VIP treatment.",
    },
    {
        "label"         : "Loyal Customers",
        "description"   : "Buy regularly with strong monetary value.",
        "color"         : "#3498DB",
        "strategy"      : "Upsell higher-value products. Ask for reviews. Enroll in loyalty programs.",
    },
    {
        "label"         : "Promising",
        "description"   : "Recent shoppers with growing frequency.",
        "color"         : "#F39C12",
        "strategy"      : "Build the relationship. Offer onboarding discounts and product recommendations to increase frequency.",
    },
    {
        "label"         : "At Risk",
        "description"   : "Previously valuable customers who haven't purchased recently.",
        "color"         : "#E67E22",
        "strategy"      : "Win them back with a personalized re-engagement campaign and time-limited offers.",
    },
    {
        "label"         : "Lost",
        "description"   : "Low recency, frequency, and monetary value.",
        "color"         : "#E74C3C",
        "strategy"      : "Low-cost reactivation attempt. If unresponsive, suppress from campaigns to protect deliverability.",
    },
]


# ── Assign Labels ─────────────────────────────────────────────────────────────

def profile_segments(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Merge cluster labels with raw RFM values and assign business segment names.

    Ranking logic:
      1. Compute a composite score per cluster:
            score = -Recency_mean + Frequency_mean + Monetary_mean (all normalized)
      2. Rank clusters from best (highest score) to worst.
      3. Map rank to SEGMENT_PROFILES list.

    Parameters
    ----------
    rfm    : pd.DataFrame  Raw RFM values (from rfm_scoring.calculate_rfm).
    labels : np.ndarray    Cluster assignments from KMeans.

    Returns
    -------
    pd.DataFrame  Customer-level DataFrame with cluster and segment label.
    """
    rfm_labeled = rfm.copy()
    rfm_labeled["Cluster"] = labels

    # ── Cluster-level summary ──────────────────────────────────────────────
    cluster_summary = (
        rfm_labeled.groupby("Cluster")
        .agg(
            Recency_mean   = ("Recency",   "mean"),
            Frequency_mean = ("Frequency", "mean"),
            Monetary_mean  = ("Monetary",  "mean"),
            Customer_count = ("CustomerID","count"),
        )
        .reset_index()
    )

    # ── Normalize for composite scoring ───────────────────────────────────
    for col in ["Recency_mean", "Frequency_mean", "Monetary_mean"]:
        mn = cluster_summary[col].min()
        mx = cluster_summary[col].max()
        cluster_summary[f"{col}_norm"] = (cluster_summary[col] - mn) / (mx - mn + 1e-9)

    # Recency: lower days = better, so invert
    cluster_summary["composite_score"] = (
        -cluster_summary["Recency_mean_norm"]
        + cluster_summary["Frequency_mean_norm"]
        + cluster_summary["Monetary_mean_norm"]
    )

    # ── Rank and assign labels ─────────────────────────────────────────────
    cluster_summary = cluster_summary.sort_values("composite_score", ascending=False).reset_index(drop=True)
    n_clusters = len(cluster_summary)

    # Use as many profiles as we have clusters
    profiles_to_use = SEGMENT_PROFILES[:n_clusters]

    cluster_summary["Segment"]     = [p["label"]       for p in profiles_to_use]
    cluster_summary["Description"] = [p["description"] for p in profiles_to_use]
    cluster_summary["Color"]       = [p["color"]       for p in profiles_to_use]
    cluster_summary["Strategy"]    = [p["strategy"]    for p in profiles_to_use]

    # ── Merge back to customer level ───────────────────────────────────────
    cluster_map = cluster_summary.set_index("Cluster")["Segment"].to_dict()
    rfm_labeled["Segment"] = rfm_labeled["Cluster"].map(cluster_map)

    return rfm_labeled, cluster_summary


# ── Executive Summary ─────────────────────────────────────────────────────────

def generate_executive_summary(rfm_labeled: pd.DataFrame, cluster_summary: pd.DataFrame) -> str:
    """
    Generate a text-based executive summary of the segmentation results.

    Parameters
    ----------
    rfm_labeled     : Customer-level DataFrame with Segment column.
    cluster_summary : Cluster-level summary with business labels.

    Returns
    -------
    str  Formatted executive summary.
    """
    total_customers = len(rfm_labeled)
    total_revenue   = rfm_labeled["Monetary"].sum()

    lines = []
    lines.append("=" * 70)
    lines.append("  CUSTOMER SEGMENTATION — EXECUTIVE SUMMARY")
    lines.append("  RFM Analysis + KMeans Clustering | UCI Online Retail Dataset")
    lines.append("=" * 70)
    lines.append(f"\n  Total Customers Analyzed : {total_customers:,}")
    lines.append(f"  Total Revenue (Period)   : £{total_revenue:,.2f}")
    lines.append(f"  Number of Segments       : {len(cluster_summary)}")
    lines.append("\n" + "-" * 70)

    for _, row in cluster_summary.iterrows():
        pct = row["Customer_count"] / total_customers * 100
        rev_share = (
            rfm_labeled[rfm_labeled["Segment"] == row["Segment"]]["Monetary"].sum()
            / total_revenue * 100
        )
        lines.append(f"\n  🎯 {row['Segment'].upper()}")
        lines.append(f"     {row['Description']}")
        lines.append(f"     Customers : {int(row['Customer_count']):,}  ({pct:.1f}% of base)")
        lines.append(f"     Avg Recency   : {row['Recency_mean']:.0f} days")
        lines.append(f"     Avg Frequency : {row['Frequency_mean']:.1f} orders")
        lines.append(f"     Avg Monetary  : £{row['Monetary_mean']:,.2f}")
        lines.append(f"     Revenue Share : {rev_share:.1f}%")
        lines.append(f"     Strategy      : {row['Strategy']}")

    lines.append("\n" + "=" * 70)
    lines.append("  KEY TAKEAWAYS")
    lines.append("-" * 70)

    # Champions insight
    champ = cluster_summary[cluster_summary["Segment"] == "Champions"]
    if not champ.empty:
        champ_rev = rfm_labeled[rfm_labeled["Segment"] == "Champions"]["Monetary"].sum()
        champ_pct_customers = champ["Customer_count"].values[0] / total_customers * 100
        champ_pct_revenue   = champ_rev / total_revenue * 100
        lines.append(f"\n  • Champions represent {champ_pct_customers:.1f}% of customers")
        lines.append(f"    but drive {champ_pct_revenue:.1f}% of revenue — protect this segment.")

    # At Risk insight
    at_risk = cluster_summary[cluster_summary["Segment"] == "At Risk"]
    if not at_risk.empty:
        ar_rev = rfm_labeled[rfm_labeled["Segment"] == "At Risk"]["Monetary"].sum()
        lines.append(f"\n  • At Risk customers represent £{ar_rev:,.2f} in recoverable revenue.")
        lines.append(f"    A targeted win-back campaign here has high ROI potential.")

    lines.append("\n" + "=" * 70 + "\n")
    return "\n".join(lines)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_profiling_pipeline(rfm: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Full profiling pipeline: assign labels → generate summary.

    Returns
    -------
    rfm_labeled     : customer-level data with segment labels
    cluster_summary : cluster-level summary table
    exec_summary    : formatted executive summary string
    """
    print("\nProfiling segments...")
    rfm_labeled, cluster_summary = profile_segments(rfm, labels)
    exec_summary = generate_executive_summary(rfm_labeled, cluster_summary)
    print(exec_summary)
    return rfm_labeled, cluster_summary, exec_summary
