"""
visualizations.py
-----------------
Generates all plots for the customer segmentation analysis.

Charts
------
1. RFM Distributions     – histograms of R, F, M
2. Elbow Curve           – inertia + silhouette vs k
3. Segment Size          – bar chart of customer counts per segment
4. RFM Heatmap           – mean R/F/M per segment
5. Scatter: F vs M       – colored by segment
6. Summary Dashboard     – 2x3 grid of key visuals
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "figure.dpi"       : 120,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig, filename):
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def _segment_colors(cluster_summary):
    return dict(zip(cluster_summary["Segment"], cluster_summary["Color"]))


# ── Plot 1: RFM Distributions ─────────────────────────────────────────────────

def plot_rfm_distributions(rfm: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("RFM Feature Distributions", fontsize=14, fontweight="bold")

    configs = [
        ("Recency",   "Days Since Last Purchase", "#E74C3C"),
        ("Frequency", "Number of Orders",          "#3498DB"),
        ("Monetary",  "Total Revenue (£)",         "#2ECC71"),
    ]

    for ax, (col, xlabel, color) in zip(axes, configs):
        ax.hist(rfm[col], bins=50, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(rfm[col].median(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Median: {rfm[col].median():.0f}")
        ax.set_title(col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Customers")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "01_rfm_distributions.png")


# ── Plot 2: Elbow Curve ───────────────────────────────────────────────────────

def plot_elbow_curve(eval_df: pd.DataFrame, optimal_k: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Optimal Number of Clusters", fontsize=14, fontweight="bold")

    # Inertia
    ax1.plot(eval_df["k"], eval_df["inertia"], "o-", color="#3498DB", linewidth=2, markersize=6)
    ax1.axvline(optimal_k, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Optimal k={optimal_k}")
    ax1.set_title("Elbow Method (Inertia)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.legend()

    # Silhouette
    ax2.plot(eval_df["k"], eval_df["silhouette_score"], "o-", color="#2ECC71", linewidth=2, markersize=6)
    ax2.axvline(optimal_k, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Optimal k={optimal_k}")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()

    plt.tight_layout()
    _save(fig, "02_elbow_curve.png")


# ── Plot 3: Segment Sizes ─────────────────────────────────────────────────────

def plot_segment_sizes(cluster_summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Segment Distribution", fontsize=14, fontweight="bold")

    colors  = cluster_summary["Color"].tolist()
    labels  = cluster_summary["Segment"].tolist()
    counts  = cluster_summary["Customer_count"].tolist()

    # Bar chart
    bars = ax1.bar(labels, counts, color=colors, alpha=0.85)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{count:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_title("Customers per Segment")
    ax1.set_ylabel("Number of Customers")
    ax1.tick_params(axis="x", rotation=15)

    # Pie chart
    ax2.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.8,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("Segment Share")

    plt.tight_layout()
    _save(fig, "03_segment_sizes.png")


# ── Plot 4: RFM Heatmap ───────────────────────────────────────────────────────

def plot_rfm_heatmap(cluster_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Recency_mean", "Frequency_mean", "Monetary_mean"]
    labels  = ["Recency\n(days, lower=better)", "Frequency\n(orders)", "Monetary\n(£)"]
    segments = cluster_summary["Segment"].tolist()

    # Normalize each metric 0-1 for color scale
    # For Recency, invert so green = good (low recency days)
    heatmap_data = []
    for _, row in cluster_summary.iterrows():
        heatmap_data.append([row["Recency_mean"], row["Frequency_mean"], row["Monetary_mean"]])
    heatmap_data = np.array(heatmap_data, dtype=float)

    norm_data = heatmap_data.copy()
    for col_idx in range(3):
        mn = heatmap_data[:, col_idx].min()
        mx = heatmap_data[:, col_idx].max()
        norm_data[:, col_idx] = (heatmap_data[:, col_idx] - mn) / (mx - mn + 1e-9)
    norm_data[:, 0] = 1 - norm_data[:, 0]  # invert recency

    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels(segments, fontsize=10, fontweight="bold")

    # Annotate with raw values
    fmt = ["{:.0f}", "{:.1f}", "£{:.0f}"]
    for i in range(len(segments)):
        for j in range(3):
            ax.text(j, i, fmt[j].format(heatmap_data[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < norm_data[i, j] < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Normalized Score (green = better)")
    ax.set_title("RFM Profile Heatmap by Segment", pad=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_rfm_heatmap.png")


# ── Plot 5: Frequency vs Monetary Scatter ─────────────────────────────────────

def plot_fm_scatter(rfm_labeled: pd.DataFrame, cluster_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))

    color_map = _segment_colors(cluster_summary)

    for segment, group in rfm_labeled.groupby("Segment"):
        ax.scatter(
            group["Frequency"], group["Monetary"],
            c=color_map.get(segment, "#999999"),
            label=segment, alpha=0.5, s=20, edgecolors="none"
        )

    ax.set_xlabel("Frequency (Number of Orders)")
    ax.set_ylabel("Monetary Value (£)")
    ax.set_title("Customer Segments: Frequency vs Monetary Value", fontweight="bold")
    ax.legend(title="Segment", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, "05_fm_scatter.png")


# ── Plot 6: Summary Dashboard ─────────────────────────────────────────────────

def plot_summary_dashboard(rfm: pd.DataFrame, rfm_labeled: pd.DataFrame,
                           cluster_summary: pd.DataFrame, eval_df: pd.DataFrame,
                           optimal_k: int):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Customer Segmentation Dashboard — RFM + KMeans",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors  = cluster_summary["Color"].tolist()
    segments = cluster_summary["Segment"].tolist()
    counts   = cluster_summary["Customer_count"].tolist()
    color_map = _segment_colors(cluster_summary)

    # ── Top-left: Elbow curve ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eval_df["k"], eval_df["inertia"], "o-", color="#3498DB", linewidth=2, markersize=5)
    ax1.axvline(optimal_k, color="#E74C3C", linestyle="--", linewidth=1.5)
    ax1.set_title(f"Elbow Method (k={optimal_k} selected)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")

    # ── Top-center: Segment sizes ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(segments, counts, color=colors, alpha=0.85)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{count:,}", ha="center", va="bottom", fontsize=7)
    ax2.set_title("Customers per Segment")
    ax2.tick_params(axis="x", rotation=20, labelsize=8)

    # ── Top-right: Pie chart ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.pie(counts, labels=segments, colors=colors, autopct="%1.0f%%",
            startangle=140, pctdistance=0.8, textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax3.set_title("Segment Share")

    # ── Bottom-left: F vs M scatter ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for segment, group in rfm_labeled.groupby("Segment"):
        ax4.scatter(group["Frequency"], group["Monetary"],
                    c=color_map.get(segment, "#999"), label=segment,
                    alpha=0.4, s=10, edgecolors="none")
    ax4.set_xlabel("Frequency")
    ax4.set_ylabel("Monetary (£)")
    ax4.set_title("Frequency vs Monetary")
    ax4.set_xlim(left=0); ax4.set_ylim(bottom=0)

    # ── Bottom-center: Recency distribution by segment ────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for segment in segments:
        grp = rfm_labeled[rfm_labeled["Segment"] == segment]["Recency"]
        ax5.hist(grp, bins=30, alpha=0.6, color=color_map[segment], label=segment)
    ax5.set_xlabel("Recency (days)")
    ax5.set_ylabel("Customers")
    ax5.set_title("Recency Distribution by Segment")
    ax5.legend(fontsize=7)

    # ── Bottom-right: Summary metrics table ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = [["Segment", "Customers", "Avg £", "Avg Orders"]]
    for _, row in cluster_summary.iterrows():
        table_data.append([
            row["Segment"],
            f"{int(row['Customer_count']):,}",
            f"£{row['Monetary_mean']:,.0f}",
            f"{row['Frequency_mean']:.1f}",
        ])
    tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.5)
    ax6.set_title("Segment Summary", pad=12)

    plt.tight_layout()
    _save(fig, "06_summary_dashboard.png")


# ── Run All ───────────────────────────────────────────────────────────────────

def generate_all_plots(rfm, rfm_labeled, cluster_summary, eval_df, optimal_k):
    print("\nGenerating visualizations...")
    plot_rfm_distributions(rfm)
    plot_elbow_curve(eval_df, optimal_k)
    plot_segment_sizes(cluster_summary)
    plot_rfm_heatmap(cluster_summary)
    plot_fm_scatter(rfm_labeled, cluster_summary)
    plot_summary_dashboard(rfm, rfm_labeled, cluster_summary, eval_df, optimal_k)
    print("All plots saved to /outputs/\n")
