# 🎯 Customer Segmentation: RFM Analysis + KMeans Clustering

An end-to-end customer segmentation pipeline using the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail). Customers are segmented using **RFM analysis** (Recency, Frequency, Monetary) and **KMeans clustering**, with the optimal number of segments determined automatically via the **Elbow Method** and **Silhouette Score**.

---

## 🎯 Business Question

> *Who are our most valuable customers, who is at risk of churning, and what marketing strategy should we apply to each segment?*

---

## 📁 Project Structure

```
customer_segmentation/
├── data/                        # Place OnlineRetail.xlsx here
├── outputs/                     # Generated charts + executive summary
├── src/
│   ├── data_prep.py             # Load and clean transaction data
│   ├── rfm_scoring.py           # Calculate and scale RFM features
│   ├── clustering.py            # Elbow method + KMeans
│   ├── profiling.py             # Segment labels & business recommendations
│   └── visualizations.py       # All charts and summary dashboard
├── main.py                      # Run the full pipeline
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**UCI Online Retail Dataset** — real-world UK e-commerce transactions (Dec 2010 – Dec 2011).

**Download:** https://archive.ics.uci.edu/dataset/352/online+retail  
Place `OnlineRetail.xlsx` in the `/data/` folder before running.

---

## 📐 Methodology

### RFM Features

| Feature | Definition | Why It Matters |
|---|---|---|
| **Recency** | Days since last purchase | Recent buyers are more likely to buy again |
| **Frequency** | Number of unique orders | Frequent buyers have higher brand loyalty |
| **Monetary** | Total revenue generated | High spenders drive disproportionate revenue |

### Preprocessing
- Log-transform applied to Frequency and Monetary (right-skewed distributions)
- StandardScaler applied to all features (KMeans is distance-based)

### Clustering
- KMeans evaluated for k = 2 to 10
- Optimal k selected via **Elbow Method** (second derivative of inertia) + **Silhouette Score**
- Segments ranked by composite RFM score and assigned business labels

### Segment Labels

| Segment | Profile | Strategy |
|---|---|---|
| **Champions** | High F, High M, Low R | VIP rewards, early access |
| **Loyal Customers** | Regular buyers, strong spend | Upsell, loyalty programs |
| **Promising** | Recent, growing frequency | Onboarding offers |
| **At Risk** | Previously valuable, gone quiet | Win-back campaigns |
| **Lost** | Low across all metrics | Low-cost reactivation or suppress |

---

## 📈 Output Charts

| File | Description |
|---|---|
| `01_rfm_distributions.png` | Histograms of R, F, M features |
| `02_elbow_curve.png` | Inertia + silhouette score vs k |
| `03_segment_sizes.png` | Bar + pie chart of segment sizes |
| `04_rfm_heatmap.png` | Mean RFM values per segment |
| `05_fm_scatter.png` | Frequency vs Monetary scatter by segment |
| `06_summary_dashboard.png` | Full summary dashboard |
| `executive_summary.txt` | Business-ready text summary with recommendations |

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/customer_segmentation.git
cd customer_segmentation
pip install -r requirements.txt
# Add OnlineRetail.xlsx to /data/
python main.py
```

---

## 🛠 Tech Stack

- **Python 3.14+**
- `pandas` — data wrangling
- `numpy` — numerical computing
- `scikit-learn` — KMeans, StandardScaler, silhouette score
- `matplotlib` — visualizations

---

## 💡 Key Concepts Demonstrated

- RFM feature engineering from raw transaction data
- Unsupervised machine learning (KMeans clustering)
- Model selection via elbow method and silhouette analysis
- Business segment profiling and marketing strategy mapping
- Executive-ready reporting and visualization

---

## 👤 Author

Built as part of a data science portfolio project.  
Background: 15+ years in Marketing Analytics | SQL | Python | Statistical Modeling
