"""
data_prep.py
------------
Loads and cleans the UCI Online Retail dataset for RFM segmentation.

Steps
-----
1. Load the Excel file.
2. Drop missing CustomerIDs, cancelled orders, and bad rows.
3. Add a LineRevenue column.
4. Return a clean transaction-level DataFrame ready for RFM scoring.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load the UCI Online Retail Excel file."""
    print(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath, dtype={"CustomerID": str})
    print(f"  Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw transaction data.

    - Drop rows with missing CustomerID
    - Remove cancelled orders (InvoiceNo starts with 'C')
    - Remove non-positive Quantity or UnitPrice
    - Parse InvoiceDate as datetime
    - Add LineRevenue column
    """
    print("Cleaning data...")

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Unique customers: {df['CustomerID'].nunique():,}")
    print(f"  Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    return df


def run_prep_pipeline(filepath: str) -> pd.DataFrame:
    """Full load + clean pipeline."""
    df_raw   = load_data(filepath)
    df_clean = clean_data(df_raw)
    return df_clean


if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "OnlineRetail.xlsx")
    df = run_prep_pipeline(path)
    print(df.head())
