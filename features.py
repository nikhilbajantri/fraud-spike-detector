"""Shared feature engineering, used by both train.py and app.py so the live
scorer builds exactly the same feature vector the model was trained on.
"""
import numpy as np
import pandas as pd

# Final ordered feature set: V1..V28 + engineered + raw Amount = 32 features.
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = V_COLS + ["Hour", "LogAmount", "AmountRobustZ", "Amount"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns in-place-safe manner and return the frame."""
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24  # daily cycle from cumulative seconds
    df["LogAmount"] = np.log1p(df["Amount"])
    amt_median = df["Amount"].median()
    q75 = df["Amount"].quantile(0.75)
    q25 = df["Amount"].quantile(0.25)
    amt_iqr = q75 - q25
    df["AmountRobustZ"] = (df["Amount"] - amt_median) / (amt_iqr if amt_iqr > 0 else 1.0)
    return df


def make_xy(df: pd.DataFrame):
    """Return (X, y) with X restricted to FEATURE_COLS in canonical order."""
    df = engineer(df)
    X = df[FEATURE_COLS]
    y = df["Class"].astype(int)
    return X, y
