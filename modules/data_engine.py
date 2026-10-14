import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def generate_synthetic_data(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Generates synthetic customer demographic and behavioral profiles."""
    np.random.seed(random_state)
    per_cluster = n_samples // 4
    
    # 4 distinct customer distributions
    c1 = np.random.multivariate_normal([24, 28, 22, 12, 1], [[15, 0, 0, 0, 0], [0, 40, 0, 0, 0], [0, 0, 80, 0, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 1]], per_cluster)
    c2 = np.random.multivariate_normal([46, 125, 82, 38, 5], [[25, 0, 0, 0, 0], [0, 110, 0, 0, 0], [0, 0, 160, 0, 0], [0, 0, 0, 20, 0], [0, 0, 0, 0, 2]], per_cluster)
    c3 = np.random.multivariate_normal([34, 72, 44, 24, 3], [[20, 0, 0, 0, 0], [0, 70, 0, 0, 0], [0, 0, 120, 0, 0], [0, 0, 0, 15, 0], [0, 0, 0, 0, 1]], per_cluster)
    c4 = np.random.multivariate_normal([58, 42, 88, 54, 4], [[30, 0, 0, 0, 0], [0, 50, 0, 0, 0], [0, 0, 95, 0, 0], [0, 0, 0, 25, 0], [0, 0, 0, 0, 2]], per_cluster)
    
    raw = np.vstack([c1, c2, c3, c4])
    df = pd.DataFrame(raw, columns=["Age", "Annual_Income_k$", "Spending_Score", "Tenure_Months", "Category_Diversity"])
    
    df["Age"] = np.clip(df["Age"].astype(int), 18, 75)
    df["Annual_Income_k$"] = np.clip(df["Annual_Income_k$"].astype(int), 15, 180)
    df["Spending_Score"] = np.clip(df["Spending_Score"].astype(int), 1, 100)
    df["Tenure_Months"] = np.clip(df["Tenure_Months"].astype(int), 1, 72)
    df["Category_Diversity"] = np.clip(df["Category_Diversity"].astype(int), 1, 8)
    df["CustomerID"] = [f"CUST-{1000 + i}" for i in range(len(df))]
    
    # Derive Monetary approximation for CLV calculations
    df["Monetary_Total"] = (df["Annual_Income_k$"] * 10 * (df["Spending_Score"] / 100) * (1 + df["Tenure_Months"] / 36)).round(2)
    
    return df[["CustomerID", "Age", "Annual_Income_k$", "Spending_Score", "Tenure_Months", "Category_Diversity", "Monetary_Total"]]

@st.cache_data
def process_rfm_transactions(df_tx: pd.DataFrame, cust_col: str, date_col: str, amount_col: str) -> pd.DataFrame:
    """Aggregates raw invoice transactions into clean Customer-level RFM features."""
    df_tx[date_col] = pd.to_datetime(df_tx[date_col])
    snapshot_date = df_tx[date_col].max() + pd.Timedelta(days=1)
    
    rfm = df_tx.groupby(cust_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(date_col, 'count'),
        Monetary=(amount_col, 'sum'),
        AOV=(amount_col, 'mean'),
        Tenure_Days=(date_col, lambda x: (snapshot_date - x.min()).days)
    ).reset_index()
    
    # Velocity scores
    rfm["Purchase_Velocity"] = (rfm["Frequency"] / (rfm["Tenure_Days"] + 1)).round(4)
    return rfm

def compute_data_quality_report(df: pd.DataFrame, feature_cols: list) -> dict:
    """Calculates missingness, basic statistics, correlations, and Variance Inflation Factor."""
    sub_df = df[feature_cols].dropna()
    corr_matrix = sub_df.corr().round(3)
    
    # Compute basic VIF approximation
    vif_data = {}
    for col in feature_cols:
        other_cols = [c for c in feature_cols if c != col]
        if len(other_cols) >= 1:
            try:
                # Regress col on others to compute 1 / (1 - R^2)
                y = sub_df[col]
                X = sub_df[other_cols]
                corr_y_X = sub_df.corr().loc[col, other_cols].values
                corr_XX = sub_df[other_cols].corr().values
                r_sq = np.dot(corr_y_X, np.linalg.pinv(corr_XX)).dot(corr_y_X)
                vif = 1.0 / (1.0 - r_sq) if (1.0 - r_sq) > 1e-4 else 999.0
            except Exception:
                vif = 1.0
        else:
            vif = 1.0
        vif_data[col] = round(vif, 2)
        
    return {
        "missing_counts": df[feature_cols].isnull().sum(),
        "describe": sub_df.describe().T,
        "correlation": corr_matrix,
        "vif": pd.Series(vif_data, name="VIF")
    }