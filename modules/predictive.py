import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def compute_cluster_centroids(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """Computes cluster geometric centroids in the normalized space."""
    centroids = {}
    for c in set(labels):
        if c != -1:
            centroids[c] = X_scaled[labels == c].mean(axis=0)
    return centroids

def predict_single_customer(input_dict: dict, feature_cols: list, scaler, centroids: dict):
    """Assigns an arbitrary new customer to the closest segment and calculates confidence."""
    row_df = pd.DataFrame([input_dict])[feature_cols]
    scaled_point = scaler.transform(row_df)
    
    dists = {c: euclidean_distances(scaled_point, [center])[0][0] for c, center in centroids.items()}
    best_cluster = min(dists, key=dists.get)
    min_dist = dists[best_cluster]
    
    # Softmax-style confidence score based on inverted distances
    inv_dists = {c: 1.0 / (d + 1e-5) for c, d in dists.items()}
    total_inv = sum(inv_dists.values())
    confidence = (inv_dists[best_cluster] / total_inv) * 100
    
    return best_cluster, round(confidence, 1), round(min_dist, 3)

def compute_clv_and_churn(df_clustered: pd.DataFrame, cluster_col: str, feature_cols: list):
    """Estimates basic customer equity (CLV) and churn likelihood heuristics per segment."""
    summary = df_clustered.groupby(cluster_col)[feature_cols].mean()
    
    # Fallback heuristics based on common columns or synthetic values
    clv_estimates = {}
    churn_risks = {}
    priority_scores = {}
    
    for cluster_id in summary.index:
        if cluster_id == "Noise (-1)":
            clv_estimates[cluster_id] = 0.0
            churn_risks[cluster_id] = 0.85
            priority_scores[cluster_id] = 0.0
            continue
            
        row = summary.loc[cluster_id]
        income = row.get("Annual_Income_k$", 50.0)
        spend = row.get("Spending_Score", 50.0)
        tenure = row.get("Tenure_Months", 24.0)
        
        # Financial heuristic: CLV proxy
        est_clv = (income * 120) * (spend / 100.0) * (1.0 + (tenure / 48.0))
        
        # Risk heuristic: low spend score or extreme tenure creates churn indicators
        churn_risk = np.clip(1.0 - (spend / 110.0) + (0.1 if tenure < 12 else -0.1), 0.05, 0.95)
        
        size = (df_clustered[cluster_col] == cluster_id).sum()
        # Priority score = Size * Value * (1 - Churn Risk)
        priority = (size * (est_clv / 1000.0)) * (1.0 - churn_risk)
        
        clv_estimates[cluster_id] = round(est_clv, 2)
        churn_risks[cluster_id] = round(churn_risk, 2)
        priority_scores[cluster_id] = round(priority, 1)
        
    return clv_estimates, churn_risks, priority_scores

def find_lookalike_customers(df_clustered: pd.DataFrame, target_id: str, feature_cols: list, scaler, top_n: int = 5) -> pd.DataFrame:
    """Finds nearest neighbor customers based on Euclidean distance in normalized feature space."""
    if "CustomerID" not in df_clustered.columns or target_id not in df_clustered["CustomerID"].values:
        return pd.DataFrame()
        
    target_idx = df_clustered[df_clustered["CustomerID"] == target_id].index[0]
    X_scaled = scaler.transform(df_clustered[feature_cols])
    target_vector = X_scaled[df_clustered.index.get_loc(target_idx)].reshape(1, -1)
    
    distances = euclidean_distances(X_scaled, target_vector).flatten()
    df_result = df_clustered.copy()
    df_result["Similarity_Distance"] = distances.round(3)
    
    # Exclude the exact target query customer
    lookalikes = df_result[df_result["CustomerID"] != target_id].sort_values("Similarity_Distance").head(top_n)
    return lookalikes[["CustomerID", "Cluster", "Similarity_Distance"] + feature_cols]