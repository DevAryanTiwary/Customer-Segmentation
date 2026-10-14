import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage

def fit_clustering_model(algorithm: str, X_scaled: np.ndarray, params: dict):
    """Fits the chosen clustering model and returns cluster labels and execution time."""
    start_time = time.time()
    
    if algorithm == "K-Means":
        model = KMeans(
            n_clusters=params.get("k", 4),
            random_state=params.get("random_state", 42),
            n_init=10
        )
        labels = model.fit_predict(X_scaled)
        
    elif algorithm == "DBSCAN":
        model = DBSCAN(
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5)
        )
        labels = model.fit_predict(X_scaled)
        
    elif algorithm == "Hierarchical (Agglomerative)":
        model = AgglomerativeClustering(
            n_clusters=params.get("k", 4),
            linkage=params.get("linkage", "ward")
        )
        labels = model.fit_predict(X_scaled)
        
    elif algorithm == "Gaussian Mixture Model (GMM)":
        model = GaussianMixture(
            n_components=params.get("k", 4),
            covariance_type=params.get("cov_type", "full"),
            random_state=params.get("random_state", 42)
        )
        labels = model.fit_predict(X_scaled)
        
    elif algorithm == "HDBSCAN":
        try:
            from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
            model = SklearnHDBSCAN(
                min_cluster_size=params.get("min_cluster_size", 5),
                min_samples=params.get("min_samples", None)
            )
            labels = model.fit_predict(X_scaled)
        except ImportError:
            # Fallback to DBSCAN if scikit-learn build lacks HDBSCAN
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
            
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    duration = time.time() - start_time
    return model, labels, duration

def calculate_metrics(X_scaled: np.ndarray, labels: np.ndarray):
    """Computes cluster validity indicators (Silhouette, Calinski-Harabasz, Davies-Bouldin)."""
    unique_labels = set(labels)
    # Check if we have at least 2 non-noise clusters
    valid_labels = [l for l in unique_labels if l != -1]
    
    if len(valid_labels) < 2:
        return {"silhouette": 0.0, "calinski": 0.0, "davies": 0.0, "noise_ratio": 1.0}
    
    mask = labels != -1 if -1 in unique_labels else np.ones(len(labels), dtype=bool)
    X_sub = X_scaled[mask]
    y_sub = labels[mask]
    
    sil = silhouette_score(X_sub, y_sub)
    cal = calinski_harabasz_score(X_sub, y_sub)
    db = davies_bouldin_score(X_sub, y_sub)
    noise_ratio = float(np.mean(labels == -1)) if -1 in unique_labels else 0.0
    
    return {
        "silhouette": round(sil, 3),
        "calinski": round(cal, 1),
        "davies": round(db, 3),
        "noise_ratio": round(noise_ratio, 3)
    }

def run_model_benchmark_comparison(X_scaled: np.ndarray, k: int = 4) -> pd.DataFrame:
    """Builds a side-by-side benchmark score comparison table across all supported engines."""
    configs = [
        ("K-Means", {"k": k}),
        ("GMM", {"k": k, "cov_type": "full"}),
        ("Hierarchical", {"k": k, "linkage": "ward"}),
        ("DBSCAN", {"eps": 0.65, "min_samples": 5})
    ]
    
    rows = []
    for name, p in configs:
        algo_name = "Gaussian Mixture Model (GMM)" if name == "GMM" else ("Hierarchical (Agglomerative)" if name == "Hierarchical" else name)
        model, labels, duration = fit_clustering_model(algo_name, X_scaled, p)
        metrics = calculate_metrics(X_scaled, labels)
        
        n_clust = len(set(labels) - {-1})
        rows.append({
            "Algorithm": name,
            "Clusters": n_clust,
            "Silhouette Score (↑)": metrics["silhouette"],
            "Calinski-Harabasz (↑)": metrics["calinski"],
            "Davies-Bouldin (↓)": metrics["davies"],
            "Noise Ratio": f"{metrics['noise_ratio']*100:.1f}%",
            "Latency (ms)": round(duration * 1000, 2)
        })
        
    return pd.DataFrame(rows)

def compute_cluster_stability(X_scaled: np.ndarray, k: int = 4, iterations: int = 5) -> float:
    """Evaluates cluster stability via Adjusted Rand Index across different random initializations."""
    ari_scores = []
    base_model = KMeans(n_clusters=k, random_state=42, n_init=5)
    base_labels = base_model.fit_predict(X_scaled)
    
    for seed in range(1, iterations + 1):
        km = KMeans(n_clusters=k, random_state=42 + seed * 10, n_init=5)
        run_labels = km.fit_predict(X_scaled)
        ari = adjusted_rand_score(base_labels, run_labels)
        ari_scores.append(ari)
        
    return float(np.mean(ari_scores))