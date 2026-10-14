import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

def detect_outliers(df: pd.DataFrame, feature_cols: list, method: str = "Isolation Forest", contamination: float = 0.05) -> pd.Series:
    """Flags outliers using Isolation Forest or Local Outlier Factor."""
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    if method == "Isolation Forest":
        detector = IsolationForest(contamination=contamination, random_state=42)
        pred = detector.fit_predict(X)
    else:
        detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        pred = detector.fit_predict(X)
        
    # Return boolean mask (True = Inlier, False = Outlier)
    return pd.Series(pred == 1, index=df.index, name="is_inlier")

def scale_and_transform(df: pd.DataFrame, feature_cols: list):
    """Standardizes features to zero-mean and unit-variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler

def run_pca_reduction(X_scaled: np.ndarray, n_components: int = 3):
    """Runs PCA for latent space reduction and variance analysis."""
    max_c = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=max_c, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    return X_pca, var_ratio, pca