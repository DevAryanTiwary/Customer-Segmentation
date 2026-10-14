import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Custom modules
from modules.ui_theme import apply_custom_css, render_kpi_banner
from modules.data_engine import generate_synthetic_data, process_rfm_transactions, compute_data_quality_report
from modules.preprocessing import detect_outliers, scale_and_transform, run_pca_reduction
from modules.modeling import (
    fit_clustering_model, calculate_metrics, run_model_benchmark_comparison,
    compute_cluster_stability
)
from modules.predictive import (
    compute_cluster_centroids, predict_single_customer, compute_clv_and_churn,
    find_lookalike_customers
)
from modules.visualizations import (
    plot_clusters, plot_radar_chart, plot_parallel_coordinates, plot_segment_sunburst
)

# Page Configuration
st.set_page_config(
    page_title="Enterprise Customer Segmentation Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Session State Initialization
if "clustered_data" not in st.session_state:
    st.session_state.clustered_data = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "active_model" not in st.session_state:
    st.session_state.active_model = None
if "active_labels" not in st.session_state:
    st.session_state.active_labels = None

# ----------------- SIDEBAR CONTROLS ----------------- #
with st.sidebar:
    st.title("🎯 Control Console")
    
    # 1. Ingestion Mode
    st.subheader("1. Ingestion & Preprocessing")
    ingest_mode = st.radio(
        "Source Selection",
        ["Built-in Multi-Persona Cohort", "Upload Customer File (CSV)", "Upload Transactions (RFM)"],
        help="Switch between synthetic multi-persona data, static CSV profiles, or transactional ledger tables."
    )
    
    # File loading
    if ingest_mode == "Built-in Multi-Persona Cohort":
        df_source = generate_synthetic_data()
    elif ingest_mode == "Upload Customer File (CSV)":
        csv_file = st.file_uploader("Upload customer CSV", type=["csv"])
        if csv_file:
            df_source = pd.read_csv(csv_file)
        else:
            st.info("Awaiting CSV. Showing built-in cohort preview.")
            df_source = generate_synthetic_data()
    else:
        tx_file = st.file_uploader("Upload transactions CSV", type=["csv"])
        if tx_file:
            df_tx = pd.read_csv(tx_file)
            c1, c2, c3 = st.columns(3)
            with c1: cust_k = st.text_input("Customer ID Col", "CustomerID")
            with c2: date_k = st.text_input("Date Col", "InvoiceDate")
            with c3: amt_k = st.text_input("Amount Col", "TotalAmount")
            try:
                df_source = process_rfm_transactions(df_tx, cust_k, date_k, amt_k)
                st.success("Transformed to RFM profiles.")
            except Exception as e:
                st.error(f"Failed to process transactions: {e}")
                df_source = generate_synthetic_data()
        else:
            st.info("Awaiting transaction log. Using built-in data.")
            df_source = generate_synthetic_data()
            
    # Sample down if >10,000 records to ensure low-latency responsiveness
    if len(df_source) > 10000:
        enable_sampling = st.checkbox("Enable Fast Sampling (>10k rows detected)", value=True)
        if enable_sampling:
            df_source = df_source.sample(10000, random_state=42).reset_index(drop=True)

    # 2. Outlier Management
    st.subheader("2. Anomaly Filtering")
    enable_outliers = st.checkbox("Run Isolation Forest outlier check", value=True)
    if enable_outliers:
        contam = st.slider("Anomaly Contamination Threshold", 0.01, 0.15, 0.03, step=0.01)
    
    # 3. Model Engine Setup
    st.subheader("3. Segmentation Engine")
    algorithm_choice = st.selectbox(
        "Clustering Algorithm",
        ["K-Means", "Gaussian Mixture Model (GMM)", "Hierarchical (Agglomerative)", "DBSCAN", "HDBSCAN"]
    )
    
    # Adaptive Model Parameters
    model_params = {}
    if algorithm_choice in ["K-Means", "Gaussian Mixture Model (GMM)", "Hierarchical (Agglomerative)"]:
        model_params["k"] = st.slider("Target Clusters (k)", 2, 10, 4)
        if algorithm_choice == "Hierarchical (Agglomerative)":
            model_params["linkage"] = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
        if algorithm_choice == "Gaussian Mixture Model (GMM)":
            model_params["cov_type"] = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
    elif algorithm_choice == "DBSCAN":
        model_params["eps"] = st.slider("Epsilon (Distance)", 0.2, 2.0, 0.65, step=0.05)
        model_params["min_samples"] = st.slider("Min Density Samples", 2, 20, 5)
    elif algorithm_choice == "HDBSCAN":
        model_params["min_cluster_size"] = st.slider("Min Cluster Size", 3, 30, 8)

    st.divider()
    if st.button("🔄 Reset Parameters to Defaults"):
        st.rerun()

# ----------------- DATA PREPARATION ----------------- #
numeric_columns = df_source.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_columns) < 2:
    st.error("The dataset must contain at least 2 numerical columns for segmentation.")
    st.stop()

# Feature Selection
with st.sidebar:
    st.subheader("4. Active Dimensions")
    selected_features = st.multiselect(
        "Select clustering dimensions",
        options=numeric_columns,
        default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
    )

if len(selected_features) < 2:
    st.warning("Please choose at least 2 features from the sidebar to continue.")
    st.stop()

# Outlier separation
clean_df = df_source.copy().dropna(subset=selected_features)
if enable_outliers:
    inlier_mask = detect_outliers(clean_df, selected_features, contamination=contam)
    clean_df = clean_df[inlier_mask].reset_index(drop=True)

# Standard Scaling
X_scaled, current_scaler = scale_and_transform(clean_df, selected_features)
st.session_state.scaler = current_scaler

# Model Execution
with st.spinner("Optimizing cluster surfaces..."):
    active_model, active_labels, exec_time = fit_clustering_model(algorithm_choice, X_scaled, model_params)

clean_df["Cluster"] = ["Noise (-1)" if c == -1 else f"Cluster {c}" for c in active_labels]
st.session_state.clustered_data = clean_df
st.session_state.active_model = active_model
st.session_state.active_labels = active_labels

# Analytics & KPI Aggregations
perf_metrics = calculate_metrics(X_scaled, active_labels)
n_active_clusters = len(set(active_labels) - {-1})
top_share = (clean_df["Cluster"].value_counts().max() / len(clean_df)) * 100

clv_map, churn_map, priority_map = compute_clv_and_churn(clean_df, "Cluster", selected_features)
total_cohort_clv = sum(clv_map.get(c, 0.0) * count for c, count in clean_df["Cluster"].value_counts().items())

# ----------------- PERSISTENT KPI BANNER ----------------- #
render_kpi_banner(
    total_customers=len(clean_df),
    n_clusters=n_active_clusters,
    avg_silhouette=perf_metrics["silhouette"],
    total_clv=total_cohort_clv,
    top_segment_share=top_share
)

# ----------------- PRIMARY WORKSPACE TABS ----------------- #
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Quality & Dimensions",
    "📈 Evaluation & Model Scorecard",
    "🎯 Interactive Clusters & Flow",
    "💡 Business Personas & Action Plan",
    "🔮 Predictive What-If & Lookalike",
    "📑 Executive Report & Export"
])

# TAB 1: DATA QUALITY
with tab1:
    st.subheader("Data Profiles & Statistical Integrity")
    q_col1, q_col2 = st.columns([1, 1])
    
    dq_report = compute_data_quality_report(clean_df, selected_features)
    
    with q_col1:
        st.markdown("**Feature Correlation Matrix**")
        st.dataframe(dq_report["correlation"], width="stretch")
        
        st.markdown("**Multicollinearity (Variance Inflation Factor)**")
        st.dataframe(dq_report["vif"], width="stretch")
        
    with q_col2:
        st.markdown("**Feature Distribution Metrics**")
        st.dataframe(dq_report["describe"], width="stretch")
        
    st.divider()
    st.subheader("Dimensionality Reduction (PCA Latent Projection)")
    X_pca, var_ratio, pca_obj = run_pca_reduction(X_scaled, n_components=3)
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["Cluster"] = clean_df["Cluster"]
    
    pca_col1, pca_col2 = st.columns([2, 1])
    with pca_col1:
        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            title=f"PCA 2D Projection (Explained Variance: {sum(var_ratio[:2])*100:.1f}%)"
        )
        st.plotly_chart(fig_pca, width="stretch")
    with pca_col2:
        st.markdown("**Explained Variance per Principal Component**")
        for i, v in enumerate(var_ratio):
            st.write(f"- **PC{i+1}:** {v*100:.2f}%")
        st.info("Use PCA projections to spot cluster separation across high-dimensional feature sets.")

# TAB 2: MODEL EVALUATION & BENCHMARKS
with tab2:
    st.subheader("Hyperparameter Tuning & Multi-Algorithm Benchmark")
    
    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        # WCSS Elbow Analysis
        k_trials = list(range(2, 10))
        inertias = []
        sils = []
        for kt in k_trials:
            km = fit_clustering_model("K-Means", X_scaled, {"k": kt})[0]
            inertias.append(km.inertia_)
            sils.append(calculate_metrics(X_scaled, km.labels_)["silhouette"])
            
        fig_elbow = px.line(x=k_trials, y=inertias, markers=True, title="Elbow Method (WCSS Optimization)", labels={"x": "k (Clusters)", "y": "Inertia"})
        st.plotly_chart(fig_elbow, width="stretch")
        
    with col_bench2:
        fig_sils = px.line(x=k_trials, y=sils, markers=True, title="Silhouette Metric Sweep", labels={"x": "k (Clusters)", "y": "Silhouette Score"}, color_discrete_sequence=["#FF9800"])
        st.plotly_chart(fig_sils, width="stretch")
        
    st.subheader("Side-by-Side Model Scorecard")
    comparison_df = run_model_benchmark_comparison(X_scaled, k=model_params.get("k", 4))
    st.dataframe(comparison_df, width="stretch")
    
    # Stability test
    stability_ari = compute_cluster_stability(X_scaled, k=model_params.get("k", 4))
    st.markdown(f"**Cluster Random Initialization Stability (Adjusted Rand Index):** `{stability_ari:.3f}` (1.0 = deterministic)")

    # Optional Dendrogram
    if algorithm_choice == "Hierarchical (Agglomerative)":
        with st.expander("Hierarchical Dendrogram View"):
            fig_d, ax = plt.subplots(figsize=(10, 4))
            Z = linkage(X_scaled, method=model_params.get("linkage", "ward"))
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=30)
            ax.set_title("Truncated Dendrogram Tree")
            ax.set_ylabel("Distance")
            st.pyplot(fig_d)

# TAB 3: VISUAL CLUSTERS & MULTIDIMENSIONAL FLOW
with tab3:
    st.subheader(f"Active Cluster Distribution ({algorithm_choice})")
    
    col_vis1, col_vis2 = st.columns([3, 2])
    with col_vis1:
        fig_scatter = plot_clusters(clean_df, selected_features, "Cluster", f"{algorithm_choice} Cluster Plot")
        st.plotly_chart(fig_scatter, width="stretch")
    with col_vis2:
        fig_radar = plot_radar_chart(clean_df, "Cluster", selected_features)
        st.plotly_chart(fig_radar, width="stretch")
        
    st.subheader("Parallel Coordinates Feature Flow")
    fig_par = plot_parallel_coordinates(clean_df, "Cluster", selected_features)
    st.plotly_chart(fig_par, width="stretch")

# TAB 4: PERSONAS & ACTION PLAYBOOK
with tab4:
    st.subheader("Cohort Personas, Churn Probability & Strategic Playbooks")
    
    summary_stats = clean_df.groupby("Cluster")[selected_features].mean().round(2)
    summary_stats["Cohort_Size"] = clean_df["Cluster"].value_counts()
    
    p_col1, p_col2 = st.columns([1, 1])
    with p_col1:
        st.markdown("#### Segment Quantitative Summary")
        st.dataframe(summary_stats, width="stretch")
    with p_col2:
        val_feature = "Monetary_Total" if "Monetary_Total" in clean_df.columns else None
        fig_sun = plot_segment_sunburst(clean_df, "Cluster", val_feature)
        st.plotly_chart(fig_sun, width="stretch")
        
    st.markdown("#### Strategic Marketing Playbooks")
    for cluster_id in sorted(clean_df["Cluster"].unique()):
        if cluster_id == "Noise (-1)":
            st.markdown("""
                <div class="persona-card" style="border-left-color: #E53935;">
                    <h4 style="color: #E53935; margin: 0 0 8px 0;">Anomaly / Noise Group (-1)</h4>
                    <p>Profiles identified as mathematical outliers. Exclude from automated recurring discount drops. Flag for compliance checks or bespoke executive VIP account management.</p>
                </div>
            """, unsafe_allow_html=True)
            continue
            
        c_stats = summary_stats.loc[cluster_id]
        c_clv = clv_map.get(cluster_id, 0.0)
        c_churn = churn_map.get(cluster_id, 0.0)
        c_prio = priority_map.get(cluster_id, 0.0)
        
        st.markdown(f"""
            <div class="persona-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #1E88E5;">{cluster_id} (Size: {int(c_stats['Cohort_Size'])} accounts)</h4>
                    <span style="font-weight: 700; color: #333;">Priority Index: {c_prio}</span>
                </div>
                <div style="display: flex; gap: 24px; font-size: 0.88rem; color: #666; margin-bottom: 12px;">
                    <span><b>Projected CLV:</b> ${c_clv:,.2f}</span>
                    <span><b>Estimated Churn Risk:</b> {c_churn*100:.1f}%</span>
                </div>
                <p><b>Recommended Strategic Playbook:</b> Apply tailored retention triggers, early-access features, and personalized multi-channel campaigns based on this cluster's attribute profile.</p>
            </div>
        """, unsafe_allow_html=True)

# TAB 5: PREDICTIVE WHAT-IF & LOOKALIKE DISCOVERY
with tab5:
    st.subheader("What-If Customer Inference & Nearest Lookalikes")
    
    col_pred1, col_pred2 = st.columns([1, 1])
    
    with col_pred1:
        st.markdown("#### Real-time What-If Profiler")
        input_payload = {}
        for feat in selected_features:
            default_val = float(clean_df[feat].mean())
            min_val = float(clean_df[feat].min())
            max_val = float(clean_df[feat].max())
            input_payload[feat] = st.slider(f"Simulate {feat}", min_val, max_val, default_val)
            
        centroids = compute_cluster_centroids(X_scaled, active_labels)
        if centroids:
            best_c, conf, dist = predict_single_customer(input_payload, selected_features, current_scaler, centroids)
            st.success(f"Assigned Segment: **Cluster {best_c}**")
            st.metric("Assignment Confidence", f"{conf}%", f"Centroid Distance: {dist}")
            
    with col_pred2:
        st.markdown("#### Lookalike Search Engine")
        if "CustomerID" in clean_df.columns:
            target_cid = st.selectbox("Select Target Customer ID", clean_df["CustomerID"].values[:200])
            top_lookalikes = find_lookalike_customers(clean_df, target_cid, selected_features, current_scaler, top_n=5)
            st.dataframe(top_lookalikes, width="stretch")
        else:
            st.info("Dataset requires a 'CustomerID' column to run Lookalike similarity matching.")

# TAB 6: EXPORT & GOVERNANCE
with tab6:
    st.subheader("Cohort Data Export & Model Persistence")
    
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.markdown("#### Download Segmented Data")
        csv_buffer = io.StringIO()
        clean_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Clustered Dataset (CSV)",
            data=csv_buffer.getvalue(),
            file_name="customer_segments_export.csv",
            mime="text/csv",
            width="stretch"
        )
        
    with col_ex2:
        st.markdown("#### Save Active Pipeline Artifacts")
        if st.button("Serialize Model & Scaler (.joblib)", width="stretch"):
            artifact = {
                "algorithm": algorithm_choice,
                "model": active_model,
                "scaler": current_scaler,
                "features": selected_features
            }
            joblib.dump(artifact, "segmentation_pipeline.joblib")
            st.success("Saved model artifact to `segmentation_pipeline.joblib`.")