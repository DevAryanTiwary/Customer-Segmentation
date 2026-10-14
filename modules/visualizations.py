import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLOR_PALETTE = ["#1E88E5", "#FFC107", "#004D40", "#D81B60", "#8E24AA", "#00ACC1", "#3949AB"]

def plot_clusters(df: pd.DataFrame, feature_cols: list, cluster_col: str, title: str):
    """Renders 3D Plotly scatter if ≥3 features are present, else 2D."""
    if len(feature_cols) >= 3:
        fig = px.scatter_3d(
            df,
            x=feature_cols[0],
            y=feature_cols[1],
            z=feature_cols[2],
            color=cluster_col,
            hover_name="CustomerID" if "CustomerID" in df.columns else None,
            hover_data=feature_cols,
            color_discrete_sequence=COLOR_PALETTE,
            title=title
        )
        fig.update_layout(scene=dict(
            xaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)"),
            yaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)"),
            zaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)")
        ))
    else:
        fig = px.scatter(
            df,
            x=feature_cols[0],
            y=feature_cols[1],
            color=cluster_col,
            hover_name="CustomerID" if "CustomerID" in df.columns else None,
            hover_data=feature_cols,
            color_discrete_sequence=COLOR_PALETTE,
            title=title
        )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig

def plot_radar_chart(df: pd.DataFrame, cluster_col: str, feature_cols: list):
    """Generates an executive radar comparison chart normalized per feature."""
    summary = df.groupby(cluster_col)[feature_cols].mean()
    
    # Min-Max normalize values for standard radial axis scaling
    norm_summary = (summary - summary.min()) / (summary.max() - summary.min() + 1e-6)
    
    fig = go.Figure()
    categories = list(feature_cols)
    categories.append(categories[0])  # Close the radar loop
    
    for idx, (cluster_id, row) in enumerate(norm_summary.iterrows()):
        values = row.tolist()
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=str(cluster_id),
            line=dict(color=COLOR_PALETTE[idx % len(COLOR_PALETTE)])
        ))
        
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Segment Trait Normalization (Radar Profile)",
        margin=dict(l=40, r=40, b=20, t=40)
    )
    return fig

def plot_parallel_coordinates(df: pd.DataFrame, cluster_col: str, feature_cols: list):
    """Renders parallel coordinates to inspect multidimensional separation."""
    sample_df = df.copy()
    if len(sample_df) > 1000:
        sample_df = sample_df.sample(1000, random_state=42)
        
    # Map cluster names to distinct integer codes for color mapping
    sample_df["Cluster_Code"] = sample_df[cluster_col].astype('category').cat.codes
    
    dimensions = []
    for col in feature_cols:
        dimensions.append(dict(
            range=[sample_df[col].min(), sample_df[col].max()],
            label=col,
            values=sample_df[col]
        ))
        
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=sample_df["Cluster_Code"],
            colorscale='Viridis',
            showscale=False
        ),
        dimensions=dimensions
    ))
    fig.update_layout(title="Multi-Attribute Parallel Flow", margin=dict(l=40, r=40, b=20, t=40))
    return fig

def plot_segment_sunburst(df: pd.DataFrame, cluster_col: str, value_col: str = None):
    """Renders a sunburst or treemap breakdown of customer cohort capital distribution."""
    if value_col and value_col in df.columns:
        fig = px.treemap(
            df,
            path=[cluster_col],
            values=value_col,
            color=cluster_col,
            color_discrete_sequence=COLOR_PALETTE,
            title="Capital Concentration by Segment (Treemap)"
        )
    else:
        fig = px.pie(
            df,
            names=cluster_col,
            title="Customer Cohort Distribution",
            color_discrete_sequence=COLOR_PALETTE,
            hole=0.45
        )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig