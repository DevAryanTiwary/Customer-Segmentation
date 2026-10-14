import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Base typography & clean executive palette */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            backdrop-filter: blur(10px);
        }
        
        .metric-card-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #888888;
            margin-bottom: 6px;
            font-weight: 600;
        }
        
        .metric-card-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #1E88E5;
            margin-bottom: 4px;
        }
        
        .metric-card-sub {
            font-size: 0.78rem;
            color: #4CAF50;
            font-weight: 500;
        }

        /* Persona Card Styling */
        .persona-card {
            border-left: 4px solid #1E88E5;
            background: rgba(30, 136, 229, 0.04);
            border-radius: 0 10px 10px 0;
            padding: 18px;
            margin-bottom: 16px;
            border-top: 1px solid rgba(128, 128, 128, 0.1);
            border-right: 1px solid rgba(128, 128, 128, 0.1);
            border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        }

        /* Streamlit Tab polish */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            border-radius: 6px 6px 0 0;
            font-weight: 500;
            padding: 8px 16px;
        }

        /* Sidebar cleanup */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }
        </style>
    """, unsafe_allow_html=True)

def render_kpi_banner(total_customers, n_clusters, avg_silhouette, total_clv, top_segment_share):
    """Renders persistent executive KPI cards across the top viewport."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Cohort</div>
                <div class="metric-card-value">{total_customers:,}</div>
                <div class="metric-card-sub">Active profiles analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Cluster Segments</div>
                <div class="metric-card-value">{n_clusters}</div>
                <div class="metric-card-sub">Discrete behavioral cohorts</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        sil_color = "#4CAF50" if avg_silhouette >= 0.5 else ("#FF9800" if avg_silhouette >= 0.3 else "#E53935")
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Mean Silhouette</div>
                <div class="metric-card-value" style="color: {sil_color};">{avg_silhouette:.3f}</div>
                <div class="metric-card-sub">Cluster separation quality</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Cohort Estimated CLV</div>
                <div class="metric-card-value">${total_clv:,.0f}</div>
                <div class="metric-card-sub">Aggregate projected equity</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Dominant Cluster Share</div>
                <div class="metric-card-value">{top_segment_share:.1f}%</div>
                <div class="metric-card-sub">Concentration risk</div>
            </div>
        """, unsafe_allow_html=True)