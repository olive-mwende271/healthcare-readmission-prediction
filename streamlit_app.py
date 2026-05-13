import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

st.set_page_config(page_title="Healthcare System", layout="wide")
st.title("🏥 Large Scale Healthcare Analytics System")
st.subheader("Milestones 1 - 6 | Diabetes Readmission Prediction")

tabs = st.tabs(["📋 Home", "📊 Milestone 1-2: Data & Sharding", 
                "🔮 Prediction", "🔒 Milestone 3: Federated Learning", 
                "📈 Results & Evaluation"])

with tabs[0]:
    st.success("✅ Full Pipeline Deployed on Streamlit")
    st.write("This app showcases your entire notebook work: Ingestion → Cleaning → Sharding → Federated Learning → Prediction")

with tabs[1]:  # Milestone 1-2
    st.header("Milestone 1 & 2: Data Ingestion & Distributed Processing")
    st.write("**Key Achievements:**")
    st.markdown("""
    - Loaded **101,766 records**
    - Cleaned missing values (`?` → None)
    - Removed high-missing columns (e.g. weight)
    - Deduplicated by patient
    - Partitioned by `admission_type_id`
    """)
    
    uploaded_file = st.file_uploader("Upload diabetic_data.csv to explore", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(df, x="age", color="readmitted", title="Readmission by Age"))
        with col2:
            st.plotly_chart(px.bar(df.groupby("admission_type_id").size(), title="Records per Admission Type"))

with tabs[2]:  # Prediction
    st.header("Real-time Prediction")
    # ... (same prediction code as before)

with tabs[3]:  # Federated Learning
    st.header("Milestone 3: Federated Learning")
    st.info("Privacy-preserving training across multiple hospitals (No raw data shared)")
    
    rounds = [1,2,3,4,5]
    auc_scores = [0.58, 0.615, 0.645, 0.662, 0.674]
    
    fig = px.line(x=rounds, y=auc_scores, markers=True, 
                  title="Federated Learning Model Improvement", 
                  labels={"x": "Round", "y": "ROC-AUC"})
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Federated AUC", "0.6740")
    with col2:
        st.metric("Privacy Cost", "≈ 8.8% drop vs Centralised")

with tabs[4]:  # Final Evaluation
    st.header("Milestone 6: Final System Evaluation")
    st.success("✅ All Milestones Completed Successfully!")
    
    metrics = {
        "Total Records": "101,766",
        "Best Centralised AUC": "0.7623",
        "Federated AUC": "0.6740",
        "Recall": "63.9% @ 0.35 threshold",
        "Pipeline Speedup": "7.1x",
        "Streaming Latency": "~200ms"
    }
    
    for label, value in metrics.items():
        st.metric(label, value)
    
    st.write("**Key Innovations:** Spark Sharding + Federated Learning + Lambda Architecture")
