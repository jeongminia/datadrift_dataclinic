import streamlit as st
from utils import upload_and_store_data

def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Upload your datasets : {dataset_name}")
    
    st.subheader("Text Data Drift Detection for Classification Tasks!")
    drift_detection_types = ["Text Data Drift", "Numerical Data Drift", "Categorical Data Drift"]
    selected_drift_detection_type = st.selectbox("Choose the type of data drift detection you want to perform:", drift_detection_types)

    st.write("Please upload your train, validation, and test datasets.")
    upload_and_store_data()