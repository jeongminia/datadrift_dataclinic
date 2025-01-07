import streamlit as st
from utils import upload_and_store_data

def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Upload your datasets : {dataset_name}")
    
    st.subheader("Text Data Drift Detection for Classification Tasks!")
    st.write("Please upload your train, validation, and test datasets.")
    upload_and_store_data()