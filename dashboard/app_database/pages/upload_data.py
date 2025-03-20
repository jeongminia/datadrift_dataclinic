import streamlit as st
from utils import upload_and_store_data

def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Upload your datasets : {dataset_name}")
    st.session_state['dataset_name'] = dataset_name
    
    data_type = ['Text', 'Image', 'Tabular']
    selected_data_type = st.selectbox("1. Choose data type:", 
                                                 data_type)
    task_type = ['Classification', 'Regression']
    selected_task_type = st.selectbox("2. Choose task type:",
                                                 task_type)
    st.subheader(f"{selected_data_type} Data Drift Detection for {selected_task_type}!")
    st.write("Please upload your train, validation, and test datasets.")
    
    uploaded = upload_and_store_data()