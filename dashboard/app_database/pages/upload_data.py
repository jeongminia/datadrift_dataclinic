import streamlit as st

# Import utils from parent directory
try:
    from ..utils import upload_and_store_data
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import upload_and_store_data

def render():
    dataset_name = st.session_state.get('dataset_name')
    st.subheader(f"Upload your datasets : {dataset_name}")
    st.session_state['dataset_name'] = dataset_name
    
    data_type = ['Text', 'Image', 'Tabular']
    selected_data_type = st.selectbox("Choose data type:", 
                                                 data_type)
    task_type = ['Classification', 'Regression']
    selected_task_type = st.selectbox("Choose task type:",
                                                 task_type)
    st.subheader(f"{selected_data_type} Data Drift Detection for {selected_task_type}!")
    st.write("Please upload your train, validation, and test datasets.")
    
    uploaded = upload_and_store_data()
    dataset_name = st.session_state['dataset_name']