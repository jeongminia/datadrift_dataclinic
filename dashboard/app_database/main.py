import streamlit as st
import os
from pages import upload_data, data_load, base_visualization, vector_database
import warnings
warnings.filterwarnings(action='ignore')

st.set_page_config(
    page_title="Embedding Drift Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "data_uploaded" not in st.session_state:
    st.session_state["data_uploaded"] = False

# 페이지 렌더링
upload_data.render()
data_load.render()
base_visualization.render()
vector_database.render()