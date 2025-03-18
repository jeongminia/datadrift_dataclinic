import streamlit as st
import asyncio
from pyppeteer import launch
import threading
import os
from pages import embedding_visualization, detect_datadrift, detect_propertydrift
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
embedding_visualization.render()
detect_datadrift.render()
detect_propertydrift.render()