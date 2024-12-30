import streamlit as st
from pages import upload_data, data_load, base_visualization, embedding_visualization, detect_datadrift
import warnings
warnings.filterwarnings(action='ignore')

# 페이지 설정
st.set_page_config(
    page_title="Data Visualization and Drift Detection",  
    page_icon="📊", 
    layout="wide" ,
    initial_sidebar_state="collapsed"
)

# 사이드바를 강제로 비우기
st.sidebar.empty()

# 페이지 구성
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Upload Data", "Data Load", 
                                        "Basic Visualization", "Embedding Visualization", 
                                        "Detect DataDrift"])


# 탭별로 해당 파일의 함수를 호출
with tab0:
    upload_data.render()

with tab1:
    data_load.render()

with tab2:
    base_visualization.render()

with tab3:
    embedding_visualization.render()

with tab4:
    detect_datadrift.render()