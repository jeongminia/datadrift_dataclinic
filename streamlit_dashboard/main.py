import streamlit as st
from pages import data_load, base_visualization, embedding_visualization, detect_datadrift

# 페이지 설정
st.set_page_config(
    page_title="Data Visualization and Drift Detection",  # 앱의 제목
    page_icon="📊",  # 아이콘
    layout="wide"  # 레이아웃: wide 또는 centered
)

# 페이지 구성
st.sidebar.title("Navigation")  # 사이드바 제목 설정
tab1, tab2, tab3, tab4 = st.tabs(["Data Load", "Basic Visualization", "Embedding Visualization", "Detect Data Drift"])


# 탭별로 해당 파일의 함수를 호출
with tab1:
    data_load.render()

with tab2:
    base_visualization.render()

with tab3:
    embedding_visualization.render()

with tab4:
    detect_datadrift.render()