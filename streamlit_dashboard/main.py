import streamlit as st
from pages import upload_data, data_load, base_visualization, embedding_visualization, detect_datadrift, detect_propertydrift
import warnings
import pdfkit
import os
warnings.filterwarnings(action='ignore')

# 페이지 설정
st.set_page_config(
    page_title="Embedding Drift Detection",  
    page_icon="📊", 
    layout="wide" ,
    initial_sidebar_state="collapsed"
)

#st.sidebar.empty()

#st.header("Upload Data")
upload_data.render()

#st.header("Data Load")
data_load.render()

#st.header("Basic Visualization")
base_visualization.render()

#st.header("Embedding Visualization")
embedding_visualization.render()

#st.header("🔴 Detect DataDrift")
detect_datadrift.render()

#st.header("🔴 Detect PropertyDrift")
detect_propertydrift.render()

# PDF로 저장하는 버튼 추가
if st.button("Save as PDF"):
    # HTML 파일 경로
    html_file_path = "/tmp/streamlit_page.html"
    pdf_file_path = "/tmp/streamlit_page.pdf"
    
    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
    
    # 현재 페이지의 HTML을 저장
    with open(html_file_path, "w") as f:
        f.write(st._get_page_html())
    
    # HTML 파일을 PDF로 변환
    config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
    pdfkit.from_file(html_file_path, pdf_file_path, configuration=config)
    
    # PDF 파일 다운로드 링크 제공
    with open(pdf_file_path, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name="streamlit_page.pdf",
            mime="application/pdf"
        )