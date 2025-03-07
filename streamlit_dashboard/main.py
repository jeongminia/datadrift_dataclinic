import streamlit as st
from pages import upload_data, data_load, base_visualization, embedding_visualization, detect_datadrift, detect_propertydrift
import warnings
import pdfkit
import os
warnings.filterwarnings(action='ignore')

st.set_page_config(
    page_title="Embedding Drift Detection",  
    page_icon="📊", 
    layout="wide" ,
    initial_sidebar_state="collapsed"
)

upload_data.render()

data_load.render()
base_visualization.render()
embedding_visualization.render()
detect_datadrift.render()
detect_propertydrift.render()

# PDF
if st.button("Save as PDF"):
    # HTML 파일 경로
    html_file_path = "/tmp/streamlit_page.html"
    pdf_file_path = "/tmp/streamlit_page.pdf"
    
    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
    
    with open(html_file_path, "w") as f:
        f.write(st._get_page_html())
    
    config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
    pdfkit.from_file(html_file_path, pdf_file_path, configuration=config)
    
    with open(pdf_file_path, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name="streamlit_page.pdf",
            mime="application/pdf"
        )