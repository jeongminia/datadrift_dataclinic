import streamlit as st
from pages import upload_data, data_load, base_visualization, embedding_visualization, detect_datadrift, detect_propertydrift
import warnings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
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

# PDF 캡처 기능
def capture_pdf():
    options = Options()
    options.add_argument("--headless")  # GUI 없이 실행
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--print-to-pdf")  # PDF 저장 옵션 추가

    driver = webdriver.Chrome(options=options)
    driver.get("http://localhost:8501")  # Streamlit 실행 페이지 (로컬 주소)
    time.sleep(3)  # 페이지 로딩 대기

    pdf_path = "/tmp/streamlit_dashboard.pdf"
    driver.execute_script(f'document.title="{pdf_path}"')
    driver.quit()

    return pdf_path

# PDF 저장 버튼
if st.button("Save as PDF"):
    pdf_file = capture_pdf()
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF",
            data=f,
            file_name="streamlit_dashboard.pdf",
            mime="application/pdf"
        )