import streamlit as st
import asyncio
from pyppeteer import launch
import os
from pages import upload_data, data_load, base_visualization, embedding_visualization, detect_datadrift, detect_propertydrift
import warnings
warnings.filterwarnings(action='ignore')

st.set_page_config(
    page_title="Embedding Drift Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 페이지 렌더링
upload_data.render()
data_load.render()
base_visualization.render()
embedding_visualization.render()
detect_datadrift.render()
detect_propertydrift.render()

# HTML 저장 및 PDF 변환 함수
async def generate_pdf():
    browser = await launch(
        headless=True, 
        args=['--no-sandbox'],
        handleSIGINT=False, 
        handleSIGTERM=False, 
        handleSIGHUP=False  # 🚨 시그널 핸들링 비활성화
    )
    page = await browser.newPage()
    await page.goto("http://localhost:8501", {'waitUntil': 'networkidle2'})  # Streamlit 서버 URL
    pdf_path = "/tmp/streamlit_dashboard.pdf"
    await page.pdf({'path': pdf_path, 'format': 'A4'})  # PDF로 저장
    await browser.close()
    return pdf_path

def run_asyncio_task(task):
    try:
        loop = asyncio.get_running_loop() 
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(task)
    else:
        return loop.run_until_complete(task)

# PDF 저장 버튼
if st.button("Save as PDF"):
    pdf_file = run_asyncio_task(generate_pdf())

    # PDF 다운로드 버튼 추가
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF",
            data=f,
            file_name="streamlit_dashboard.pdf",
            mime="application/pdf"
        )