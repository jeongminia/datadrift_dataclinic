import streamlit as st
import asyncio
from pyppeteer import launch
import threading
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

if "data_uploaded" not in st.session_state:
    st.session_state["data_uploaded"] = False

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

    await page.reload()
    await asyncio.sleep(15)  
    await page.reload()  
    await asyncio.sleep(5)

    await page.waitForSelector("div[data-testid='stVerticalBlock']", timeout=20000)  
    await page.waitForSelector("section[data-testid='stSidebar']", timeout=20000) 
    await page.waitForSelector("div.stButton", timeout=20000) 
    await asyncio.sleep(5)

    pdf_path = "/tmp/streamlit_dashboard.pdf"
    await page.pdf({'path': pdf_path, 'format': 'A4', 'printBackground': True})  # 배경 포함하여 PDF 생성
    await browser.close()
    return pdf_path

async def get_pdf():
    return await generate_pdf()

if st.session_state["data_uploaded"]:
    if st.button("Save as PDF"):
        with st.spinner("📄 PDF를 생성하는 중... 잠시만 기다려 주세요."):
            pdf_file = asyncio.run(get_pdf())  

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download PDF",
                data=f,
                file_name="streamlit_dashboard.pdf",
                mime="application/pdf"
            )
else:
    st.warning("⚠ 데이터가 업로드되지 않았습니다. 먼저 데이터를 업로드하세요.")