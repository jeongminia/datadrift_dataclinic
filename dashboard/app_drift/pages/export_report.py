import streamlit as st
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

def save_html_to_pdf_via_browser(html_path, pdf_path):
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')  # 안정적인 헤드리스 모드
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1280,1024')
    chrome_options.add_argument(f'--print-to-pdf={os.path.abspath(pdf_path)}')

    try:
        # 만약 chromedriver 경로를 직접 지정해야 하면 아래 주석 해제 후 경로 설정
        driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver', options=chrome_options)
        # driver = webdriver.Chrome(options=chrome_options)

        file_url = f"file://{os.path.abspath(html_path)}"
        st.info("📤 Chrome을 통해 PDF로 변환 중입니다...")
        driver.get(file_url)

        # 페이지가 모두 렌더링될 시간을 확보
        time.sleep(3)
        driver.quit()
    except WebDriverException as e:
        st.error(f"❌ ChromeDriver 실행 실패: {e}")
        return False
    return True

def render():
    st.write("📄 Export Final PDF Report")

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    html_path = f"./reports/{dataset_name} train_test_drift_report.html"
    pdf_path = html_path.replace(".html", ".pdf")

    if not os.path.exists(html_path):
        st.error(f"❌ HTML report file not found: {html_path}")
        return

    success = save_html_to_pdf_via_browser(html_path, pdf_path)

    if success and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Final PDF Report",
                data=f,
                file_name=f"{dataset_name}_drift_report.pdf",
                mime="application/pdf"
            )
        st.success("✅ PDF successfully generated from HTML using Chrome!")
    else:
        st.error("🚨 PDF 생성에 실패했습니다. ChromeDriver 설치 및 호환성 확인이 필요합니다.")
