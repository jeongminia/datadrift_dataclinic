import streamlit as st
import base64
import pandas as pd
import os
from datetime import datetime
import pdfkit
from ..assets.report_layout import integrated_report

def render():    
    html_content = integrated_report()

    pdf_bytes = pdfkit.from_string(html_content, False, options={
        'page-size': 'A4',
        'margin-top': '0.5in',
        'margin-right': '0.5in',
        'margin-bottom': '0.5in',
        'margin-left': '0.5in',
        'encoding': "UTF-8",
        'enable-local-file-access': ''
    })

    dataset_name = st.session_state.get('dataset_name')
    db_html = st.session_state.database_html
    drift_html = st.session_state.drift_html

    selected_model = st.session_state.get('selected_model')
    temperature = st.session_state.get('model_temperature')
    max_tokens = st.session_state.get('max_tokens')
    top_p = st.session_state.get('top_p')
    custom_prompt = st.session_state.get('custom_prompt')
    '''
    1. LLM 설정에 집어넣은 후, 호출해 답변을 임시 저장하기
    2. HTML 두가지 가져오기
    3. 세가지 병합하여 리포트 반환하기
    '''







    st.download_button(
        label="📄 PDF 다운로드",
        data=pdf_bytes,
        file_name=f"{dataset_name}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        key=f"pdf_dl_{dataset_name}"
    )
    