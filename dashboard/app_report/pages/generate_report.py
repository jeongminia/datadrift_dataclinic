import os
import re
import pdfkit
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
from datetime import datetime
from ..assets.make_html import database_html, drift_html # report_database
from ..assets.design_html import head_footer_html # report_html
#from ..assets.llm_report import llm_html # report_llm

# ------------------------------------- HTML 관련 유틸리티 -------------------------------------
def get_html_body(html):
    # HTML에서 <body> 태그만 추출하고 h1 태그 제거
    if not html:
        return ''
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find('body')
        if body:
            for h1 in body.find_all('h1'):
                h1.decompose()
            return str(body)
        else:
            return str(soup)
    else:
        return re.sub(r'<h1[^>]*>.*?</h1>', '', html, flags=re.DOTALL)

def get_cached_html(cache_key, generator_func, *args, **kwargs):
    # 캐시된 HTML 가져오기 또는 생성하기
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    html = generator_func(*args, **kwargs)
    body = get_html_body(html)
    st.session_state[cache_key] = body
    return body

# ----------------------------------- integrate ----------------------------------
def final_report(dataset_name): 
    # 통합 리포트 생성
    db_cache_key = f"db_html_{dataset_name}"
    drift_cache_key = f"drift_html_{dataset_name}" 
    llm_cache_key = f"llm_html_{dataset_name}"

    database_content = get_cached_html(db_cache_key, database_html, dataset_name)    
    drift_content = get_cached_html(drift_cache_key, drift_html, dataset_name)
    #llm_content = get_cached_html(llm_cache_key, llm_html, dataset_name)
    llm_content = "<p>LLM content is not available.</p>" 
    
    return head_footer_html(dataset_name, database_content, drift_content, llm_content)

# ------------------------------------- main -------------------------------------
def render():
    """통합 리포트 생성 페이지"""
    st.markdown("""
            <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
                        padding: 15px 25px; border-radius: 15px; margin-bottom: 25px;
                        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        transform: scale(0.95);">
                <div style="text-align: center;">
                    <h3 style="color: white; margin: 0; font-weight: 700; font-size: 18px; 
                            text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        🚀 Integrated Report Generation
                    </h3>
                    <div style="color: rgba(255, 255, 255, 0.95); font-size: 12px; margin-top: 8px;
                            text-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                        Database Pipeline과 Drift Analysis 결과를 통합하여 최종 리포트를 생성합니다
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 필요한 데이터 확인
    selected_dataset = st.session_state.get('selected_dataset')
    if not selected_dataset:
        st.warning("⚠️ 먼저 Load Results에서 데이터셋을 선택해주세요.")
        return
    
    # LLM 설정 확인
    model_configured = st.session_state.get('model_configured', False)
    if not model_configured:
        st.warning("⚠️ 먼저 Build Custom LLM에서 AI 모델을 설정해주세요.")
        return
    
    st.success(f"📊 선택된 데이터셋: **{selected_dataset}**")
    st.success(f"🤖 AI 모델: **{st.session_state.get('model_name', 'N/A')}**")

    dataset_name = st.session_state.get('dataset_name')

    db_html = st.session_state.database_html
    drift_html = st.session_state.drift_html

    selected_model = st.session_state.get('selected_model')
    temperature = st.session_state.get('model_temperature')
    max_tokens = st.session_state.get('max_tokens')
    top_p = st.session_state.get('top_p')
    custom_prompt = st.session_state.get('custom_prompt')


    html_content = final_report(dataset_name)

    pdf_bytes = pdfkit.from_string(html_content, False, options={
        'page-size': 'A4',
        'margin-top': '0.5in',
        'margin-right': '0.5in',
        'margin-bottom': '0.5in',
        'margin-left': '0.5in',
        'encoding': "UTF-8",
        'enable-local-file-access': ''
    })

    st.download_button(
        label="📄 PDF 다운로드",
        data=pdf_bytes,
        file_name=f"{dataset_name}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        key=f"pdf_dl_{dataset_name}"
    )
    