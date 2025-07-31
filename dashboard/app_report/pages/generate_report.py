import streamlit as st
import base64
import pandas as pd
import os
from datetime import datetime
import pdfkit
from app_report.assets.report_layout import integrated_report

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
    
    # 리포트 생성
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
    