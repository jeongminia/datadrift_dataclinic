# 통합 리포트 생성 함수들
import streamlit as st
import base64
import pandas as pd
import os
from datetime import datetime
from report_layout import check_drift_analysis_complete, generate_combined_html
import pdfkit

def render_combined_report(database_export_report=None, drift_export_report=None):
    """최적화된 통합 리포트 렌더링 - rerun 최소화"""

    # 상태 체크
    has_database = database_export_report is not None
    has_drift = check_drift_analysis_complete()
    dataset_name = st.session_state.get('dataset_name', 'Dataset')

    # 상태 표시 (컴팩트)
    col1, col2 = st.columns(2)
    with col1:
        status = "✅ 준비됨" if has_database else "⏳ 대기중"
        st.write(f"**Database:** {status}")
    with col2:
        status = "✅ 완료됨" if has_drift else "⏳ 대기중"
        st.write(f"**Drift Analysis:** {status}")

    # 리포트 생성 가능 여부
    can_generate = has_database or has_drift

    if not can_generate:
        st.info("💡 Database Pipeline 또는 Drift Analysis를 먼저 완료해주세요.")
        return

    # HTML 콘텐츠 생성
    html_content = generate_combined_html(database_export_report, drift_export_report)

    # PDF 변환 및 다운로드 버튼
    if pdfkit:
        try:
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
        except Exception as e:
            st.error(f"리포트 생성 중 오류: {e}")
    else:
        st.error("pdfkit 모듈이 설치되어 있지 않습니다. PDF 변환을 위해 pdfkit을 설치해주세요.")