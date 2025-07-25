import streamlit as st
import base64
import pandas as pd
import os
from datetime import datetime
import pdfkit

from assets.llms_settings import custom_llm
from assets.report_layout import integrated_report


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

    dataset_name = st.session_state.dataset_name
    st.download_button(
        label="📄 PDF 다운로드",
        data=pdf_bytes,
        file_name=f"{dataset_name}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        key=f"pdf_dl_{dataset_name}"
    )
    