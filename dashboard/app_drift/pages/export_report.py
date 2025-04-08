import streamlit as st
import os
import pdfkit
from bs4 import BeautifulSoup  # 💡 HTML 정제용

def generate_html_from_session(dataset_name):
    html_parts = []

    html_parts.append(f"<h1>{dataset_name} Drift Report</h1>")

    if 'train_embeddings' in st.session_state:
        shape = st.session_state['train_embeddings'].shape
        html_parts.append(f"<p><b>Train Embedding Shape:</b> {shape}</p>")

    if 'valid_embeddings' in st.session_state:
        shape = st.session_state['valid_embeddings'].shape
        html_parts.append(f"<p><b>Valid Embedding Shape:</b> {shape}</p>")

    if 'test_embeddings' in st.session_state:
        shape = st.session_state['test_embeddings'].shape
        html_parts.append(f"<p><b>Test Embedding Shape:</b> {shape}</p>")

    # Evidently Drift Report 삽입
    if 'train_test_drift_report_html' in st.session_state:
        html_parts.append("<hr><h2>Drift Report</h2>")

        # 🧼 BeautifulSoup으로 body 안의 내용만 추출
        soup = BeautifulSoup(st.session_state['train_test_drift_report_html'], "html.parser")
        drift_body = soup.body or soup  # body가 없으면 전체 사용
        html_parts.append(str(drift_body))

    # 전체 HTML 템플릿
    html_template = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; }}
            table, th, td {{ border: 1px solid #ccc; padding: 8px; }}
        </style>
    </head>
    <body>
        {''.join(html_parts)}
    </body>
    </html>
    """
    return html_template


def render():
    st.title("📄 Export Final PDF Report")

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    html_output_path = f"./reports/{dataset_name}_compiled_report.html"
    pdf_output_path = html_output_path.replace(".html", ".pdf")

    final_html = generate_html_from_session(dataset_name)
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    try:
        config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')
        pdfkit.from_file(html_output_path, pdf_output_path, configuration=config)

        with open(pdf_output_path, "rb") as f:
            st.download_button(
                label="📥 Download Final PDF Report",
                data=f,
                file_name=f"{dataset_name}_final_report.pdf",
                mime="application/pdf"
            )
        st.success("✅ PDF successfully generated from session data!")
    except Exception as e:
        st.error(f"❌ PDF 변환 실패: {e}")
