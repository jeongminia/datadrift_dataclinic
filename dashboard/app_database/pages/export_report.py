import pandas as pd
import pdfkit
import streamlit as st
import os
from utils import gen_summarization, gen_explanation


def generate_html_from_session(dataset_name):
    html_parts = []
    html_parts.append(f"<h1>{dataset_name} Dataset Report</h1>")

    summary_dict = st.session_state.get("dataset_summary", {})
    for name, summary in summary_dict.items():
        html_parts.append(f"<h2>{name} Dataset</h2>")
        html_parts.append("<h3>Preview</h3>")
        html_parts.append(summary["preview"].to_html(index=False))

        html_parts.append("<h3>Description</h3>")
        html_parts.append(summary["description"].to_html(index=False))

        html_parts.append("<h3>Info</h3>")
        html_parts.append(summary["info"].to_html(index=False))

    html_parts.append("<hr><h2>Visualizations</h2>")

    if "class_dist_path" in st.session_state:
        abs_path = os.path.abspath(st.session_state["class_dist_path"])
        html_parts.append(f"<h3>Class Distribution</h3><img src='file://{abs_path}' width='800'><br><br>")

    if "doc_len_path" in st.session_state:
        abs_path = os.path.abspath(st.session_state["doc_len_path"])
        html_parts.append(f"<img src='file://{abs_path}' width='800'><br><br>")

    if "doc_len_table" in st.session_state:
        html_parts.append(st.session_state["doc_len_table"])

    if "wordcloud_path" in st.session_state:
        abs_path = os.path.abspath(st.session_state["wordcloud_path"])
        html_parts.append(f"<h3>Word Cloud</h3><img src='file://{abs_path}' width='900'><br><br>")

    # 📌 통계 기반 요약
    try:
        summarization = gen_summarization()
        html_parts.append("<h3>📌 통계 요약 코멘트:</h3>")
        html_parts.append('<div class="comment-box"><ul>')
        for line in summarization.splitlines():
            line = line.strip()
            if line:
                if line[0].isdigit() and line[1] == '.':
                    line = line[2:].strip()
                html_parts.append(f"<li>{line}</li>")
        html_parts.append("</ul></div>")
    except Exception as e:
        html_parts.append(f"<p><strong>통계 요약 실패:</strong> {e}</p>")

    # 📌 드리프트 추정
    try:
        explanation = gen_explanation()
        html_parts.append("<h3>📌 드리프트 관련 해석:</h3>")
        html_parts.append('<div class="comment-box"><ul>')
        for line in explanation.splitlines():
            line = line.strip()
            if line:
                if line[0].isdigit() and line[1] == '.':
                    line = line[2:].strip()
                html_parts.append(f"<li>{line}</li>")
        html_parts.append("</ul></div>")
    except Exception as e:
        html_parts.append(f"<p><strong>드리프트 설명 실패:</strong> {e}</p>")

    # ✅ 스타일 포함한 템플릿
    html_template = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 30px;
                }}
                h1 {{ font-size: 28px; }}
                h2 {{ font-size: 22px; margin-top: 40px; }}
                h3 {{ font-size: 18px; margin-top: 25px; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                    font-size: 13px;
                }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 6px;
                    text-align: left;
                }}
                th {{ background-color: #f5f5f5; }}
                img {{
                    margin-top: 10px;
                    margin-bottom: 20px;
                }}
                .comment-box {{
                    background-color: #f4f4f4;
                    padding: 15px;
                    margin: 10px 0 30px 0;
                    border-radius: 8px;
                }}
                ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 6px;
                }}
            </style>
        </head>
        <body>
            {''.join(html_parts)}
        </body>
    </html>
    """
    return html_template

def render():
    st.write("📦 현재 session_state:")
    
    if 'dataset_summary' not in st.session_state:
        st.error("No dataset info found.")
        return

    dataset_name = st.session_state.get('dataset_name', 'Dataset')

    dataset_name = st.session_state.get('dataset_name', 'Dataset')

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    html_out = generate_html_from_session(dataset_name)
    html_path = os.path.join(reports_dir, f"{dataset_name}_report.html")
    pdf_path = os.path.join(reports_dir, f"{dataset_name}_report.pdf")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    try:
        config = pdfkit.configuration()
        options = {
            'enable-local-file-access': None
        }
        pdfkit.from_file(html_path, pdf_path, configuration=config, options=options)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Dataset Report (PDF)",
                data=f,
                file_name=f"{dataset_name}_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")