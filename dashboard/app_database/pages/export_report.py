import pandas as pd
import pdfkit
import streamlit as st
import os
from utils import generate_explanation

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

    try:
        context = f"""
        총 문서 수: {st.session_state.get('total_docs', 0)}
        평균 문서 길이: {st.session_state.get('avg_length', 0)} 단어
        주요 키워드: {', '.join(st.session_state.get('top_keywords', []))}
        """
        comment = generate_explanation(context)
        html_parts.append(f"<p><strong>📌 요약 코멘트:</strong> {comment}</p>")
    except Exception as e:
        html_parts.append(f"<p><strong>📌 요약 코멘트 생성 실패:</strong> {e}</p>")

    html_parts.append("<hr><h2>Visualizations</h2>")


    # if "descriptors_msg" in st.session_state:
    #    html_parts.append(f"<p>{st.session_state['descriptors_msg']}</p>")

    if "class_dist_path" in st.session_state:
        abs_path = os.path.abspath(st.session_state["class_dist_path"])
        html_parts.append(f"<h3>Class Distribution</h3><img src='file://{abs_path}' width='800'><br><br>")

    if "doc_len_msg" in st.session_state:
        html_parts.append(f"<h3>Text Length Summary</h3><p>{st.session_state['doc_len_msg']}</p>")

    if "doc_len_path" in st.session_state:
        abs_path = os.path.abspath(st.session_state["doc_len_path"])
        html_parts.append(f"<img src='file://{abs_path}' width='800'><br><br>")

    if "doc_len_table" in st.session_state:
        html_parts.append(st.session_state["doc_len_table"])

    if "wordcloud_path" in st.session_state:
        abs_path = os.path.abspath(st.session_state["wordcloud_path"])
        html_parts.append(f"<h3>Word Cloud</h3><img src='file://{abs_path}' width='900'><br><br>")

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
            </style>
        </head>
        <body>
            {''.join(html_parts)}
        </body>
    </html>
    """
    return html_template

def render():
    if 'dataset_summary' not in st.session_state:
        st.error("No dataset info found.")
        return

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
