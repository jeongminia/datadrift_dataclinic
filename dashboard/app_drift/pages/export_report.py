import streamlit as st
import base64
from bs4 import BeautifulSoup
import os
import pdfkit

# Import utils from parent directory
try:
    from ..utils import gen_drift_score_explanation
    # RAG 기능을 위해 build_RAG_docs.py에서 함수 import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from datadrift_dataclinic.dashboard.app_report.utills.build_RAG_docs import generate_llm_drift_explanation
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import gen_drift_score_explanation
    # RAG 기능을 위해 build_RAG_docs.py에서 함수 import
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from datadrift_dataclinic.dashboard.app_report.utills.build_RAG_docs import generate_llm_drift_explanation
    except ImportError:
        generate_llm_drift_explanation = None

def generate_html_from_session():
    html_parts = []

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    html_parts.append(f"<h1>{dataset_name} Drift Report</h1>")

    for key in ["train_embeddings", "valid_embeddings", "test_embeddings"]:
        if key in st.session_state:
            shape = st.session_state[key].shape
            html_parts.append(f"<p><b>{key.replace('_', ' ').title()}:</b> {shape}</p>")

    if 'embedding_distance_img' in st.session_state:
        img_bytes = st.session_state['embedding_distance_img'].getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        html_parts.append("<hr><h2>Embedding Distance (Original Dimension)</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')

    if 'pca_selected_dim' in st.session_state:
        html_parts.append(f"<p><b>PCA Reduced Dimension:</b> {st.session_state['pca_selected_dim']}</p>")

    if 'embedding_pca_distance_img' in st.session_state:
        img_bytes = st.session_state['embedding_pca_distance_img'].getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        html_parts.append("<hr><h2>Embedding Distance after PCA</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')

    if 'embedding_pca_img' in st.session_state:
        img_bytes = st.session_state['embedding_pca_img'].getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        html_parts.append("<hr><h2>Embedding Visualization after PCA</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')
    

    if 'drift_score_summary' in st.session_state:
        html_parts.append("<hr><h2>Quantitative Drift Scores</h2>")
        html_parts.append(f"<pre>{st.session_state['drift_score_summary']}</pre>")
        
        score_text = st.session_state['drift_score_summary']
        dataset_name = st.session_state.get("dataset_name")
        
        # RAG 기능 사용 시도, 실패시 기존 방식 사용
        if generate_llm_drift_explanation is not None:
            try:
                explanation = generate_llm_drift_explanation(dataset_name)
                if explanation:  # RAG 해설이 성공적으로 생성된 경우
                    html_parts.append(explanation)
                else:  # RAG 해설 생성 실패시 기존 방식 사용
                    explanation = gen_drift_score_explanation(score_text)
                    formatted_explanation = explanation.replace('\n', '</p><p>')
                    html_parts.append(f"""
                    <div class="drift-explanation">
                        <h2>📘 Drift Analysis Summary</h2>
                        <p>{formatted_explanation}</p>
                    </div>
                    """)
            except Exception as e:
                # RAG 실행 중 에러 발생시 기존 방식 사용
                st.warning(f"RAG 해설 생성 실패, 기존 방식 사용: {e}")
                explanation = gen_drift_score_explanation(score_text)
                formatted_explanation = explanation.replace('\n', '</p><p>')
                html_parts.append(f"""
                <div class="drift-explanation">
                    <h2>� Drift Analysis Summary</h2>
                    <p>{formatted_explanation}</p>
                </div>
                """)
        else:
            # RAG 함수를 import하지 못한 경우 기존 방식 사용
            explanation = gen_drift_score_explanation(score_text)
            formatted_explanation = explanation.replace('\n', '</p><p>')
            html_parts.append(f"""
            <div class="drift-explanation">
                <h2>📘 Drift Analysis Summary</h2>
                <p>{formatted_explanation}</p>
            </div>
            """)

    if 'train_test_drift_report_html' in st.session_state:
        #html_parts.append("<hr><h2>Drift Report</h2>")
        soup = BeautifulSoup(st.session_state['train_test_drift_report_html'], "html.parser")
        drift_body = soup.body or soup
        html_parts.append(str(drift_body))

    return f"""
    <html>
    <head>
        <meta charset=\"utf-8\">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; }}
            table, th, td {{ border: 1px solid #ccc; padding: 8px; }}
            pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        {''.join(html_parts)}
    </body>
    </html>
    """

def render():
    if 'dataset_summary' not in st.session_state:
        st.error("No dataset info found.")
        return

    dataset_name = st.session_state.get('dataset_name')

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    html_out = generate_html_from_session()
    html_path = os.path.join(reports_dir, f"{dataset_name}_report.html")
    pdf_path = os.path.join(reports_dir, f"{dataset_name}_report.pdf")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    try:
        config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")  # 필요 시 수정
        options = {
            'enable-local-file-access': None
        }
        pdfkit.from_file(html_path, pdf_path, configuration=config, options=options)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Dataset Report (PDF)",
                data=f,
                file_name=f"{dataset_name}_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")
