import streamlit as st
import pandas as pd 
import numpy as np
# data & model load, Embedding
from utils import load_data, split_columns
# Detect DataDrift
from evidently.metrics import EmbeddingsDriftMetric
from evidently.report import Report
from evidently.metrics.data_drift.embedding_drift_methods import mmd
from evidently.metrics.data_drift.embedding_drift_methods import ratio
from evidently import ColumnMapping
import streamlit.components.v1 as components  # HTML 렌더링을 위한 Streamlit 컴포넌트
import os
import matplotlib.pyplot as plt
# HTML 저장 경로 설정
HTML_SAVE_PATH = "./reports"


## --------------- main --------------- ##
def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Detect {dataset_name} DataDrift Page")

    if 'train_embeddings' not in st.session_state or 'valid_embeddings' not in st.session_state or 'test_embeddings' not in st.session_state:
        st.error("Embeddings are not available. Please generate embeddings in the 'Embedding Visualization' tab first.")
        st.write("Debug Info: ", st.session_state)
        return
    
    train_embeddings = st.session_state['train_embeddings']
    valid_embeddings = st.session_state['valid_embeddings']
    test_embeddings = st.session_state['test_embeddings']

    # 데이터 형식 확인 및 변환
    if not isinstance(train_embeddings, np.ndarray):
        train_embeddings = np.array(train_embeddings)
    if not isinstance(valid_embeddings, np.ndarray):
        valid_embeddings = np.array(valid_embeddings)
    if not isinstance(test_embeddings, np.ndarray):
        test_embeddings = np.array(test_embeddings)
    
    # evidentlyai - 데이터 드리프트 검출
    st.subheader("Train(reference)-Test(current) Data Drift Detection")
    reference_df = pd.DataFrame(train_embeddings, 
                            columns=[f"dim_{i}" for i in range(train_embeddings.shape[1])])
    current_df = pd.DataFrame(test_embeddings, 
                                columns=[f"dim_{i}" for i in range(test_embeddings.shape[1])])

    column_mapping = ColumnMapping(
        embeddings={'all_dimensions': reference_df.columns.tolist()}
    )

    test_option = st.selectbox("Select Test Type", 
                               ["MMD", "Wasserstein Distance", 
                                "KL Divergence", "JensenShannon Divergence", 
                                "Energy Distance"])

    if test_option == "MMD":
        report = Report(metrics=[EmbeddingsDriftMetric('all_dimensions', 
                            drift_method = mmd(
                              threshold = 0.015,
                              bootstrap = None,
                              quantile_probability = 0.95,
                              pca_components = None,
                          ))])
    elif test_option == "Wasserstein Distance":
        report = Report(metrics=[EmbeddingsDriftMetric('all_dimensions', 
                            drift_method = ratio(
                                component_stattest='wasserstein',
                                component_stattest_threshold=0.1,
                                threshold = 0.015,
                                pca_components = None,
                            ))])
        
    elif test_option == "KL Divergence":
        report = Report(metrics=[EmbeddingsDriftMetric('all_dimensions', 
                            drift_method = ratio(
                                component_stattest='kl_div',
                                component_stattest_threshold=0.1,
                                threshold = 0.015,
                                pca_components = None,
                            ))])
    elif test_option == "JensenShannon Divergence":
        report = Report(metrics=[EmbeddingsDriftMetric('all_dimensions', 
                            drift_method = ratio(
                                component_stattest='jensenshannon',
                                component_stattest_threshold=0.1,
                                threshold = 0.015,
                                pca_components = None,
                            ))])
    elif test_option == "Energy Distance":
        report = Report(metrics=[EmbeddingsDriftMetric('all_dimensions', 
                            drift_method = ratio(
                                component_stattest='ed',
                                component_stattest_threshold=0.1,
                                threshold = 0.015,
                                pca_components = None,
                            ))])

    
    report.run(reference_data = reference_df, current_data = current_df, 
           column_mapping = column_mapping)
    
    from weasyprint import HTML

    # 1. HTML 저장
    train_test_report_path = os.path.join(HTML_SAVE_PATH, f"{dataset_name} train_test_drift_report.html")
    report.save_html(train_test_report_path)

    # 2. HTML → PDF 변환
    pdf_path = train_test_report_path.replace(".html", ".pdf")
    HTML(train_test_report_path).write_pdf(pdf_path)

    # 3. 다운로드 버튼 제공
    st.success("✅ PDF 저장 완료! 아래 버튼을 눌러 다운로드하세요.")

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf"
    )