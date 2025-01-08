import streamlit as st
import pandas as pd 
# data & model load, Embedding
from utils import load_data, split_columns, EmbeddingPipeline
# Detect DataDrift
from evidently.metrics import EmbeddingsDriftMetric
from evidently.report import Report
from evidently.metrics.data_drift.embedding_drift_methods import mmd
from evidently import ColumnMapping
import streamlit.components.v1 as components  # HTML 렌더링을 위한 Streamlit 컴포넌트
import os

# HTML 저장 경로 설정
HTML_SAVE_PATH = "./reports"


## --------------- main --------------- ##
def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Detect {dataset_name} DataDrift Page")

    if 'train_embeddings' not in st.session_state or 'valid_embeddings' not in st.session_state or 'test_embeddings' not in st.session_state:
        st.error("Embeddings are not available. Please generate embeddings in the 'Embedding Visualization' tab first.")
        return
    
    train_embeddings = st.session_state['train_embeddings']
    valid_embeddings = st.session_state['valid_embeddings']
    test_embeddings = st.session_state['test_embeddings']
    

    # 데이터 드리프트 검출
    st.subheader("Train-Validation Data Drift Detection")
    reference_df = pd.DataFrame(train_embeddings, 
                            columns=[f"dim_{i}" for i in range(train_embeddings.shape[1])])
    current_df = pd.DataFrame(valid_embeddings, 
                                columns=[f"dim_{i}" for i in range(valid_embeddings.shape[1])])

    column_mapping = ColumnMapping(
        embeddings={'all_dimensions': reference_df.columns.tolist()}
    )
    report = Report(metrics=[
    EmbeddingsDriftMetric('all_dimensions', 
                         drift_method = mmd(
                              threshold = 0.015,
                              bootstrap = None,
                              quantile_probability = 0.95,
                              pca_components = None,
                          )
    )
    ])
    report.run(reference_data = reference_df, current_data = current_df, 
           column_mapping = column_mapping)
    
    # report 출력
    train_valid_report_path = os.path.join(HTML_SAVE_PATH, f"{dataset_name} train_valid_drift_report.html")
    report.save_html(train_valid_report_path)
    # HTML 렌더링
    with open(train_valid_report_path, "r") as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)



    st.subheader("Train-Test Data Drift Detection")
    reference_df = pd.DataFrame(train_embeddings, 
                            columns=[f"dim_{i}" for i in range(train_embeddings.shape[1])])
    current_df = pd.DataFrame(test_embeddings, 
                                columns=[f"dim_{i}" for i in range(test_embeddings.shape[1])])

    column_mapping = ColumnMapping(
        embeddings={'all_dimensions': reference_df.columns.tolist()}
    )
    report = Report(metrics=[
    EmbeddingsDriftMetric('all_dimensions', 
                         drift_method = mmd(
                              threshold = 0.015,
                              bootstrap = None,
                              quantile_probability = 0.95,
                              pca_components = None,
                          )
    )
    ])
    report.run(reference_data = reference_df, current_data = current_df, 
           column_mapping = column_mapping)
    
     # HTML 파일 저장
    train_test_report_path = os.path.join(HTML_SAVE_PATH, f"{dataset_name} train_test_drift_report.html")
    report.save_html(train_test_report_path)

    # HTML 렌더링
    with open(train_test_report_path, "r") as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)
