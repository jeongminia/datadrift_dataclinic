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

    # 데이터 로드
    # train_df, valid_df, test_df = load_data()
    # train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기

    # 임베딩
    # pipeline = EmbeddingPipeline()
    # pipeline.load_model()

    # max_len = pipeline.calculate_max_len(train_df, train_text_cols)

    # with st.spinner('Generating embeddings for train dataset...'):
    #     try:
    #         train_embeddings = pipeline.generate_embeddings(train_df, train_text_cols, max_len=max_len)
    #         st.write(f"Train embeddings shape: {train_embeddings.shape}")
    #     except Exception as e:
    #         st.error(f"Error in generating train embeddings: {e}")
    #         return
    # with st.spinner('Generating embeddings for validation dataset...'):
    #     try:
    #         valid_embeddings = pipeline.generate_embeddings(valid_df, train_text_cols, max_len=max_len)
    #         st.write(f"Validation embeddings shape: {valid_embeddings.shape}")
    #     except Exception as e:
    #         st.error(f"Error in generating validation embeddings: {e}")
    #         return
    # with st.spinner('Generating embeddings for test dataset...'):
    #     try:
    #         test_embeddings = pipeline.generate_embeddings(test_df, train_text_cols, max_len=max_len)
    #         st.write(f"Test embeddings shape: {test_embeddings.shape}")
    #     except Exception as e:
    #         st.error(f"Error in generating test embeddings: {e}")
    #        return

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
    train_valid_report_path = os.path.join(HTML_SAVE_PATH, "train_valid_drift_report.html")
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
    train_test_report_path = os.path.join(HTML_SAVE_PATH, "train_test_drift_report.html")
    report.save_html(train_test_report_path)

    # HTML 렌더링
    with open(train_test_report_path, "r") as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)
