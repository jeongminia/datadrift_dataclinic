import streamlit as st
import pandas as pd 
# data & model load, Embedding
from utils import load_data, split_columns, EmbeddingPipeline
# Detect DataDrift
from evidently.metrics import EmbeddingsDriftMetric
from evidently.report import Report
from evidently.metrics.data_drift.embedding_drift_methods import mmd
from evidently import ColumnMapping

## --------------- main --------------- ##
def render():
    st.title("Detect DataDrift Page")

    # 데이터 로드
    train_df, valid_df, test_df = load_data()
    train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기

    # 임베딩
    pipeline = EmbeddingPipeline()
    pipeline.load_model()

    max_len = pipeline.calculate_max_len(train_df, train_text_cols[0])

    train_embeddings = pipeline.generate_embeddings(train_df, train_text_cols[0], max_len=max_len)
    valid_embeddings = pipeline.generate_embeddings(valid_df, train_text_cols[0], max_len=max_len)
    test_embeddings = pipeline.generate_embeddings(test_df, train_text_cols[0], max_len=max_len)

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
    st.pyplot(report.visualize())


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
    st.pyplot(report.visualize())
