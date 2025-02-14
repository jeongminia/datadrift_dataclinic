import streamlit as st
import pandas as pd 
import numpy as np
# data & model load, Embedding
from utils import load_data, split_columns, EmbeddingPipeline
# Detect DataDrift
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PropertyDrift
import streamlit.components.v1 as components  # HTML 렌더링을 위한 Streamlit 컴포넌트
import os
import matplotlib.pyplot as plt
# HTML 저장 경로 설정
HTML_SAVE_PATH = "./reports"


## --------------- main --------------- ##
def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Detect {dataset_name} Property Drift Page")
    
    train_df, valid_df, test_df = load_data()
    if train_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return
    train_text_cols, train_class_cols = split_columns(train_df)

    train_text = TextData(train_df[train_text_cols])
    train_text.calculate_builtin_properties()
    st.session_state['train_properties'] = train_text.properties

    test_text = TextData(test_df[train_text_cols])
    test_text.calculate_builtin_properties()
    st.session_state['test_properties'] = test_text.properties

    dataset_option = st.selectbox("Select Dataset", ["Train", "Test"])

    dataset = None
    if dataset_option == "Train":
        dataset = train_text.properties
    elif dataset_option == "Test":
        dataset = test_text.properties

    # 데이터셋 미리보기
    st.subheader(f"{dataset_option} Dataset Preview")
    st.dataframe(dataset.head(10), use_container_width=True)


    if ('train_embeddings' not in st.session_state) or ('test_embeddings' not in st.session_state):
        st.error("Embeddings are not available. Please generate embeddings in the 'Embedding Visualization' tab first.")
        return
    
    train_embeddings = st.session_state['train_embeddings']
    test_embeddings = st.session_state['test_embeddings']

    # TextData 객체 생성 및 임베딩 추가
    train_text.set_embeddings(train_embeddings)
    test_text.set_embeddings(test_embeddings)

    check = PropertyDrift()
    result = check.run(train_dataset=train_text, test_dataset=test_text)

    # 결과 디버깅
    st.write("Result object:", result)

    # 결과 출력
    import webbrowser
    html_path = os.path.join(HTML_SAVE_PATH, f"{dataset_name}_property_drift_report.html")

    if os.path.exists(html_path):
        st.success(f"📄 Report is ready: {html_path}")

    # 브라우저에서 직접 열기
        if st.button("🚀 Open Report in Browser"):
            st.write("✅ Button Clicked!")
            webbrowser.open(f"file://{html_path}")

    else:
        st.error("🚨 HTML report file was not found.")