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
    
    # ✅ 세션 상태에서 직접 데이터프레임 불러오기
    if "train_df" not in st.session_state or "test_df" not in st.session_state:
        st.error("❌ Datasets are not loaded. Please upload them in the 'Upload Data' tab.")
        return

    train_df = st.session_state["train_df"]
    test_df = st.session_state["test_df"]


    # 텍스트 컬럼 분리
    train_text_cols, train_class_cols = split_columns(train_df)

    # TextData 객체로 변환
    train_text = TextData(train_df[train_text_cols])
    train_text.calculate_builtin_properties()
    st.session_state['train_properties'] = train_text.properties

    test_text = TextData(test_df[train_text_cols])
    test_text.calculate_builtin_properties()
    st.session_state['test_properties'] = test_text.properties

    # 데이터셋 미리보기
    dataset_option = st.selectbox("Select Dataset", ["Train", "Test"])
    dataset = train_text.properties if dataset_option == "Train" else test_text.properties

    st.subheader(f"{dataset_option} Dataset Preview")
    st.dataframe(dataset.head(10), use_container_width=True)

    # 임베딩 존재 여부 확인
    if 'train_embeddings' not in st.session_state or 'test_embeddings' not in st.session_state:
        st.error("❌ Embeddings are not available. Please generate embeddings in the 'Embedding Visualization' tab first.")
        return
    
    train_embeddings = st.session_state['train_embeddings']
    test_embeddings = st.session_state['test_embeddings']

    # 임베딩 연결
    train_text.set_embeddings(train_embeddings)
    test_text.set_embeddings(test_embeddings)

    # Property Drift 실행
    check = PropertyDrift()
    result = check.run(train_dataset=train_text, test_dataset=test_text)

    # 결과 출력
    st.write("Result object:", result)

    # HTML 경로
    html_path = os.path.join(HTML_SAVE_PATH, f"{dataset_name}_property_drift_report.html")
    if os.path.exists(html_path):
        st.success(f"📄 Report is ready: {html_path}")
        if st.button("🚀 Open Report in Browser"):
            st.write("✅ Button Clicked!")
            import webbrowser
            webbrowser.open(f"file://{html_path}")
    else:
        st.error("🚨 HTML report file was not found.")
