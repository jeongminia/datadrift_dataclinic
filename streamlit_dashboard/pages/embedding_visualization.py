import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# data & model load, Embedding & visualization
from utils import load_data, split_columns, EmbeddingPipeline, visualize_similarity_distance, plot_reduced
import torch
# 치원축소
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action='ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"


def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Embedding Visualization Page of {dataset_name}")

    train_df, valid_df, test_df = load_data()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return
    
    if train_df is None:
        st.error("Train dataset is None!")
        return
    if valid_df is None:
        st.error("Validation dataset is None!")
        return
    if test_df is None:
        st.error("Test dataset is None!")
        return

    train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기

    # 임베딩
    pipeline = EmbeddingPipeline()
    pipeline.load_model()

    max_len = pipeline.calculate_max_len(train_df, train_text_cols)
    st.write(f"Max Length: {max_len}")

    with st.spinner('Generating embeddings for train dataset...'):
        try:
            train_embeddings = pipeline.generate_embeddings(train_df, train_text_cols, max_len=max_len)
            st.write(f"Train embeddings shape: {train_embeddings.shape}")
            st.session_state["train_embeddings"] = train_embeddings
        except Exception as e:
            st.error(f"Error in generating train embeddings: {e}")
            return
    with st.spinner('Generating embeddings for validation dataset...'):
        try:
            valid_embeddings = pipeline.generate_embeddings(valid_df, train_text_cols, max_len=max_len)
            st.write(f"Validation embeddings shape: {valid_embeddings.shape}")
            st.session_state["valid_embeddings"] = valid_embeddings
        except Exception as e:
            st.error(f"Error in generating validation embeddings: {e}")
            return
    with st.spinner('Generating embeddings for test dataset...'):
        try:
            test_embeddings = pipeline.generate_embeddings(test_df, train_text_cols, max_len=max_len)
            st.write(f"Test embeddings shape: {test_embeddings.shape}")
            st.session_state["test_embeddings"] = test_embeddings
        except Exception as e:
            st.error(f"Error in generating test embeddings: {e}")
            return
    
    train_embeddings = st.session_state['train_embeddings']
    valid_embeddings = st.session_state['valid_embeddings']
    test_embeddings = st.session_state['test_embeddings']

    # distance 시각화
    st.subheader("Original Dimension")
    try:
        visualize_similarity_distance(valid_embeddings, test_embeddings, train_embeddings)
    except Exception as e:
        st.error(f"Error in visualizing similarity distance: {e}")
    
    # PCA 차원에 따라 시각화
    st.subheader("Dimension Reduction with PCA")
    dim_option = st.selectbox("Select Size of Dimension", [10, 50, 100, 200, 300, 400, 500])

    pca = PCA(n_components = dim_option)
    pca.fit(train_embeddings)
    train_pca = pca.transform(train_embeddings)
    valid_pca = pca.transform(valid_embeddings)
    test_pca = pca.transform(test_embeddings)

    st.write(f"Train PCA shape: {train_pca.shape}")
    st.write(f"Validation PCA shape: {valid_pca.shape}")
    st.write(f"Test PCA shape: {test_pca.shape}")

    try:
        visualize_similarity_distance(valid_pca, test_pca, train_pca)
        fig = plot_reduced(valid_pca, test_pca, train_pca)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in PCA visualization: {e}")