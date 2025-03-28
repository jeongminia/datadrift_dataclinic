import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# data & model load, Embedding & visualization
from utils import visualize_similarity_distance, plot_reduced
import torch
# 치원축소
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action='ignore')


def render():
    if 'embedding_data' not in st.session_state or not st.session_state['embedding_data']:
        st.error("❌ 'embedding_data' is not initialized or empty. Please load it from VectorDB.")
        return

    embedding_data = st.session_state['embedding_data']

    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Embedding Visualization Page of {dataset_name}")

    if 'embedding_data' not in st.session_state:
        st.error("Embedding data is not loaded. "
                    "Please load the embeddings in the 'Load Embeddings' tab.")
        return

    # 데이터셋 분리
    train_embeddings = np.array([res["vector"] for res in embedding_data if res.get("set_type", "").lower() == "train"],
                                dtype=np.float32)
    valid_embeddings = np.array([res["vector"] for res in embedding_data if res.get("set_type", "").lower() == "valid"],
                                dtype=np.float32)
    test_embeddings = np.array([res["vector"] for res in embedding_data if res.get("set_type", "").lower() == "test"],
                               dtype=np.float32)

    # 데이터가 비어 있는지 확인
    if train_embeddings.size == 0 or valid_embeddings.size == 0 or test_embeddings.size == 0:
        st.error("One or more embedding datasets are empty. Please check the data.")
        return

    st.write(f"Train embeddings shape: {train_embeddings.shape}")
    st.write(f"Validation embeddings shape: {valid_embeddings.shape}")
    st.write(f"Test embeddings shape: {test_embeddings.shape}")

    # distance 시각화
    st.subheader("Original Dimension")
    try:
        visualize_similarity_distance(valid_embeddings, test_embeddings, train_embeddings)
    except Exception as e:
        st.error(f"Error in visualizing similarity distance: {e}")
    
    # PCA 차원에 따라 시각화
    st.subheader("Dimension Reduction with PCA")
    dim_option = st.selectbox("Select Size of Dimension", [10, 50, 100, 200, 300, 400, 500])

    # PCA 결과 캐싱
    @st.cache_data
    def apply_pca(embeddings, n_components):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)

    train_pca = apply_pca(train_embeddings, dim_option)
    valid_pca = apply_pca(valid_embeddings, dim_option)
    test_pca = apply_pca(test_embeddings, dim_option)

    st.write(f"Train PCA shape: {train_pca.shape}")
    st.write(f"Validation PCA shape: {valid_pca.shape}")
    st.write(f"Test PCA shape: {test_pca.shape}")

    visualize_similarity_distance(valid_pca, test_pca, train_pca)
    fig = plot_reduced(valid_pca, test_pca, train_pca)
    st.pyplot(fig)