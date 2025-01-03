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
    st.title("Embedding Visualization Page")

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
        train_embeddings = pipeline.generate_embeddings(train_df, train_text_cols, max_len=max_len)
    with st.spinner('Generating embeddings for validation dataset...'):
        valid_embeddings = pipeline.generate_embeddings(valid_df, train_text_cols, max_len=max_len)
    with st.spinner('Generating embeddings for test dataset...'):
        test_embeddings = pipeline.generate_embeddings(test_df, train_text_cols, max_len=max_len)
    
    # distance 시각화
    st.subheader("Original Dimension")
    visualize_similarity_distance(valid_embeddings, test_embeddings, train_embeddings)
    

    # PCA 차원에 따라 시각화
    st.subheader("Dimension Reduction with PCA")
    dim_option = st.selectbox("Select Size of Dimension", [10, 50, 100, 200, 300, 400, 500])
    pca = PCA(n_components = dim_option)
    pca.fit(train_embeddings)
    train_pca = pca.transform(train_embeddings)
    valid_pca = pca.transform(valid_embeddings)
    test_pca = pca.transform(test_embeddings)

    visualize_similarity_distance(valid_pca, test_pca, train_pca)
    fig = plot_reduced(valid_pca, test_pca, train_pca)
    st.pyplot(fig)