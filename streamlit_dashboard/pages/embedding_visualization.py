import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
# data & model load, Embedding
from utils import load_data, split_columns, EmbeddingPipeline
import torch
# 치원축소
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action='ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"

## --------------- Visualization --------------- ##
def visualize_similarity_distance(valid_embeddings, test_embeddings, train_embeddings):
    # 코사인 유사도 및 유클리디안 거리 계산
    cosine_valid_train = cosine_similarity(valid_embeddings, train_embeddings)
    cosine_test_train = cosine_similarity(test_embeddings, train_embeddings)
    euclidean_valid_train = euclidean_distances(valid_embeddings, train_embeddings)
    euclidean_test_train = euclidean_distances(test_embeddings, train_embeddings)
    
    # 시각화
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Cosine Similarity (Valid-Train)
    sns.heatmap(cosine_valid_train, cmap="YlGnBu", 
                xticklabels=False, yticklabels=False, ax=axes[0])
    axes[0].set_title("Cosine: Valid-Train")

    # Cosine Similarity (Test-Train)
    sns.heatmap(cosine_test_train, cmap="YlGnBu", 
                xticklabels=False, yticklabels=False, ax=axes[1])
    axes[1].set_title("Cosine: Test-Train")

    # Euclidean Distance (Valid-Train)
    sns.heatmap(euclidean_valid_train, cmap="YlGnBu", 
                xticklabels=False, yticklabels=False, ax=axes[2])
    axes[2].set_title("Euclidean: Valid-Train")

    # Euclidean Distance (Test-Train)
    sns.heatmap(euclidean_test_train, cmap="YlGnBu", 
                xticklabels=False, yticklabels=False, ax=axes[3])
    axes[3].set_title("Euclidean: Test-Train")

    plt.tight_layout()
    return fig, axes


def plot_reduced(valid_pca, test_pca, train_pca, 
                 label_valid="Valid", label_test="Test", label_train="Train"):
    # 분포 시각화
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # 1x6 플롯 생성

    # 색상 설정
    colors = {label_train: "orange", label_valid: "blue", label_test: "green"}

    # 2D Scatter Plot (valid-train)
    axes[0].scatter(
        train_pca[:, 0], train_pca[:, 1], alpha=0.5, label=label_train, c=colors[label_train]
    )
    axes[0].scatter(
        valid_pca[:, 0], valid_pca[:, 1], alpha=0.5, label=label_valid, c=colors[label_valid]
    )
    axes[0].set_title("2D Scatter Plot (valid-train)", fontsize=12)
    axes[0].set_xlabel("PC1", fontsize=10)
    axes[0].set_ylabel("PC2", fontsize=10)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2D Scatter Plot (test-train)
    axes[1].scatter(
        train_pca[:, 0], train_pca[:, 1], alpha=0.5, label=label_train, c=colors[label_train]
    )
    axes[1].scatter(
        test_pca[:, 0], test_pca[:, 1], alpha=0.5, label=label_test, c=colors[label_test]
    )
    axes[1].set_title("2D Scatter Plot (test-train)", fontsize=12)
    axes[1].set_xlabel("PC1", fontsize=10)
    axes[1].set_ylabel("PC2", fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 2D Density Plot (valid-train)
    sns.kdeplot(
        x=train_pca[:, 0], y=train_pca[:, 1], ax=axes[2], fill=True, alpha=0.5, color=colors[label_train], label=label_train
    )
    sns.kdeplot(
        x=valid_pca[:, 0], y=valid_pca[:, 1], ax=axes[2], fill=True, alpha=0.5, color=colors[label_valid], label=label_valid
    )
    axes[2].set_title("2D Density Plot (valid-train)", fontsize=12)
    axes[2].set_xlabel("PC1", fontsize=10)
    axes[2].set_ylabel("PC2", fontsize=10)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # 2D Density Plot (test-train)
    sns.kdeplot(
        x=train_pca[:, 0], y=train_pca[:, 1], ax=axes[3], fill=True, alpha=0.5, color=colors[label_train], label=label_train
    )
    sns.kdeplot(
        x=test_pca[:, 0], y=test_pca[:, 1], ax=axes[3], fill=True, alpha=0.5, color=colors[label_test], label=label_test
    )
    axes[3].set_title("2D Density Plot (test-train)", fontsize=12)
    axes[3].set_xlabel("PC1", fontsize=10)
    axes[3].set_ylabel("PC2", fontsize=10)
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    # 3D Scatter Plot (valid-train)
    ax_3d_valid = fig.add_subplot(1, 6, 5, projection="3d")
    ax_3d_valid.scatter(
        train_pca[:, 0], train_pca[:, 1], train_pca[:, 2], alpha=0.5, label=label_train, c=colors[label_train]
    )
    ax_3d_valid.scatter(
        valid_pca[:, 0], valid_pca[:, 1], valid_pca[:, 2], alpha=0.5, label=label_valid, c=colors[label_valid]
    )
    ax_3d_valid.set_title("3D Scatter Plot (valid-train)", fontsize=12)
    ax_3d_valid.set_xlabel("PC1", fontsize=10)
    ax_3d_valid.set_ylabel("PC2", fontsize=10)
    ax_3d_valid.set_zlabel("PC3", fontsize=10)
    ax_3d_valid.legend()

    # 3D Scatter Plot (test-train)
    ax_3d_test = fig.add_subplot(1, 6, 6, projection="3d")
    ax_3d_test.scatter(
        train_pca[:, 0], train_pca[:, 1], train_pca[:, 2], alpha=0.5, label=label_train, c=colors[label_train]
    )
    ax_3d_test.scatter(
        test_pca[:, 0], test_pca[:, 1], test_pca[:, 2], alpha=0.5, label=label_test, c=colors[label_test]
    )
    ax_3d_test.set_title("3D Scatter Plot (test-train)", fontsize=12)
    ax_3d_test.set_xlabel("PC1", fontsize=10)
    ax_3d_test.set_ylabel("PC2", fontsize=10)
    ax_3d_test.set_zlabel("PC3", fontsize=10)
    ax_3d_test.legend()

    plt.tight_layout()
    return fig, axes


def render():
    st.title("Embedding Visualization Page")

    train_df, valid_df, test_df, column_info = load_data()
    
    train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기

    # 임베딩
    pipeline = EmbeddingPipeline()
    pipeline.load_model()

    max_len = pipeline.calculate_max_len(train_df, train_text_cols[0])
    st.write(f"Max Length: {max_len}")

    train_embeddings = pipeline.generate_embeddings(train_df, train_text_cols[0], max_len=max_len)
    valid_embeddings = pipeline.generate_embeddings(valid_df, train_text_cols[0], max_len=max_len)
    test_embeddings = pipeline.generate_embeddings(test_df, train_text_cols[0], max_len=max_len)
    
    # distance 시각화
    st.subheader("Original Dimension")
    fig, axes = visualize_similarity_distance(valid_embeddings, test_embeddings, train_embeddings)
    st.pyplot(fig)

    # PCA 차원에 따라 시각화
    st.subheader("Dimension Reduction with PCA")
    dim_option = st.selectbox("Select Size of Dimension", [10, 50, 100, 200, 300, 400, 500])
    pca = PCA(n_components = dim_option)
    pca.fit(train_embeddings)
    train_pca = pca.transform(train_embeddings)
    valid_pca = pca.transform(valid_embeddings)
    test_pca = pca.transform(test_embeddings)

    fig, axes = visualize_similarity_distance(valid_embeddings = valid_pca, 
                                              test_embeddings = test_pca, 
                                              train_embeddings = train_pca)
    st.pyplot(fig)

    fig, axes = plot_reduced(valid_pca, test_pca, train_pca)
    st.pyplot(fig)