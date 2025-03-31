import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

## --------------- Load Data --------------- ##

def load_data():
    if 'train_df' not in st.session_state or 'valid_df' not in st.session_state or 'test_df' not in st.session_state:
        st.error("Datasets are not loaded. Please upload the datasets in the Upload Data tab.")
        return None, None, None

    # 세션 상태에서 데이터 가져오기
    train_df = st.session_state['train_df']
    valid_df = st.session_state['valid_df']
    test_df = st.session_state['test_df']

    return train_df, valid_df, test_df

def split_columns(df):
    if df is None:
        raise ValueError("Dataframe is None!")

    # text_col : 가장 긴 문자열을 가진 컬럼 선택
    text_col = max(
        (col for col in df.columns),
        key=lambda col: df[col].dropna().astype(str).str.len().max(),
        default=None
    )
    # 모든 컬럼 object로 지정 후 텍스트 컬럼 제외한 나머지를 category로 변환
    for col in df.columns:
        if col != text_col:
            df[col] = df[col].astype("category")

    text_col = str(text_col) if text_col else None
    class_cols = [col for col in df.columns if col != text_col] if text_col else list(df.columns)

    return text_col, class_cols


## --------------- Visualization --------------- ##
def visualize_similarity_distance(valid_embeddings, test_embeddings, train_embeddings):
    try:
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

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in visualize_similarity_distance: {e}")

import matplotlib.lines as mlines

def plot_reduced(valid_pca, test_pca, train_pca, 
                 label_valid="Valid", label_test="Test", label_train="Train"):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # 1x6 플롯 생성
    colors = {label_train: "orange", label_valid: "blue", label_test: "green"}

    # 2D Scatter Plot (valid-train)
    axes[0].scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.5, label=label_train, c=colors[label_train])
    axes[0].scatter(valid_pca[:, 0], valid_pca[:, 1], alpha=0.5, label=label_valid, c=colors[label_valid])
    axes[0].set_title("2D Scatter Plot (valid-train)", fontsize=12)
    axes[0].set_xlabel("PC1", fontsize=10)
    axes[0].set_ylabel("PC2", fontsize=10)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2D Scatter Plot (test-train)
    axes[1].scatter(train_pca[:, 0], train_pca[:, 1], alpha=0.5, label=label_train, c=colors[label_train])
    axes[1].scatter(test_pca[:, 0], test_pca[:, 1], alpha=0.5, label=label_test, c=colors[label_test])
    axes[1].set_title("2D Scatter Plot (test-train)", fontsize=12)
    axes[1].set_xlabel("PC1", fontsize=10)
    axes[1].set_ylabel("PC2", fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 2D Density Plot (valid-train)
    sns.kdeplot(x=train_pca[:, 0], y=train_pca[:, 1], ax=axes[2], fill=True, alpha=0.5, color=colors[label_train])
    sns.kdeplot(x=valid_pca[:, 0], y=valid_pca[:, 1], ax=axes[2], fill=True, alpha=0.5, color=colors[label_valid])
    axes[2].set_title("2D Density Plot (valid-train)", fontsize=12)
    axes[2].set_xlabel("PC1", fontsize=10)
    axes[2].set_ylabel("PC2", fontsize=10)
    handles2 = [
        mlines.Line2D([], [], color=colors[label_train], label=label_train),
        mlines.Line2D([], [], color=colors[label_valid], label=label_valid)
    ]
    axes[2].legend(handles=handles2)
    axes[2].grid(alpha=0.3)

    # 2D Density Plot (test-train)
    sns.kdeplot(x=train_pca[:, 0], y=train_pca[:, 1], ax=axes[3], fill=True, alpha=0.5, color=colors[label_train])
    sns.kdeplot(x=test_pca[:, 0], y=test_pca[:, 1], ax=axes[3], fill=True, alpha=0.5, color=colors[label_test])
    axes[3].set_title("2D Density Plot (test-train)", fontsize=12)
    axes[3].set_xlabel("PC1", fontsize=10)
    axes[3].set_ylabel("PC2", fontsize=10)
    handles3 = [
        mlines.Line2D([], [], color=colors[label_train], label=label_train),
        mlines.Line2D([], [], color=colors[label_test], label=label_test)
    ]
    axes[3].legend(handles=handles3)
    axes[3].grid(alpha=0.3)

    # 3D Scatter Plot (valid-train)
    ax_3d_valid = fig.add_subplot(1, 6, 5, projection="3d")
    ax_3d_valid.scatter(train_pca[:, 0], train_pca[:, 1], train_pca[:, 2], alpha=0.5, label=label_train, c=colors[label_train])
    ax_3d_valid.scatter(valid_pca[:, 0], valid_pca[:, 1], valid_pca[:, 2], alpha=0.5, label=label_valid, c=colors[label_valid])
    ax_3d_valid.set_title("3D Scatter Plot (valid-train)", fontsize=12)
    ax_3d_valid.set_xlabel("PC1", fontsize=10)
    ax_3d_valid.set_ylabel("PC2", fontsize=10)
    ax_3d_valid.set_zlabel("PC3", fontsize=10)
    ax_3d_valid.legend()

    # 3D Scatter Plot (test-train)
    ax_3d_test = fig.add_subplot(1, 6, 6, projection="3d")
    ax_3d_test.scatter(train_pca[:, 0], train_pca[:, 1], train_pca[:, 2], alpha=0.5, label=label_train, c=colors[label_train])
    ax_3d_test.scatter(test_pca[:, 0], test_pca[:, 1], test_pca[:, 2], alpha=0.5, label=label_test, c=colors[label_test])
    ax_3d_test.set_title("3D Scatter Plot (test-train)", fontsize=12)
    ax_3d_test.set_xlabel("PC1", fontsize=10)
    ax_3d_test.set_ylabel("PC2", fontsize=10)
    ax_3d_test.set_zlabel("PC3", fontsize=10)
    ax_3d_test.legend()

    plt.tight_layout()
    return fig

## --------------- Embedding --------------- ##
class EmbeddingPipeline:
    def __init__(self, model_name="klue/roberta-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        # 세션 상태를 사용하여 모델 로드 과정을 최적화
        if 'tokenizer' not in st.session_state or 'model' not in st.session_state:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            st.session_state['tokenizer'] = self.tokenizer
            st.session_state['model'] = self.model
        else:
            self.tokenizer = st.session_state['tokenizer']
            self.model = st.session_state['model']

    class CustomDataset(Dataset):
        def __init__(self, dataframe, text_col):
            self.texts = dataframe[text_col]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts.iloc[idx]