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
        cosine_valid_train = cosine_similarity(valid_embeddings, train_embeddings)
        cosine_test_train = cosine_similarity(test_embeddings, train_embeddings)
        euclidean_valid_train = euclidean_distances(valid_embeddings, train_embeddings)
        euclidean_test_train = euclidean_distances(test_embeddings, train_embeddings)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        sns.heatmap(cosine_valid_train, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axes[0])
        axes[0].set_title("Cosine: Valid-Train")

        sns.heatmap(cosine_test_train, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axes[1])
        axes[1].set_title("Cosine: Test-Train")

        sns.heatmap(euclidean_valid_train, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axes[2])
        axes[2].set_title("Euclidean: Valid-Train")

        sns.heatmap(euclidean_test_train, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axes[3])
        axes[3].set_title("Euclidean: Test-Train")

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error in visualize_similarity_distance: {e}")
        return None


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


# ------- llm expalaination --------------- #
# Ollama API를 사용하여 데이터 요약 및 드리프트 추정
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "joonoh/HyperCLOVAX-SEED-Text-Instruct-1.5B:latest"

def ollama_generate(prompt, max_tokens=300, temperature=0.7, top_p=0.9, repeat_penalty=1.1):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repeat_penalty": repeat_penalty
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        result = response.json()
        result = result["response"].strip()
        import re
        result = re.sub(r"[*_`#>]", "", result)
        return result

    except Exception as e:
        return f"Ollama 호출 오류: {e}"

def gen_drift_score_explanation(score_text: str) -> str:
    """
    드리프트 수치 결과를 해석하는 LLM 프롬프트
    """
    
    prompt = f"""
            당신은 통계적 데이터 드리프트 해석에 특화된 데이터 분석 전문가입니다.

            아래는 데이터셋 간 분포 차이를 나타내는 수치 기반 드리프트 결과입니다:

            {score_text}

            이 정보를 바탕으로 다음 목적에 맞춰 자연스럽게 서술해 주세요:

            1. 각 지표(MMD, Wasserstein, KL Divergence 등)의 의미와 수치 해석  
            2. 어떤 지표에서 실제로 drift가 발생했는지 판단 근거  
            3. drift가 발생했을 경우 모델 성능에 어떤 영향을 줄 수 있는지  
            4. 전반적으로 데이터셋 간 차이를 어떻게 해석해야 할지 요약

            단, 아래 조건을 지켜 주세요:
            - 마크다운이나 기호(`•`, `*`, `#`)는 사용하지 말고, 문장으로만 작성해 주세요.
            - 글머리 기호 없이 하나의 간결한 분석 보고서 형식으로 작성하세요.
            - 필요한 경우 핵심 수치는 문장 중간에서 다시 언급해 주세요.
            """

    return ollama_generate(prompt, max_tokens=300, temperature=0.7, top_p=0.9, repeat_penalty=1.1)


def gen_summarization() -> str:
    """Train/Validation/Test 통계를 모두 반영하여 데이터 특성 요약"""
    # 각 데이터셋에서 통계 추출
    train_df = st.session_state.get('train_df')
    valid_df = st.session_state.get('valid_df')
    test_df = st.session_state.get('test_df')

    context = f"""
    {train_df}
    {valid_df}
    {test_df}
        """

    prompt = f"""
        당신은 전문적인 데이터 분석가입니다.

        아래는 한 데이터셋에 대한 간단한 통계 정보입니다:
        {context}

        이 정보를 바탕으로 다음의 목적에 따라 간결하게 자연어 해석을 작성해 주세요:

        1. 데이터의 전반적인 특성을 요약하고,  
        2. 키워드와 길이 등으로부터 데이터 성격이나 주제의 변화 가능성을 추론하며,  
        3. train, validation, test 데이터셋 간의 차이를 분석합니다.

        ※ 단, 반드시 위 구조를 따를 필요는 없으며, 자연스럽고 다양한 문장 구성을 자유롭게 사용해도 됩니다.  
        ※ 마크다운, 별표, 강조 기호는 사용하지 말고, 설명만 반환해 주세요.
        """

    return ollama_generate(prompt, max_tokens=300, temperature=0.7, top_p=0.9, repeat_penalty=1.1)