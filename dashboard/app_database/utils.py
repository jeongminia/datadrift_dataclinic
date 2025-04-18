import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

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

def upload_and_store_data():
    dataset_name = st.text_input("Enter the name for the dataset :", "")
    if dataset_name:
        st.session_state['dataset_name'] = dataset_name

    train_file = st.file_uploader(f"▶️ Train Dataset of {dataset_name}", type=["csv"])
    valid_file = st.file_uploader(f"▶️ Validation Dataset of {dataset_name}", type=["csv"])
    test_file = st.file_uploader(f"▶️ Test Dataset of {dataset_name}", type=["csv"])

    # 업로드 상태 확인 및 경고 메시지
    if not train_file:
        st.warning("Train dataset is missing!")
    if not valid_file:
        st.warning("Validation dataset is missing!")
    if not test_file:
        st.warning("Test dataset is missing!")

    # 파일이 모두 업로드된 경우
    if train_file and valid_file and test_file:
        train_df = pd.read_csv(train_file)
        valid_df = pd.read_csv(valid_file)
        test_df = pd.read_csv(test_file)

        # 세션 상태에 데이터 저장
        st.session_state['train_df'] = train_df
        st.session_state['valid_df'] = valid_df
        st.session_state['test_df'] = test_df

        st.success("✅ successfully uploaded!")
        return train_df, valid_df, test_df
    
    # 하나라도 누락된 경우
    return None, None, None


def get_data_from_session():
    train_df = st.session_state.get('train_df')
    valid_df = st.session_state.get('valid_df')
    test_df = st.session_state.get('test_df')
    return train_df, valid_df, test_df

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

    def calculate_max_len(self, dataframe, text_col, thresholds=[64, 128, 256, 512]):
        token_lengths = [len(self.tokenizer.encode(text)) for text in dataframe[text_col]]
        suggested_max_len = np.percentile(token_lengths, 90)  # 90th percentile
        max_len = min(thresholds, key=lambda x: abs(x - suggested_max_len))
        return max_len

    def generate_embeddings(self, dataframe, text_col, max_len=128, batch_size=16):
        # 데이터셋 준비
        dataset = self.CustomDataset(dataframe, text_col)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 임베딩 생성
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                try:
                    inputs = self.tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device)
                    outputs = self.model(**inputs)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 임베딩
                    embeddings.append(cls_embeddings.cpu())
                except Exception as e:
                    st.error(f"Error in generating embeddings for batch: {e}")
        
        return torch.cat(embeddings).numpy()

## --------------- LLM Explainer --------------- ##
from gpt4all import GPT4All

# 모델 로딩 (최초 1회)
model = GPT4All("gpt4all-j")  # 또는 절대 경로 지정 가능

def generate_explanation(context: str) -> str:
    prompt = f"""
    다음은 텍스트 데이터셋의 특성 요약입니다. 아래 내용을 기반으로 분석 결과를 이해하기 쉽게 설명해 주세요.

    {context}

    설명:
    """
    return model.generate(prompt, max_tokens=300, temp=0.7).strip()
