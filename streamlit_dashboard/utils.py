import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch

## --------------- Load Data --------------- ##

def load_data():
    if 'train_df' not in st.session_state or 'valid_df' not in st.session_state or 'test_df' not in st.session_state:
        st.error("Datasets are not loaded. Please upload the datasets in the Upload Data tab.")
        return None, None, None, None

    # 세션 상태에서 데이터 가져오기
    train_df = st.session_state['train_df']
    valid_df = st.session_state['valid_df']
    test_df = st.session_state['test_df']

    # 컬럼 정보 추출
    train_text_col, train_class_cols = split_columns(train_df)
    valid_text_col, valid_class_cols = split_columns(valid_df)
    test_text_col, test_class_cols = split_columns(test_df)

    column_info = {
        "train": {"text_col": train_text_col, "class_cols": train_class_cols},
        "valid": {"text_col": valid_text_col, "class_cols": valid_class_cols},
        "test": {"text_col": test_text_col, "class_cols": test_class_cols},
    }

    return train_df, valid_df, test_df, column_info

def split_columns(df):
    # text_col : 가장 긴 문자열을 가진 컬럼 선택
    text_col = max(
        (col for col in df.columns if pd.api.types.is_string_dtype(df[col])),
        key=lambda col: df[col].dropna().astype(str).apply(len).max(),
        default=None,
    )
    # ckass_col : 나머지 컬럼들을 클래스 컬럼으로 구분
    class_cols = [col for col in df.columns if col != text_col]
    if class_cols:
        df[class_cols] = df[class_cols].astype('category')
    
    return text_col, class_cols

def upload_and_store_data():
    train_file = st.file_uploader("▶️ Train Dataset", type=["csv"])
    valid_file = st.file_uploader("▶️ Validation Dataset", type=["csv"])
    test_file = st.file_uploader("▶️ Test Dataset", type=["csv"])

    if train_file and valid_file and test_file:
        train_df = pd.read_csv(train_file)
        valid_df = pd.read_csv(valid_file)
        test_df = pd.read_csv(test_file)

        # 세션 상태에 데이터 저장
        st.session_state['train_df'] = train_df
        st.session_state['valid_df'] = valid_df
        st.session_state['test_df'] = test_df

        st.success("Data uploaded and stored successfully!")
        return train_df, valid_df, test_df
    else:
        st.error("Please upload all three files!")
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

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

    def generate_embeddings(self, dataframe, text_col, max_len=128, batch_size=32):
        # 데이터셋 준비
        dataset = self.CustomDataset(dataframe, text_col)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 임베딩 생성
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = self.tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device)
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 임베딩
                embeddings.append(cls_embeddings.cpu())
        
        return torch.cat(embeddings).numpy()
