import streamlit as st
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch

## --------------- Load Data --------------- ##
# 현재 파일의 디렉토리를 기준으로 데이터 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 절대 경로
data_dir = os.path.join(base_dir, "data")

def load_data():
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "val_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    train_df['class'] = train_df['class'].astype('category')
    valid_df['class'] = valid_df['class'].astype('category')
    test_df['class'] = test_df['class'].astype('category')
    return train_df, valid_df, test_df

def split_columns(df):
        text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        class_columns = df.select_dtypes(include=["int64", "float64", "category"]).columns.tolist()
        return text_columns, class_columns


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
