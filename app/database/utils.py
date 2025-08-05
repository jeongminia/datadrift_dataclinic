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

# top keywords 저장
from collections import Counter
import re

def extract_top_keywords_from_train(n_top=5):
    """Train 데이터에서 상위 키워드 n개 추출하고 세션에 저장"""
    # Train 데이터 불러오기
    train_df = st.session_state['dataset_summary'].get('Train', {}).get('preview', None)
    if train_df is None:
        st.warning("Train 데이터가 없습니다.")
        return

    # 모든 텍스트 합치기 (train 데이터의 text 컬럼이 있다고 가정)
    all_text = " ".join(train_df['text'].astype(str).tolist())

    # 간단한 토큰화: 단어 단위로 자르기
    tokens = re.findall(r'\w+', all_text)

    # 너무 짧은 단어 제외 (예: 한 글자, 특수문자 등)
    tokens = [token for token in tokens if len(token) > 1]

    # 가장 많이 등장한 단어 뽑기
    counter = Counter(tokens)
    top_keywords = [word for word, count in counter.most_common(n_top)]

    # 세션에 저장
    st.session_state['top_keywords'] = top_keywords

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

# 전처리 함수
import re
from collections import Counter

def get_stats(df, name):
    if df is None:
        return f"{name}: 데이터 없음"
    # 텍스트 컬럼 자동 탐색
    text_col_candidates = ['text', 'comments', '본문', '내용', 'comment', 'sentence']
    text_col = None
    for col in text_col_candidates:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        # 가장 긴 문자열 컬럼 자동 선택 (백업)
        text_col = max(
            (col for col in df.columns if df[col].dtype == object),
            key=lambda col: df[col].dropna().astype(str).str.len().max(),
            default=None
        )
    if text_col is None:
        return f"{name}: 텍스트 컬럼 없음"
    total_docs = len(df)
    avg_length = int(df[text_col].astype(str).apply(lambda x: len(x.split())).mean())
    text = " ".join(df[text_col].astype(str).tolist())
    tokens = re.findall(r'\w+', text)
    tokens = [token for token in tokens if len(token) > 1]

    # 조사 제거 함수
    def remove_josa(token):
        return re.sub(r'(이|가|은|는|을|를|와|과|도|로|에|의|에서|에게|께|한테|부터|까지|보다|처럼|만큼|밖에|마다|조차|까지도|이라도|라도|이나|나|이며|든지|든가)$', '', token)

    tokens = [remove_josa(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]
    stopwords = [
        "이", "가", "은", "는", "을", "를", "에", "의", "와", "과", "도", "로", "에서", "에게", "께", "한테", "부터", "까지",
        "보다", "처럼", "만큼", "밖에", "마다", "조차", "까지도", "이라도", "라도", "이나", "나", "이며", "든지", "든가"
    ]
    tokens = [token for token in tokens if token not in stopwords]
    counter = Counter(tokens)
    top_keywords = [word for word, count in counter.most_common(5)]
    return (
        f"{name} - 문서 수: {total_docs}, 평균 문장 길이: {avg_length} 단어, "
        f"주요 키워드: {', '.join(top_keywords)}"
    )

def gen_summarization() -> str:
    """Train/Validation/Test 통계를 모두 반영하여 데이터 특성 요약"""
    # 각 데이터셋에서 통계 추출
    train_df = st.session_state.get('train_df')
    valid_df = st.session_state.get('valid_df')
    test_df = st.session_state.get('test_df')

    train_stats = get_stats(train_df, "Train")
    valid_stats = get_stats(valid_df, "Validation")
    test_stats = get_stats(test_df, "Test")

    context = f"""
    {train_stats}
    {valid_stats}
    {test_stats}
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