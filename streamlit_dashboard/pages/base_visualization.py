import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# wordcloud
import re
from collections import Counter
from wordcloud import WordCloud
from matplotlib import font_manager
import warnings
warnings.filterwarnings(action='ignore')
# data load
from utils import load_data, split_columns

# 나눔 폰트 경로를 직접 설정 
font_path = './fonts/NanumGothic.ttf'
fontprop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()

# 텍스트 전처리 함수
def clean_text(sent):
    sent_clean = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", sent)
    return sent_clean

# 워드클라우드 생성 함수
def generate_wordcloud(df, column, font_path):
    # 텍스트 전처리
    df['cleaned_facts'] = df[column].apply(clean_text)
    text_data = df['cleaned_facts'].tolist()
    text_data = [str(text) for text in text_data if isinstance(text, str)]
    token_sentences = [text.split() for text in text_data]  # 간단히 공백 기준으로 나눔
    counter = Counter([token for tokens in token_sentences for token in tokens])
    wc = WordCloud(font_path=font_path, background_color="white")
    return wc.generate_from_frequencies(counter)


## --------------- main --------------- ##
def render():
    st.title("Base Visualization Page")

    train_df, valid_df, test_df, column_info = load_data()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return
    
    train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기
    
    datasets = {"Train": train_df, "Validation": valid_df, "Test": test_df}

    ## 1. class column
    st.subheader("Class Column Analysis")
    st.write(train_class_cols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (name, df) in zip(axes, datasets.items()):
        class_col = train_class_cols[0] if train_class_cols else None  # class 컬럼 이름 가져오기
        if class_col and class_col in df.columns:
            class_counts = df[class_col].value_counts()
            ax.bar(list(class_counts.index), list(class_counts.values))
            ax.set_title(f'{name} Set Class Distribution')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
        else:
            ax.set_title(f'{name} Set Class Distribution')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.text(0.5, 0.5, 'No class column found', horizontalalignment='center', verticalalignment='center')
    
    st.pyplot(fig) # st.pyplot(fig, transparent=True)

    ## 2. Text column
    st.subheader("Text Column Analysis")
    st.write(train_text_cols)

    data_summary = []
    for name, df in datasets.items():
        text_col = train_text_cols if train_text_cols else None
        if text_col and text_col in df.columns:
            df['doc_len'] = df[text_col].apply(lambda words: len(words.split()))
            data_summary.append({
                "Dataset": name,
                "Longest Sentence": df['doc_len'].max(),
                "Shortest Sentence": df['doc_len'].min(),
                "Mean Sentence Length": int(round(df['doc_len'].mean())),
                "Sum of Sentences": len(df)
            })
        else:
            st.warning(f"No text column found in the {name} dataset.")

    # 2.1. Text Column Length
    st.write("Length Plot of Text Column")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (name, df) in zip(axes, datasets.items()):
        text_col = train_text_cols if train_text_cols else None
        if text_col and text_col in df.columns:
            mean_seq_len = np.round(df['doc_len'].mean()).astype(int)
            sns.histplot(df['doc_len'], kde=True, ax=ax, bins=10, label='Document lengths', color='blue')
            ax.axvline(x=mean_seq_len, color='k', linestyle='--', label=f'Seq length mean: {mean_seq_len}')
            ax.set_title(f'{name} Set Document Lengths')
            ax.set_xlabel('Length')
            ax.set_ylabel('Frequency')
            ax.legend()

    st.pyplot(fig) # st.pyplot(fig, transparent=True)
    st.write("Length Dataframe of Text Column")
    st.dataframe(pd.DataFrame(data_summary, index=datasets.keys()), use_container_width=True)


    # 2.2. Text Column Word Cloud    
    st.write("Word Cloud of Text Column")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, df) in zip(axes, datasets.items()):
        text_col = train_text_cols if train_text_cols else None
        if text_col and text_col in df.columns:
            cloud = generate_wordcloud(df, text_col, font_path)
            ax.imshow(cloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{name} WordCloud")
        else:
            st.warning(f"No text column found in the {name} dataset.")
        
    st.pyplot(fig)