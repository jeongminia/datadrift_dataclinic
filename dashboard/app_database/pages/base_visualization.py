
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pecab import PeCab
from collections import Counter
from wordcloud import WordCloud
from matplotlib import font_manager
import warnings
warnings.filterwarnings(action='ignore')
import os
import streamlit.components.v1 as components
# evidently 최신 버전용 import
from evidently.report import Report
from evidently.metrics import DataDriftMetric, TextDescriptorsMetric

# Import utils from parent directory
try:
    from ..utils import load_data, split_columns
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_data, split_columns

HTML_SAVE_PATH = "./reports"
font_path = './static/NanumGothic.ttf'
fontprop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()

stopwords = ['아니', '근데', '진짜', '너무', 'ㅋㅋ', '이게', '그런데', '정말', '그리고', 'ㅠ', 'ㅠㅠ', 'ㅋ', 'ㅎㅎ', '왜', '좀', '이거',
             '보고', '그럼', '이제', '그래서', '그거', '그런', '그래', '그냥', '뭐', '제발', '잘', '못', '안', '더', "만", "억", "원", "천",
             "등", "년", "일", "명", "월", "위", "중", '시', '주', '전', '후', '조', '게', '거', '듯', '씨', '때', '애', '이', '것', '나',
             '수', '건', '점', '줄']

analyzer = PeCab()

def clean_text(sent):
    sent_clean = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", sent)
    nouns = analyzer.nouns(sent_clean)
    filtered_words = [word for word in nouns if word not in stopwords]
    return ' '.join(filtered_words)

def generate_wordcloud(df, column, font_path):
    df['cleaned_facts'] = df[column].apply(clean_text)
    text_data = df['cleaned_facts'].tolist()
    text_data = [str(text) for text in text_data if isinstance(text, str)]
    token_sentences = [text.split() for text in text_data]
    counter = Counter([token for tokens in token_sentences for token in tokens])
    wc = WordCloud(font_path=font_path, background_color="white")
    return wc.generate_from_frequencies(counter), counter

def render():
    dataset_name = st.session_state.get('dataset_name')
    # st.subheader(f"Base Visualization of {dataset_name}")

    train_df, valid_df, test_df = load_data()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return

    train_text_cols, train_class_cols = split_columns(train_df)
    st.write(f"Text Column   : {train_text_cols}")
    st.write(f"Class Columns : {',  '.join(train_class_cols)}")

    # --- 여기서 cleaned_facts 컬럼을 각 데이터셋에 미리 생성 ---
    text_col = train_text_cols if train_text_cols else None
    if text_col:
        for df in [train_df, valid_df, test_df]:
            if text_col in df.columns:
                df[text_col] = df[text_col].astype(str).fillna("")
                df['cleaned_facts'] = df[text_col].apply(clean_text)
            else:
                df['cleaned_facts'] = ""
    # ------------------------------------------------------

    datasets = {"Train": train_df, "Validation": valid_df, "Test": test_df}


    # evidently 최신 버전용 column_mapping 및 metric 사용
    column_mapping = {
        "target": None,
        "prediction": None,
        "numerical_features": [],
        "categorical_features": train_class_cols,
        "text_features": train_text_cols if isinstance(train_text_cols, list) else ([train_text_cols] if train_text_cols else []),
    }
    metrics = [
        DataDriftMetric(),
    ]
    # 텍스트 컬럼이 있으면 TextDescriptorsMetric 추가
    if train_text_cols:
        if isinstance(train_text_cols, list):
            for col in train_text_cols:
                metrics.append(TextDescriptorsMetric(column_name=col))
        else:
            metrics.append(TextDescriptorsMetric(column_name=train_text_cols))

    dashboard = Report(metrics=metrics)
    dashboard.run(reference_data=train_df, current_data=test_df, column_mapping=column_mapping)

    visualization_report_path = os.path.join(HTML_SAVE_PATH, f"{dataset_name}_visualization.html")
    dashboard.save_html(visualization_report_path)
    with open(visualization_report_path, "r") as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)

    if any(df.isnull().values.any() for df in [train_df, valid_df, test_df]):
        st.error("One or more datasets contain missing values. Please handle the missing values and upload the datasets again.")
        return

    st.subheader("Class Column Analysis")
    for class_col in train_class_cols:
        st.write(f"Class Column: {class_col}")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for ax, (name, df) in zip(axes, datasets.items()):
            if class_col in df.columns:
                class_counts = df[class_col].value_counts()
                ax.bar(class_counts.index, class_counts.values)
                ax.set_title(f'{name} Set')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
            else:
                ax.set_title(f'{name} Set')
                ax.text(0.5, 0.5, 'No class column found', ha='center', va='center')
        save_path = os.path.join("reports", f"class_dist.png")
        fig.savefig(save_path, bbox_inches="tight")
        st.session_state["class_dist_path"] = save_path
        st.pyplot(fig)

    st.subheader("Text Column Analysis")
    data_summary = []
    for name, df in datasets.items():
        text_col = train_text_cols if train_text_cols else None
        if text_col and text_col in df.columns:
            df[text_col] = df[text_col].astype(str).fillna("")
            df['doc_len'] = df[text_col].apply(lambda words: len(words.split()))
            data_summary.append({
                "Dataset": name,
                "Longest Sentence": df['doc_len'].max(),
                "Shortest Sentence": df['doc_len'].min(),
                "Mean Sentence Length": int(round(df['doc_len'].mean())),
                "Sum of Sentences": len(df)
            })

        if name == "Train":
            st.session_state["total_docs"] = len(df)
            st.session_state["avg_length"] = int(round(df['doc_len'].mean()))

    st.write("Length Plot of Text Column")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (name, df) in zip(axes, datasets.items()):
        if train_text_cols and train_text_cols in df.columns:
            mean_seq_len = int(round(df['doc_len'].mean()))
            sns.histplot(df['doc_len'], kde=True, ax=ax, bins=10, color='blue')
            ax.axvline(x=mean_seq_len, color='k', linestyle='--', label=f'Mean: {mean_seq_len}')
            ax.set_title(f'{name} Set')
            ax.set_xlabel('Length')
            ax.set_ylabel('Frequency')
            ax.legend()
    save_path = os.path.join("reports", f"doc_len.png")
    fig.savefig(save_path, bbox_inches="tight")
    st.session_state["doc_len_path"] = save_path
    st.pyplot(fig)

    text_len_table = pd.DataFrame(data_summary, index=[d["Dataset"] for d in data_summary])
    st.write("Length Dataframe of Text Column")
    st.dataframe(text_len_table, use_container_width=True)
    st.session_state["doc_len_table"] = text_len_table.to_html(index=False)

    # 전체 요약 통계
    st.session_state["summary_total_docs"] = train_df.shape[0] + valid_df.shape[0] + test_df.shape[0]
    st.session_state["summary_avg_length"] = int(round(
        (train_df['doc_len'].mean() * len(train_df) +
         valid_df['doc_len'].mean() * len(valid_df) +
         test_df['doc_len'].mean() * len(test_df)) /
        (len(train_df) + len(valid_df) + len(test_df))
    ))
    all_cleaned_text = " ".join(
        train_df['cleaned_facts'].tolist() +
        valid_df['cleaned_facts'].tolist() +
        test_df['cleaned_facts'].tolist()
    )
    st.session_state["summary_top_keywords"] = [w for w, _ in Counter(all_cleaned_text.split()).most_common(5)]

    # 각 세트별 요약 통계
    for name, df in datasets.items():
        if "cleaned_facts" in df.columns:
            doc_len = df['doc_len'].mean()
            counter = Counter(" ".join(df['cleaned_facts']).split())
            stats = {
                "total_docs": len(df),
                "avg_length": int(round(doc_len)),
                "top_keywords": [w for w, _ in counter.most_common(5)]
            }
            st.session_state[f"{name.lower()}_stats"] = stats

    st.write("Word Cloud of Text Column")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, df) in zip(axes, datasets.items()):
        text_col = train_text_cols if train_text_cols else None
        if text_col and text_col in df.columns:
            cloud, counter = generate_wordcloud(df, text_col, font_path)
            
            if name == "Train":
                top_keywords = [word for word, count in counter.most_common(5)]
                st.session_state["top_keywords"] = top_keywords
                #st.write("💬 top_keywords:", st.session_state["top_keywords"])

            ax.imshow(cloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{name} WordCloud")
        else:
            ax.axis("off")
            ax.set_title(f"{name} (No Text Column)")
    save_path = os.path.join("reports", f"wordcloud.png")
    fig.savefig(save_path, bbox_inches="tight")
    st.session_state["wordcloud_path"] = save_path
    st.pyplot(fig)