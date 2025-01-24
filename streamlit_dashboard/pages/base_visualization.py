import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# wordcloud
import re
from pecab import PeCab
from collections import Counter
from wordcloud import WordCloud
from matplotlib import font_manager
import warnings
warnings.filterwarnings(action='ignore')
# evidently
import os
from evidently import ColumnMapping
import streamlit.components.v1 as components  # HTML 렌더링을 위한 Streamlit 컴포넌트
from evidently.metric_preset import TextEvals
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
# data load
from utils import load_data, split_columns

# HTML 저장 경로 설정
HTML_SAVE_PATH = "./reports"

# 나눔 폰트 경로를 직접 설정 
font_path = './fonts/NanumGothic.ttf'
fontprop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()

# 텍스트 전처리 함수
stopwords = ['아니', '근데', '진짜', '너무', 'ㅋㅋ', '이게', '그런데', 
             '정말', '그리고', 'ㅠ', 'ㅠㅠ', 'ㅋ', 'ㅎㅎ', '왜', '좀', '이거', '보고', 
             '그럼', '이제', '그래서', '그거', '그런', '그래', '그냥', 
             '뭐', '제발', '잘', '못', '안', '더', '이제']

analyzer = PeCab()

def clean_text(sent):
    sent_clean = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", sent)
    nouns = analyzer.nouns(sent_clean)
    filtered_words = [word for word in nouns if word not in stopwords]
    return ' '.join(filtered_words)

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
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Base Visualization Page of {dataset_name}")

    train_df, valid_df, test_df = load_data()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return
    
    train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기
    st.write(f"Text Column   : {train_text_cols}")
    st.write(f"Class Columns : {',  '.join(train_class_cols)}")
    
    datasets = {"Train": train_df, "Validation": valid_df, "Test": test_df}

    # evidently dashboard
    st.subheader("_Dashboard based_ :red[EvidentlyAI]")
    column_mapping = ColumnMapping(categorical_features=train_class_cols, 
                                   text_features=[train_text_cols])
        
    dashboard = Report(metrics=[DataDriftPreset(),
                                TextEvals(column_name=train_text_cols)
                                ])
    dashboard.run(reference_data=train_df, 
                  current_data=test_df, 
                  column_mapping=column_mapping)
    
    visaulization_report_path = os.path.join(HTML_SAVE_PATH, 
                                             f"{dataset_name}_visualization.html")
    dashboard.save_html(visaulization_report_path)
    with open(visaulization_report_path, "r") as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)

    # Displaying the text in Streamlit
    st.markdown("""
                 **▶️ All Descriptors**

                - **`TextLength`**: Calculates the length of the text
                - **`SentenceCount`**: Calculates the number of sentences in the text
                - **`Sentiment`**: Performs sentiment analysis on the text to identify emotional tone, negative (-1) - neutral - positive (1)
                - **`OOV`**: Measures the percentage of words in the text that are outside the defined vocabulary
                - **`NonLetterCharacterPercentage`**: Calculates the percentage of non-letter characters in the text
                """)



    if train_df.isnull().values.any() or valid_df.isnull().values.any() or test_df.isnull().values.any():
        st.error("One or more datasets contain missing values. Please handle the missing values and upload the datasets again.")
    else:
        ## 1. class column
        st.subheader("Class Column Analysis")

        for class_col in train_class_cols:
            st.write(f"Class Column: {class_col}")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

            for ax, (name, df) in zip(axes, datasets.items()):
                if class_col in df.columns:
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

        data_summary = []
        for name, df in datasets.items():
            text_col = train_text_cols if train_text_cols else None
            if text_col and text_col in df.columns:
                # text_col을 명시적으로 문자열로 변환하고 Null 값을 빈 문자열로 채움
                df[text_col] = df[text_col].astype(str).fillna("")
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