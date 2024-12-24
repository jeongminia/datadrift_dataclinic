import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt

def load_data():
    train_df = pd.read_csv("data/train_data.csv")
    valid_df = pd.read_csv("data/val_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    return train_df, valid_df, test_df

def split_columns(df):
        text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        class_columns = df.select_dtypes(include=["int64", "float64", "category"]).columns.tolist()
        return text_columns, class_columns


def render():
    st.title("Base Visualization Page")

    train_df, valid_df, test_df = load_data()
    ## base preprocessing
    train_df['class'] = train_df['class'].astype('category')
    valid_df['class'] = valid_df['class'].astype('category')
    test_df['class'] = test_df['class'].astype('category')

    train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기
    valid_text_cols, valid_class_cols = split_columns(valid_df)
    test_text_cols, test_class_cols = split_columns(test_df)
    
    datasets = {"Train": train_df, "Validation": valid_df, "Test": test_df}
    ## 1. class column
    st.subheader("Class Column Analysis")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (name, df) in zip(axes, datasets.items()):
        class_col = train_class_cols[0]  # class 컬럼 이름 가져오기
        class_counts = df[class_col].value_counts()
        ax.bar(list(class_counts.index), list(class_counts.values))
        ax.set_title(f'{name} Set Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')

    st.pyplot(fig)

    ## 2. Text column
    st.subheader("Text Column Analysis")
    for name, df in datasets.items():
        df['doc_len'] = df.text.apply(lambda words: len(words.split()))
    
    # 2.1. Text Column Length
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (name, df) in zip(axes, datasets.items()):
        mean_seq_len = np.round(df['doc_len'].mean()).astype(int)
        sns.histplot(df['doc_len'], kde=True, ax=ax, bins=10, label='Document lengths', color='blue')
        ax.axvline(x=mean_seq_len, color='k', linestyle='--', label=f'Seq length mean: {mean_seq_len}')
        ax.set_title(f'{name} Set Document Lengths')
        ax.set_xlabel('Length')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.show()
    
    
    # 2.2. Text Column Word Cloud
