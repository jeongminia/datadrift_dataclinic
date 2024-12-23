import streamlit as st
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import altair as alt
import io

# 0. 데이터 로드 ---------------------------------------------------- 
def load_data():
    train_df = pd.read_csv("data/train_data.csv")
    valid_df = pd.read_csv("data/val_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    return train_df, valid_df, test_df

train_df, valid_df, test_df = load_data()

## base preprocessing
train_df['class'] = train_df['class'].astype('category')
valid_df['class'] = valid_df['class'].astype('category')
test_df['class'] = test_df['class'].astype('category')

def split_columns(df):
    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    class_columns = df.select_dtypes(include=["int64", "float64", "category"]).columns.tolist()
    return text_columns, class_columns

train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기
valid_text_cols, valid_class_cols = split_columns(valid_df)
test_text_cols, test_class_cols = split_columns(test_df)

# 1. page 구성 ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Data Load", "Basic Visualization", "Embedding Visualization", "Detect Data Drift"])

# 2. p.data_load 구성 ------------------------------------ 

with tab1:
    st.title("Data Load Page")

    # Selectbox for dataset selection
    dataset_option = st.selectbox("Select Dataset", ["Train", "Validation", "Test"])

    # Placeholder for datasets (replace with actual dataset loading logic)
    if dataset_option == "Train":
        dataset = train_df
    elif dataset_option == "Validation":
        dataset = valid_df
    else:  # Test
        dataset = test_df

    # 데이터셋 미리보기
    st.subheader(f"{dataset_option} Dataset Preview")
    st.write(dataset.head(10))

    # 데이터 정보
    st.subheader("Dataset Description")
    st.write(dataset.describe())
    
    st.subheader("Dataset Information")
    info_dict = {
    "Column": dataset.columns,
    "Non-Null Count": dataset.notnull().sum().values,
    "Null Count": dataset.isnull().sum().values,
    "Dtype": dataset.dtypes.values
    }
    info_df = pd.DataFrame(info_dict)
    st.dataframe(info_df)

# 3. p.compare_visualization 구성 ---------------------------------- 

with tab2:
    st.title("Base Visualization Page")
    
    ## 3.1. class column
    st.subheader("Class Column Analysis")

    datasets = {"Train": train_df, "Validation": valid_df, "Test": test_df}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (name, df) in zip(axes, datasets.items()):
        class_col = train_class_cols[0]  # class 컬럼 이름 가져오기
        class_counts = df[class_col].value_counts()
        ax.bar(list(class_counts.index), list(class_counts.values))
        ax.set_title(f'{name} Set Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')

    st.pyplot(fig)

    ## 3.2. Text column
    st.subheader("Text Column Analysis")
        ### 3.2.1. Text Column Length

        ### 3.2.2. Text Column Word Cloud


# 4. p.embedding_visualization 구성 ---------------------------------- 
with tab3:
    st.title("Embedding Visualization Page")
    

# 5. p.embedding_visualization 구성 ---------------------------------- 
with tab4:
    st.title("Detect Data Drift Page")