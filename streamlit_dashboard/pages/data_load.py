import streamlit as st
import pandas as pd 

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
    st.title("Data Load Page")

    train_df, valid_df, test_df = load_data()
    dataset_option = st.selectbox("Select Dataset", ["Train", "Validation", "Test"])

    if dataset_option == "Train":
        dataset = train_df
    elif dataset_option == "Validation":
        dataset = valid_df
    else:
        dataset = test_df

    ## base preprocessing
    train_df['class'] = train_df['class'].astype('category')
    valid_df['class'] = valid_df['class'].astype('category')
    test_df['class'] = test_df['class'].astype('category')

    train_text_cols, train_class_cols = split_columns(train_df) # 각 데이터셋의 컬럼 나누기
    valid_text_cols, valid_class_cols = split_columns(valid_df)
    test_text_cols, test_class_cols = split_columns(test_df)
        
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