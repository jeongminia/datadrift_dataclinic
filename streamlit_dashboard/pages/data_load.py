import streamlit as st
import pandas as pd 
import warnings
warnings.filterwarnings(action='ignore')

def load_data():
    train_df = pd.read_csv("data/train_data.csv")
    valid_df = pd.read_csv("data/val_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    return train_df, valid_df, test_df

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
        
    # 데이터셋 미리보기
    st.subheader(f"{dataset_option} Dataset Preview")
    st.dataframe(dataset.head(10), use_container_width=True)
    
    # 데이터 요약
    st.subheader("Dataset Description")
    st.dataframe(dataset.describe(), use_container_width=True)

    # 데이터 정보
    st.subheader("Dataset Information")
    info_dict = {
#    "Column": ,
    "Non-Null Count": dataset.notnull().sum().values,
    "Null Count": dataset.isnull().sum().values,
    "Dtype": dataset.dtypes.values
    }
    info_df = pd.DataFrame(info_dict, index=dataset.columns)
    st.dataframe(info_df, use_container_width=True)

    