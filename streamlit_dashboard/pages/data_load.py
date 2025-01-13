import streamlit as st
import pandas as pd 
import warnings
warnings.filterwarnings(action='ignore')
# data load
from utils import load_data, split_columns

## --------------- main --------------- ##
def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"{dataset_name} Data Load Page")

    train_df, valid_df, test_df = load_data()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return
    
    # 데이터 타입 변경
    train_text_cols, train_class_cols = split_columns(train_df)
    for col in train_df.columns:
        if col in train_text_cols:
            train_df[col] = train_df[col].astype('object')
        elif col in train_class_cols:
            train_df[col] = train_df[col].astype('category')

    for col in valid_df.columns:
        if col in train_text_cols:
            valid_df[col] = valid_df[col].astype('object')
        elif col in train_class_cols:
            valid_df[col] = valid_df[col].astype('category')

    for col in test_df.columns:
        if col in train_text_cols:
            test_df[col] = test_df[col].astype('object')
        elif col in train_class_cols:
            test_df[col] = test_df[col].astype('category')

    # 세션 상태에 변경된 데이터프레임 저장
    st.session_state['train_df'] = train_df
    st.session_state['valid_df'] = valid_df
    st.session_state['test_df'] = test_df

    dataset_option = st.selectbox("Select Dataset", ["Train", "Validation", "Test"])

    dataset = None
    if dataset_option == "Train":
        dataset = train_df
    elif dataset_option == "Validation":
        dataset = valid_df
    elif dataset_option == "Test":
        dataset = test_df

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

    # 결측값 확인
    if dataset.isnull().values.any():
        st.error("The dataset contains missing values. Please handle the missing values and upload the dataset again.")
        st.stop()

    st.success("The dataset is ready for the next step.")
