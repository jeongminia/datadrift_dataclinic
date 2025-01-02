import streamlit as st
import pandas as pd 
import warnings
warnings.filterwarnings(action='ignore')
# data load
from utils import load_data

## --------------- main --------------- ##
def render():
    st.title("Data Load Page")

    train_df, valid_df, test_df, column_info = load_data()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return
    
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

    # 결측값이 있는 로우 제거
    initial_row_count = len(dataset)
    if dataset.isnull().values.any():
        missing_value_count = dataset.isnull().sum().sum()
        dataset = dataset.dropna()
        final_row_count = len(dataset)
        st.write(f"{dataset_option} dataset contained {missing_value_count} missing values. "
                 f"Initially, there were {initial_row_count} rows. After dropping rows with missing values, "
                 f"there are now {final_row_count} rows.")
    else:
        st.success(f"No missing values found in the {dataset_option} dataset.")

