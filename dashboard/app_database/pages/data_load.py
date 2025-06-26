import streamlit as st
import pandas as pd 
import warnings

# Import utils from parent directory
try:
    from ..utils import load_data, split_columns
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_data, split_columns

warnings.filterwarnings(action='ignore')

def get_summary_info(df):
    return {
        "preview": df.head(10),
        "description": df.describe(),
        "info": pd.DataFrame({
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Dtype": df.dtypes.values
        }, index=df.columns)
    }

def render():
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.subheader(f"{dataset_name} Data Load Page")

    train_df, valid_df, test_df = load_data()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Failed to load datasets. Please upload datasets in the 'Upload Data' tab.")
        return

    # 데이터 타입 정리
    train_text_cols, train_class_cols = split_columns(train_df)
    for df in [train_df, valid_df, test_df]:
        for col in df.columns:
            if col in train_text_cols:
                df[col] = df[col].astype('object')
            elif col in train_class_cols:
                df[col] = df[col].astype('category')

    # 결측값 확인
    if any(df.isnull().values.any() for df in [train_df, valid_df, test_df]):
        st.error("The dataset contains missing values. Please handle the missing values and upload the dataset again.")
        st.stop()

    # 세션에 데이터프레임 저장
    st.session_state['train_df'] = train_df
    st.session_state['valid_df'] = valid_df
    st.session_state['test_df'] = test_df

    # 세션에 summary 정보 저장
    st.session_state['dataset_summary'] = {
        "Train": get_summary_info(train_df),
        "Validation": get_summary_info(valid_df),
        "Test": get_summary_info(test_df)
    }

    # 인터페이스: 데이터셋 선택
    dataset_option = st.selectbox("Select Dataset", ["Train", "Validation", "Test"])
    selected_df = {
        "Train": train_df,
        "Validation": valid_df,
        "Test": test_df
    }[dataset_option]

    st.subheader(f"{dataset_option} Dataset Preview")
    st.dataframe(selected_df.head(10), use_container_width=True)

    st.subheader("Dataset Description")
    st.dataframe(selected_df.describe(), use_container_width=True)

    st.subheader("Dataset Information")
    info_df = pd.DataFrame({
        "Non-Null Count": selected_df.notnull().sum().values,
        "Null Count": selected_df.isnull().sum().values,
        "Dtype": selected_df.dtypes.values
    }, index=selected_df.columns)
    st.dataframe(info_df, use_container_width=True)

    st.success("The dataset is ready for the next step.")
