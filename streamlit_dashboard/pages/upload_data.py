import os
import pandas as pd
import numpy as np
import streamlit as st

def get_uploaded_data():
    """
    CSV 파일 업로드 및 세션 상태에 저장.
    이미 세션에 데이터가 존재하면 업로드 없이 데이터 반환.
    """
    if 'train_df' in st.session_state and 'valid_df' in st.session_state and 'test_df' in st.session_state:
        st.success("Datasets are already loaded.")
        return st.session_state['train_df'], st.session_state['valid_df'], st.session_state['test_df']

    # 파일 업로드 위젯
    train_file = st.file_uploader("Upload Train Data CSV", type="csv", key="train_data_upload")
    valid_file = st.file_uploader("Upload Validation Data CSV", type="csv", key="valid_data_upload")
    test_file = st.file_uploader("Upload Test Data CSV", type="csv", key="test_data_upload")

    # 버튼 클릭 시 데이터 처리
    if st.button("Process Uploaded Data"):
        if train_file and valid_file and test_file:
            train_df = pd.read_csv(train_file)
            valid_df = pd.read_csv(valid_file)
            test_df = pd.read_csv(test_file)

            # 세션 상태에 데이터 저장
            st.session_state['train_df'] = train_df
            st.session_state['valid_df'] = valid_df
            st.session_state['test_df'] = test_df

            st.success("Data uploaded and stored successfully!")
            return train_df, valid_df, test_df
        else:
            st.error("Please upload all three files!")
    
    return None, None, None

# 데이터를 업로드하고 세션에 저장
def render():
    st.subheader("Upload your datasets")
    st.write("Please upload your train, validation, and test datasets.")

    get_uploaded_data()