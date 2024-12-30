import os
import pandas as pd
import numpy as np
import streamlit as st

def get_uploaded_data():
    train_file = st.file_uploader("Upload Train Data CSV", type="csv", key="train_data_upload")
    valid_file = st.file_uploader("Upload Validation Data CSV", type="csv", key="valid_data_upload")
    test_file = st.file_uploader("Upload Test Data CSV", type="csv", key="test_data_upload")
    
    if st.button("Process Uploaded Data"):
        if train_file and valid_file and test_file:
            # 데이터프레임 읽기
            train_df = pd.read_csv(train_file)
            valid_df = pd.read_csv(valid_file)
            test_df = pd.read_csv(test_file)

            # 세션에 데이터 저장
            st.session_state['train_df'] = train_df
            st.session_state['valid_df'] = valid_df
            st.session_state['test_df'] = test_df
            st.success("Data uploaded and stored successfully!")
        else:
            st.error("Please upload all three files!")

def render():
    st.subheader("Upload your datasets")
    st.write("Please upload your train, validation, and test datasets.")

    # 업로드된 데이터를 가져오기
    train_df, valid_df, test_df = get_uploaded_data()

    # 데이터 업로드 상태 확인
    if not (train_df and valid_df and test_df):
        st.warning("Please upload all three datasets (train, validation, test).")