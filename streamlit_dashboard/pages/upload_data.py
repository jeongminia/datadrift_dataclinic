import streamlit as st
from utils import upload_and_store_data

def render():
    st.subheader("Upload your datasets")
    st.write("Please upload your train, validation, and test datasets.")
    upload_and_store_data()