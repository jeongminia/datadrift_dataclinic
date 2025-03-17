import streamlit as st
from pymilvus import MilvusClient
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Initialize Milvus client
client = MilvusClient()

# Function to insert vectors into the database
def insert_vectors(collection_name, vectors):
    ids = client.insert(collection_name, vectors)
    return ids

# Function to load data and save to vector database
def load_and_save_data(data, collection_name):
    vectors = np.array(data)
    ids = insert_vectors(collection_name, vectors)
    return ids

# Example usage
if __name__ == "__main__":
    data = np.random.rand(100, 128)  # Example data
    collection_name = "example_collection"
    ids = load_and_save_data(data, collection_name)
    st.write(f"Inserted vector IDs: {ids}")