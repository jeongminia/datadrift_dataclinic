import streamlit as st
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
import numpy as np
from utils import EmbeddingPipeline, split_columns, get_data_from_session

client = MilvusClient()

def create_collection(collection_name):
    if not client.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="set_type", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="class", dtype=DataType.VARCHAR, max_length=50), 
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768) 
        ]
        schema = CollectionSchema(fields=fields, description="Text Embeddings")
        client.create_collection(collection_name, schema)

def insert_vectors(collection_name, vectors, set_type, class_labels):
    # 데이터셋 타입과 클래스 정보를 메타데이터로 추가
    data = [
        {"name": "vector", "values": vectors},
        {"name": "set_type", "values": [set_type] * len(vectors)},
        {"name": "class", "values": class_labels}
    ]
    ids = client.insert(collection_name, data)
    return ids

def load_and_save_data(data, collection_name, set_type, class_labels):
    create_collection(collection_name)
    vectors = np.array(data)
    ids = insert_vectors(collection_name, vectors, set_type, class_labels)
    return ids

def render():
   # st.title("Embedding and Save to Vector Database")

    train_df, valid_df, test_df = get_data_from_session()
    if train_df is None or valid_df is None or test_df is None:
        st.error("Datasets are not loaded. Please upload the datasets in the Upload Data tab.")
        return

    # 데이터 임베딩
    st.subheader("Data Embedding")
    text_col, class_col = split_columns(train_df)  # 텍스트 컬럼과 클래스 컬럼 가져오기
    if text_col and class_col:
        embedding_pipeline = EmbeddingPipeline()
        embedding_pipeline.load_model()

        collection_name = "text_embeddings"

        # Train 데이터셋 임베딩
        train_embeddings = embedding_pipeline.generate_embeddings(train_df, text_col)
        train_class_labels = train_df[class_col].tolist()
        train_ids = load_and_save_data(train_embeddings, collection_name, "train", train_class_labels)
        st.write(f"Inserted vector IDs for train dataset: {train_ids}")

        # Validation 데이터셋 임베딩
        valid_embeddings = embedding_pipeline.generate_embeddings(valid_df, text_col)
        valid_class_labels = valid_df[class_col].tolist()
        valid_ids = load_and_save_data(valid_embeddings, collection_name, "valid", valid_class_labels)
        st.write(f"Inserted vector IDs for validation dataset: {valid_ids}")

        # Test 데이터셋 임베딩
        test_embeddings = embedding_pipeline.generate_embeddings(test_df, text_col)
        test_class_labels = test_df[class_col].tolist()
        test_ids = load_and_save_data(test_embeddings, collection_name, "test", test_class_labels)
        st.write(f"Inserted vector IDs for test dataset: {test_ids}")
    else:
        st.error("The dataset does not contain valid text or class columns.")

    return