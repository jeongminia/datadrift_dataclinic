import streamlit as st
from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType, connections
import numpy as np
from utils import EmbeddingPipeline, split_columns, get_data_from_session
 
# Milvus 서버에 연결
connections.connect("default", host="localhost", port="19530")
 
def create_collection(collection_name):
     if not utility.has_collection(collection_name):
         fields = [
             FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
             FieldSchema(name="set_type", dtype=DataType.VARCHAR, max_length=10),
             FieldSchema(name="class", dtype=DataType.VARCHAR, max_length=50), 
             FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768) 
         ]
         schema = CollectionSchema(fields=fields, description="Text Embeddings")
         collection = Collection(name=collection_name, schema=schema)
         collection.create_index(field_name="vector", 
                                 index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
         st.write(f"✅ Created and loaded collection: {collection_name}")
     else:
         collection = Collection(name=collection_name)
     collection.load()
 
def insert_vectors(collection_name, vectors, set_type, class_labels, batch_size=500):
    collection = Collection(name=collection_name)
    class_labels = [str(label) if not isinstance(label, str) else label for label in class_labels]
    total = len(vectors)
    ids = []
    for i in range(0, total, batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_labels = class_labels[i:i+batch_size]
        batch_set_type = [set_type] * len(batch_vectors)
        data = [
            batch_set_type,
            batch_labels,
            batch_vectors
            ]
        fields = ["set_type", "class", "vector"]
        batch_ids = collection.insert(data, fields=fields)
        ids.append(batch_ids)
        collection.flush()
    return ids

def load_and_save_data(data, collection_name, set_type, class_labels):
    create_collection(collection_name)
    vectors = np.array(data)
    ids = insert_vectors(collection_name, vectors, set_type, class_labels)
    # insert_count 합산
    total_inserted = sum(batch.insert_count for batch in ids)
    st.write(f"✅ Successfully saved {total_inserted} records for set_type: {set_type}.")
    return ids
 
def render():
     if 'dataset_name' not in st.session_state:
         st.error("Dataset name is not set. Please upload the datasets in the Upload Data tab.")
         return
 
     dataset_name = st.session_state['dataset_name']
 
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
 
         collection_name = dataset_name 
 
         train_embeddings = embedding_pipeline.generate_embeddings(train_df, text_col)
         train_class_labels = train_df[class_col[0]].squeeze().tolist()
         load_and_save_data(train_embeddings, collection_name, "train", train_class_labels)
 
         collection = Collection(name=collection_name)
         collection.load()
         train_results = collection.query(expr="set_type == 'train'", output_fields=["set_type"], limit=10000)
         st.write(f"🔍 Train records in collection: {len(train_results)}")
 
         valid_embeddings = embedding_pipeline.generate_embeddings(valid_df, text_col)
         valid_class_labels = valid_df[class_col[0]].squeeze().tolist()
         load_and_save_data(valid_embeddings, collection_name, "valid", valid_class_labels)
 
         valid_results = collection.query(expr="set_type == 'valid'", output_fields=["set_type"], limit=10000)
         st.write(f"🔍 Valid records in collection: {len(valid_results)}")
 
         test_embeddings = embedding_pipeline.generate_embeddings(test_df, text_col)
         test_class_labels = test_df[class_col[0]].squeeze().tolist()
         load_and_save_data(test_embeddings, collection_name, "test", test_class_labels)
 
         test_results = collection.query(expr="set_type == 'test'", output_fields=["set_type"], limit=10000)
         st.write(f"🔍 Test records in collection: {len(test_results)}")
 
         st.success("✅ All datasets (train, validation, test) have been successfully inserted into VectorDB.")
     else:
         st.error("The dataset does not contain valid text or class columns.")
         return
 
     collection = Collection(name=collection_name)
     collection.load()
     results = collection.query(expr="set_type in ['train', 'valid', 'test']", output_fields=["set_type"], limit=10000)
     set_types = [res["set_type"] for res in results]
     unique_types = set(set_types)
 
     st.write("📊 Current set_type values in collection:", unique_types)
     # 각 set_type별 데이터 개수 출력
     for set_type in unique_types:
         count = set_types.count(set_type)
         st.write(f"  {set_type}: {count} records")