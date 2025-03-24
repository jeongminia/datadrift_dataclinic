import streamlit as st
from pymilvus import connections, Collection, utility

# Milvus 서버에 연결
connections.connect("default", host="localhost", port="19530")

def get_collection_names():
    return utility.list_collections()

def get_collection_fields(collection_name):
    collection = Collection(name=collection_name)
    return [field.name for field in collection.schema.fields]

def load_collection(collection_name):
    collection = Collection(name=collection_name)
    collection.load()

def query_collection(collection_name, expr="", output_fields=None, limit=100):
    collection = Collection(name=collection_name)
    if expr:
        results = collection.query(expr=expr, output_fields=output_fields)
    else:
        results = collection.query(expr="id >= 0", output_fields=output_fields, limit=limit)
    return results

def render():
    st.title("Load Embeddings from VectorDB")

    collection_names = get_collection_names()
    collection_name = st.selectbox("Select the collection name", options=collection_names)

    if st.button("Load Data"):
        # 컬렉션의 필드 이름 확인, 필드 설정
        fields = get_collection_fields(collection_name)
        output_fields = ["id", "set_type", "class", "vector"]
        valid_fields = [field for field in output_fields if field in fields]

        if not valid_fields:
            st.error("No valid fields to query in the selected collection.")
            return

        results = query_collection(collection_name, output_fields=valid_fields)

        # 세션 상태에 데이터 저장
        st.session_state['embedding_data'] = results
        st.success("✅ Embedding data successfully loaded and stored in session state.")

    return