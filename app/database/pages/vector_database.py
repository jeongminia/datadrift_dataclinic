import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType, connections
from ..utils import EmbeddingPipeline, split_columns, get_data_from_session

# Milvus 서버에 연결
connections.connect("default", host="localhost", port="19530")

# ----------------------------- Vector Database Functions -----------------------------
## MilvusDB Schema
def create_collection(collection_name):
    """통합 컬렉션 생성 (데이터 + 메타데이터)"""
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="set_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="class", dtype=DataType.VARCHAR, max_length=50), 
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            # 메타데이터 필드들
            FieldSchema(name="dataset_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="summary_dict", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="data_previews", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="class_dist_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="doc_len_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="doc_len_table", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="wordcloud_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            # 데이터 드리프트 필드
            FieldSchema(name="dimension", dtype=DataType.FLOAT),
            FieldSchema(name="embedding_size", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="original_distance_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="PCA_distance_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="PCA_visualization_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="drift_score_summary", dtype=DataType.VARCHAR, max_length=10000),

        ]
        schema = CollectionSchema(fields=fields, description="Text Embeddings with Metadata")
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="vector", 
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        )
        st.write(f"✅ Created collection: {collection_name}")
    else:
        collection = Collection(name=collection_name)
    collection.load()
    return collection

# ----------------------------------- Prepare Metadata ----------------------------------- 
## 객체를 JSON 직렬화 가능하게 변환: 복잡한 객체들을 문자열이나 기본 타입으로 변환
def make_json_serializable(obj):
    try:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            df_copy = obj.copy()
            for col in df_copy.columns:
                # 컬럼이 범주형이면 문자열로 변환
                if pd.api.types.is_categorical_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].astype(str)
                # NaN 값을 빈 문자열로 변환
                df_copy[col] = df_copy[col].fillna("")
            return df_copy.to_dict("records")
        elif isinstance(obj, pd.Series):
            # Categorical Series를 문자열로 변환
            if pd.api.types.is_categorical_dtype(obj):
                return obj.astype(str).fillna("").tolist()
            return obj.fillna("").tolist()
        elif isinstance(obj, pd.Categorical):
            # Categorical 객체를 직접 처리
            return obj.astype(str).tolist()
        elif isinstance(obj, pd.CategoricalDtype):
            # CategoricalDtype 객체를 문자열로 변환
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif pd.isna(obj):
            return ""
        elif hasattr(obj, 'dtype'):
            # pandas dtype이 있는 객체들 처리
            if 'category' in str(obj.dtype):
                return str(obj)
            elif 'object' in str(obj.dtype):
                return str(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return str(obj)
    except Exception as e:
        # 모든 예외를 문자열로 변환하여 안전하게 처리
        return str(obj)
        
## Session State에서 메타데이터 추출 후 직렬화
def prepare_metadata():
    summary_dict = st.session_state.get("dataset_summary", {})
    class_dist_path = st.session_state.get("class_dist_path", "")
    doc_len_path = st.session_state.get("doc_len_path", "")
    doc_len_table = st.session_state.get("doc_len_table", "")
    wordcloud_path = st.session_state.get("wordcloud_path", "")
    dataset_name = st.session_state.get("dataset_name", "")
        
    train_df, valid_df, test_df = get_data_from_session()
    data_previews = {}
        
    if train_df is not None and len(train_df) > 0:
        # Categorical 컬럼을 미리 처리한 후 preview 생성
        train_preview = train_df.head(5).copy()
        for col in train_preview.columns:
            if pd.api.types.is_categorical_dtype(train_preview[col]):
                train_preview[col] = train_preview[col].astype(str)
        preview_data = make_json_serializable(train_preview.to_dict('records'))
        data_previews["train"] = {"data": preview_data,
                                "total_rows": len(train_df),
                                "columns": list(train_df.columns)
                                }
        
    if valid_df is not None and len(valid_df) > 0:
        # Categorical 컬럼을 미리 처리한 후 preview 생성
        valid_preview = valid_df.head(5).copy()
        for col in valid_preview.columns:
            if pd.api.types.is_categorical_dtype(valid_preview[col]):
                valid_preview[col] = valid_preview[col].astype(str)
        preview_data = make_json_serializable(valid_preview.to_dict('records'))
        data_previews["valid"] = {"data": preview_data,
                                "total_rows": len(valid_df),
                                "columns": list(valid_df.columns)
                                }
            
    if test_df is not None and len(test_df) > 0:
        # Categorical 컬럼을 미리 처리한 후 preview 생성
        test_preview = test_df.head(5).copy()
        for col in test_preview.columns:
            if pd.api.types.is_categorical_dtype(test_preview[col]):
                test_preview[col] = test_preview[col].astype(str)
        preview_data = make_json_serializable(test_preview.to_dict('records'))
        data_previews["test"] = {"data": preview_data,
                                "total_rows": len(test_df),
                                "columns": list(test_df.columns)
                                }

    paths = {"class_dist_path": os.path.abspath(class_dist_path) if class_dist_path else "",
            "doc_len_path": os.path.abspath(doc_len_path) if doc_len_path else "",
            "wordcloud_path": os.path.abspath(wordcloud_path) if wordcloud_path else ""
            }
        
    # 안전한 JSON 직렬화
    summary_dict_json = ""
    doc_len_table_json = ""
    data_previews_json = ""
        
    if summary_dict:
        try:
            summary_dict_json = json.dumps(make_json_serializable(summary_dict), default=str)
        except Exception as e:
            st.warning(f"summary_dict 직렬화 실패: {e}")
            summary_dict_json = json.dumps(str(summary_dict))
        
    if doc_len_table:
        try:
            doc_len_table_json = json.dumps(make_json_serializable(doc_len_table), default=str)
        except Exception as e:
            st.warning(f"doc_len_table 직렬화 실패: {e}")
            doc_len_table_json = json.dumps(str(doc_len_table))
        
    if data_previews:
        try:
            data_previews_json = json.dumps(make_json_serializable(data_previews), default=str)
        except Exception as e:
            st.warning(f"data_previews 직렬화 실패: {e}")
            data_previews_json = json.dumps(str(data_previews))
        
    return {
            "dataset_name": dataset_name,
            "summary_dict": summary_dict_json,
            "data_previews": data_previews_json,
            "class_dist_path": paths["class_dist_path"],
            "doc_len_path": paths["doc_len_path"],
            "doc_len_table": doc_len_table_json,
            "wordcloud_path": paths["wordcloud_path"],
            "timestamp": int(time.time())
        }

# ----------------------------------- Save Metadata ----------------------------------- 
def save_metadata_to_vectordb(collection_name):
    collection = Collection(name=collection_name)
    collection.load()
        
    # 기존 메타데이터 삭제
    metadata_results = collection.query(
            expr="set_type == 'metadata'",
            output_fields=["id"],
            limit=1
        )
        
    if metadata_results:
        metadata_id = metadata_results[0]["id"]
        collection.delete(f"id == {metadata_id}")
        
    # 새 메타데이터 준비
    metadata = prepare_metadata()
    dummy_vector = [0.0] * 768
        
    # 메타데이터 삽입 (데이터 드리프트 필드 포함)
    data = [
            ["metadata"],
            ["metadata"],
            [dummy_vector],
            [metadata["dataset_name"]],
            [metadata["summary_dict"]],
            [metadata["data_previews"]],
            [metadata["class_dist_path"]],
            [metadata["doc_len_path"]],
            [metadata["doc_len_table"]],
            [metadata["wordcloud_path"]],
            [metadata["timestamp"]],
            # 데이터 드리프트 필드들을 빈 값으로 초기화
            [0.0],  # dimension
            [""],   # embedding_size
            [""],   # original_distance_path
            [""],   # PCA_distance_path
            [""],   # PCA_visualization_path
            [""],   # drift_score_summary
        ]
        
    fields = ["set_type", "class", "vector", "dataset_name", "summary_dict", 
                 "data_previews", "class_dist_path", "doc_len_path", "doc_len_table", "wordcloud_path", "timestamp",
                 "dimension", "embedding_size", "original_distance_path", 
                 "PCA_distance_path", "PCA_visualization_path", "drift_score_summary"]
        
    result = collection.insert(data, fields=fields)
    collection.flush()
        
    st.success(f"✅ 메타데이터가 저장되었습니다.")
    return result.primary_keys[0]

# ----------------------------------- Save Vector ----------------------------------- 
## text Embedding 벡터를 batch 단위로 벡터DB에 저장
def insert_vectors(collection_name, vectors, set_type, class_labels, batch_size=500, metadata=None):
    collection = Collection(name=collection_name)
    class_labels = [str(label) if not isinstance(label, str) else label for label in class_labels]
    total = len(vectors)
    ids = []
    
    # 메타데이터 삽입 (train 데이터와 함께 처리)
    if metadata and set_type == "train":
        dummy_vector = [0.0] * 768
        
        metadata_data = [
            ["metadata"],
            ["metadata"],
            [dummy_vector],
            [metadata["dataset_name"]],
            [metadata["summary_dict"]],
            [metadata["data_previews"]], 
            [metadata["class_dist_path"]],
            [metadata["doc_len_path"]],
            [metadata["doc_len_table"]],
            [metadata["wordcloud_path"]],
            [metadata["timestamp"]],
            # Drift 관련 필드들을 빈 값으로 초기화
            [0.0],  # dimension
            [""],   # embedding_size
            [""],   # original_distance_path
            [""],   # PCA_distance_path
            [""],   # PCA_visualization_path
            [""],   # drift_score_summary
        ]
        
        fields = ["set_type", "class", "vector", "dataset_name", "summary_dict", 
                 "data_previews", "class_dist_path", "doc_len_path", "doc_len_table", "wordcloud_path", "timestamp",
                 "dimension", "embedding_size", "original_distance_path", 
                 "PCA_distance_path", "PCA_visualization_path", "drift_score_summary"]
        
        metadata_ids = collection.insert(metadata_data, fields=fields)
        ids.append(metadata_ids)
        collection.flush()
    
    # 일반 데이터 삽입
    for i in range(0, total, batch_size):
        batch_vectors = vectors[i:i+batch_size]
        batch_labels = class_labels[i:i+batch_size]
        batch_set_type = [set_type] * len(batch_vectors)
        
        # 메타데이터 필드들은 빈 값으로 설정
        empty_metadata = [""] * len(batch_vectors)
        zero_timestamp = [0] * len(batch_vectors)
        zero_dimension = [0.0] * len(batch_vectors)  # Drift 필드용
        
        data = [
            batch_set_type,
            batch_labels,
            batch_vectors,
            empty_metadata,  # dataset_name
            empty_metadata,  # summary_dict
            empty_metadata,  # data_previews
            empty_metadata,  # class_dist_path
            empty_metadata,  # doc_len_path
            empty_metadata,  # doc_len_table
            empty_metadata,  # wordcloud_path
            zero_timestamp,  # timestamp
            zero_dimension,  # dimension
            empty_metadata,  # embedding_size
            empty_metadata,  # original_distance_path
            empty_metadata,  # PCA_distance_path
            empty_metadata,  # PCA_visualization_path
            empty_metadata,  # drift_score_summary
        ]
        
        fields = ["set_type", "class", "vector", "dataset_name", "summary_dict", 
                 "data_previews", "class_dist_path", "doc_len_path", "doc_len_table", "wordcloud_path", "timestamp",
                 "dimension", "embedding_size", "original_distance_path", 
                 "PCA_distance_path", "PCA_visualization_path", "drift_score_summary"]
        
        batch_ids = collection.insert(data, fields=fields)
        ids.append(batch_ids)
        collection.flush()
    
    return ids

# ----------------------------------- Load Metadata ----------------------------------- 
## JSON
def load_metadata_from_vectordb(collection_name):
    try:
        if not utility.has_collection(collection_name):
            return None
        
        collection = Collection(name=collection_name)
        collection.load()
        
        # 메타데이터 조회 (데이터 드리프트 필드 포함)
        results = collection.query(
            expr="set_type == 'metadata'",
            output_fields=["dataset_name", "summary_dict", "data_previews", "class_dist_path", 
                          "doc_len_path", "doc_len_table", "wordcloud_path", "timestamp",
                          "dimension", "embedding_size", "original_distance_path", 
                          "PCA_distance_path", "PCA_visualization_path", "drift_score_summary"],
            limit=1
        )
        
        if not results:
            return None
            
        result = results[0]
        
        # 세션 스테이트에 로드
        st.session_state["dataset_name"] = result["dataset_name"]
        st.session_state["class_dist_path"] = result["class_dist_path"]
        st.session_state["doc_len_path"] = result["doc_len_path"]
        st.session_state["wordcloud_path"] = result["wordcloud_path"]
        
        # JSON 파싱
        if result["summary_dict"]:
            try:
                st.session_state["dataset_summary"] = json.loads(result["summary_dict"])
            except:
                st.session_state["dataset_summary"] = {}
        
        # data_previews 파싱 (JSON 형태로 저장됨)
        if result.get("data_previews"):
            try:
                st.session_state["data_previews"] = json.loads(result["data_previews"])
            except:
                st.session_state["data_previews"] = {}
        else:
            st.session_state["data_previews"] = {}
        
        if result["doc_len_table"]:
            try:
                st.session_state["doc_len_table"] = json.loads(result["doc_len_table"])
            except:
                st.session_state["doc_len_table"] = result["doc_len_table"]
        else:
            st.session_state["doc_len_table"] = ""
        
        return result
        
    except Exception as e:
        st.error(f"❌ 메타데이터 로드 실패: {e}")
        return None

## numpy 
def load_and_save_data(data, collection_name, set_type, class_labels, metadata=None):
    """데이터를 로드하고 벡터DB에 저장"""
    create_collection(collection_name)
    vectors = np.array(data)
    ids = insert_vectors(collection_name, vectors, set_type, class_labels, metadata=metadata)
    total_inserted = sum(batch.insert_count for batch in ids)
    st.write(f"✅ {set_type} data (size: {total_inserted}) successfully inserted into {collection_name} collection.")
    return ids

#  --------------------------------------------- Main ---------------------------------------------
def render():
    """Vector Database 페이지 렌더링"""
    if 'dataset_name' not in st.session_state:
        st.error("데이터셋 이름이 설정되지 않았습니다. Upload Data 탭에서 데이터를 업로드해주세요.")
        return

    dataset_name = st.session_state['dataset_name']
    
    # 이전 세션 데이터 복원 시도
    if 'metadata_restored' not in st.session_state:
        st.info("이전 세션 데이터 확인 중...")
        restored_data = load_metadata_from_vectordb(dataset_name)
        if restored_data:
            st.session_state['metadata_restored'] = True
            st.success("✅ 이전 세션 데이터가 복원되었습니다.")
        else:
            st.session_state['metadata_restored'] = False

    # 데이터 로드
    train_df, valid_df, test_df = get_data_from_session()
    if train_df is None or valid_df is None or test_df is None:
        st.error("데이터셋이 로드되지 않았습니다. Upload Data 탭에서 데이터를 업로드해주세요.")
        return

    # 데이터 임베딩 및 저장
    text_col, class_col = split_columns(train_df)
    
    if not text_col or not class_col:
        st.error("데이터셋에 유효한 텍스트 또는 클래스 컬럼이 없습니다.")
        return
    
    # 임베딩 파이프라인 초기화
    embedding_pipeline = EmbeddingPipeline()
    embedding_pipeline.load_model()
    
    collection_name = dataset_name
    metadata = prepare_metadata()
    
    # 데이터 임베딩 및 저장
    with st.spinner("데이터 임베딩 중..."):
        # Train 데이터 (메타데이터와 함께)
        train_embeddings = embedding_pipeline.generate_embeddings(train_df, text_col)
        train_class_labels = train_df[class_col[0]].squeeze().tolist()
        load_and_save_data(train_embeddings, collection_name, "train", train_class_labels, metadata=metadata)
        
        # Validation 데이터
        valid_embeddings = embedding_pipeline.generate_embeddings(valid_df, text_col)
        valid_class_labels = valid_df[class_col[0]].squeeze().tolist()
        load_and_save_data(valid_embeddings, collection_name, "valid", valid_class_labels)
        
        # Test 데이터
        test_embeddings = embedding_pipeline.generate_embeddings(test_df, text_col)
        test_class_labels = test_df[class_col[0]].squeeze().tolist()
        load_and_save_data(test_embeddings, collection_name, "test", test_class_labels)
    
    # 결과 확인
    collection = Collection(name=collection_name)
    collection.load()
    
    # 각 set_type별 레코드 수 확인
    for set_type in ['train', 'valid', 'test', 'metadata']:
        results = collection.query(expr=f"set_type == '{set_type}'", output_fields=["set_type"], limit=10000)
        #st.write(f"🔍 {set_type.capitalize()} 레코드: {len(results)}개")
    
    st.success("✅ 모든 데이터셋이 Vector Database에 성공적으로 저장되었습니다.")
    
    # 전체 데이터 타입 요약
    results = collection.query(
        expr="set_type in ['train', 'valid', 'test', 'metadata']", 
        output_fields=["set_type"], 
        limit=10000
    )
    set_types = [res["set_type"] for res in results]
    unique_types = set(set_types)
    st.write("📊 컬렉션에 저장된 데이터 타입:", unique_types)