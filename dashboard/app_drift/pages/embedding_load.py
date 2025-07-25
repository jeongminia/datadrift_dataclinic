import streamlit as st
from pymilvus import connections, Collection, utility
import json
import datetime

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

def query_by_set_type(collection_name, set_types, output_fields, limit_per_type=10000):
    collection = Collection(name=collection_name)
    all_results = []

    for stype in set_types:
        expr = f"set_type == '{stype}'"
        results = collection.query(expr=expr, output_fields=output_fields, limit=limit_per_type)
        all_results.extend(results)

    return all_results

def query_collection(collection_name, expr="", output_fields=None, limit=None):
    collection = Collection(name=collection_name)
    if expr:
        results = collection.query(expr=expr, output_fields=output_fields)
    else:
        results = collection.query(expr="id >= 0", output_fields=output_fields, limit=limit or 100000)  # 충분히 큰 숫자
    return results

# 사용자에게 데이터 설명을 위함
def get_collection_metadata(collection_name):
    """컬렉션에서 대표 메타데이터(예: dataset_name, summary_dict, timestamp 등) 추출"""
    collection = Collection(name=collection_name)
    # 메타데이터 필드만 추출 (필요시 필드명 수정)
    meta_fields = ["dataset_name", "summary_dict", "timestamp"]
    # 실제 존재하는 필드만 사용
    fields = get_collection_fields(collection_name)
    valid_meta_fields = [f for f in meta_fields if f in fields]
    if not valid_meta_fields:
        return {}
    # 한 레코드만 조회
    results = collection.query(expr="id >= 0", output_fields=valid_meta_fields, limit=1)
    return results[0] if results else {}

# ----------------- main ------------------

def render():
    collection_names = get_collection_names()
    collection_name = st.selectbox("Select the collection name", options=collection_names)

    if collection_name:
        meta = get_collection_metadata(collection_name)
        st.session_state['dataset_name'] = meta.get('dataset_name')
        
        if meta:
            # Summary 포맷팅
            summary = meta.get('summary_dict', 'N/A')
            if isinstance(summary, dict):
                summary_str = json.dumps(summary, indent=2, ensure_ascii=False)
            else:
                summary_str = str(summary) if summary else 'N/A'

            # timestamp 변환
            ts = meta.get('timestamp', 'N/A')
            if isinstance(ts, (int, float, str)) and str(ts).isdigit():
                try:
                    ts_dt = datetime.datetime.fromtimestamp(int(ts))
                    ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts_str = str(ts)
            else:
                ts_str = str(ts)
            
            st.subheader(f"Collection Info")
            st.markdown(
                f"""
                <div style="background-color:#23272f;padding:18px 20px 18px 20px;border-radius:12px;border:1.5px solid #3a3f4b; margin-bottom:16px;">
                    <div style="font-size:17px;line-height:1.7;">
                        <b>📁 Dataset Name:</b> {meta.get('dataset_name')}<br>
                        <b>📝 Summary:</b><pre style="background:none;padding:0;margin:0 0 0 10px;color:#d1d5db;">{summary_str}</pre>
                        <b>⏰ Created At:</b> {ts_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No metadata found for this collection.")
    
    # 차원 축소 옵션 선택
    st.subheader("🔧 Analysis Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dimension Reduction**")
        dimension_options = [10, 50, 100, 200, 300, 400, 500]
        selected_dimension = st.selectbox(
            "Select Target Dimension:",
            options=dimension_options,
            index=2,  # 기본값: 100
            key="global_dimension_select"
        )
        st.session_state['selected_dimension'] = selected_dimension
    
    with col2:
        st.markdown("**Test Type for Drift Detection**")
        test_type_options = [
            "MMD",
            "Wasserstein Distance", 
            "KL Divergence",
            "JensenShannon Divergence",
            "Energy Distance"
        ]
        selected_test_type = st.selectbox(
            "Select Test Type:",
            options=test_type_options,
            index=0,  # 기본값: MMD
            key="global_test_type_select"
        )
        st.session_state['selected_test_type'] = selected_test_type
    
    # 선택된 설정 표시
    st.info(f"🎯 **Selected Configuration:** Dimension={selected_dimension}, Test Type={selected_test_type}")

    if st.button("Load Data"):

        # 컬렉션의 필드 이름 확인, 필드 설정
        fields = get_collection_fields(collection_name)
        output_fields = ["id", "set_type", "class", "vector"]
        valid_fields = [field for field in output_fields if field in fields]

        if not valid_fields:
            st.error("No valid fields to query in the selected collection.")
            return
        
        load_collection(collection_name)

        # 미리 set_type 값 추출
        collection = Collection(name=collection_name)
        set_type_results = collection.query(expr="id >= 0", output_fields=["set_type"], limit=10000)
        set_types = set([res["set_type"] for res in set_type_results])
        st.write(f"📌 Detected set_type values: `{set_types}`")

        results = query_by_set_type(collection_name, set_types, valid_fields)
        st.session_state['embedding_data'] = results
        st.success("✅ Embedding data successfully loaded and stored in session state.")

    return