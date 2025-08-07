import os
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import json
from pymilvus import Collection, utility

# Detect DataDrift
from evidently.metrics import EmbeddingsDriftMetric
from evidently.report import Report
from evidently.metrics.data_drift.embedding_drift_methods import mmd
from evidently.metrics.data_drift.embedding_drift_methods import ratio
from evidently import ColumnMapping
import streamlit.components.v1 as components  # HTML 열론링을 위한 Streamlit 컨퍼런트

# HTML 저장 경로 설정
HTML_SAVE_PATH = "reports"

# reports 디렉토리가 없으면 생성
if not os.path.exists(HTML_SAVE_PATH):
    os.makedirs(HTML_SAVE_PATH)

#  --------------------------------------------- Update Drift Metadata ---------------------------------------------
def update_metadata_to_vectordb(dataset_name):
    """드리프트 관련 메타데이터를 Milvus Collection에 업데이트"""
    try:
        if not utility.has_collection(dataset_name):
            st.error(f"Collection '{dataset_name}'이 존재하지 않습니다.")
            return None
            
        collection = Collection(name=dataset_name)
        collection.load()
        
        # 기존 메타데이터 조회
        metadata_results = collection.query(
            expr="set_type == 'metadata'",
            output_fields=["id", "dataset_name", "summary_dict", "data_previews", "class_dist_path",
                          "doc_len_path", "doc_len_table", "wordcloud_path", "timestamp"],
            limit=1
        )
        
        if not metadata_results:
            st.error("기존 메타데이터를 찾을 수 없습니다.")
            return None
            
        existing_metadata = metadata_results[0]
        metadata_id = existing_metadata["id"]
        
        # 기존 메타데이터 삭제
        collection.delete(f"id == {metadata_id}")
        collection.flush()
        
        # 세션에서 드리프트 관련 데이터 가져오기
        embedding_size = st.session_state.get("embedding_overview_text", "")
        dimension = float(st.session_state.get("selected_dimension", 0))  # selected_dimension 사용
        drift_score_summary = st.session_state.get("drift_score_summary", "")
        original_distance_path = st.session_state.get("original_distance_path", "")
        pca_distance_path = st.session_state.get("PCA_distance_path", "")
        pca_visualization_path = st.session_state.get("PCA_visualization_path", "")
        
        # 새 메타데이터 삽입 (기존 데이터 + 드리프트 데이터)
        dummy_vector = [0.0] * 768
        
        data = [
            ["metadata"],                                    # set_type
            ["metadata"],                                    # class
            [dummy_vector],                                  # vector
            [existing_metadata["dataset_name"]],            # dataset_name
            [existing_metadata["summary_dict"]],            # summary_dict
            [existing_metadata["data_previews"]],           # data_previews
            [existing_metadata["class_dist_path"]],         # class_dist_path
            [existing_metadata["doc_len_path"]],            # doc_len_path
            [existing_metadata["doc_len_table"]],           # doc_len_table
            [existing_metadata["wordcloud_path"]],          # wordcloud_path
            [existing_metadata["timestamp"]],               # timestamp
            # 데이터 드리프트 필드들 업데이트
            [dimension],                                     # dimension
            [embedding_size],                                # embedding_size
            [original_distance_path],                        # original_distance_path
            [pca_distance_path],                            # PCA_distance_path
            [pca_visualization_path],                       # PCA_visualization_path
            [drift_score_summary],                          # drift_score_summary
        ]
        
        fields = ["set_type", "class", "vector", "dataset_name", "summary_dict", 
                 "data_previews", "class_dist_path", "doc_len_path", "doc_len_table", 
                 "wordcloud_path", "timestamp", "dimension", "embedding_size", 
                 "original_distance_path", "PCA_distance_path", "PCA_visualization_path", 
                 "drift_score_summary"]
        
        result = collection.insert(data, fields=fields)
        collection.flush()
        
        st.success("✅ 데이터 드리프트 메타데이터가 벡터DB에 저장되었습니다.")
        return result.primary_keys[0]
        
    except Exception as e:
        st.error(f"❌ 드리프트 메타데이터 업데이트 실패: {e}")
        return None


#  --------------------------------------------- Main ---------------------------------------------
def render():
    dataset_name = st.session_state.get('dataset_name')
    # st.title(f"Detect {dataset_name} DataDrift Page")

    if 'train_embeddings' not in st.session_state or 'valid_embeddings' not in st.session_state or 'test_embeddings' not in st.session_state:
        st.error("Embeddings are not available. Please generate embeddings in the 'Embedding Visualization' tab first.")
        return

    train_embeddings = st.session_state['train_embeddings']
    valid_embeddings = st.session_state['valid_embeddings']
    test_embeddings = st.session_state['test_embeddings']

    # 데이터 형식 확인 및 변환
    if not isinstance(train_embeddings, np.ndarray):
        train_embeddings = np.array(train_embeddings)
    if not isinstance(valid_embeddings, np.ndarray):
        valid_embeddings = np.array(valid_embeddings)
    if not isinstance(test_embeddings, np.ndarray):
        test_embeddings = np.array(test_embeddings)

    # PCA 적용된 임베딩이 있으면 사용 (embedding_visualization에서 생성된 것)
    if ('train_embeddings_pca' in st.session_state and 
        'test_embeddings_pca' in st.session_state and
        'selected_dimension' in st.session_state):

        train_embeddings = st.session_state['train_embeddings_pca']
        test_embeddings = st.session_state['test_embeddings_pca']
        selected_dim = st.session_state['selected_dimension']
        st.info(f"🎯 Using PCA-reduced embeddings from Visualization page (Dimension: {selected_dim})")
    else:
        st.warning("⚠️ Using original embeddings. Please visit Embedding Visualization page first to apply dimension reduction.")

    # evidentlyai - 데이터 드리프트 검사
    st.write("Train(reference)-Test(current) Data Drift Detection")
    reference_df = pd.DataFrame(train_embeddings, 
                                columns=[f"dim_{i}" for i in range(train_embeddings.shape[1])])
    current_df = pd.DataFrame(test_embeddings, 
                              columns=[f"dim_{i}" for i in range(test_embeddings.shape[1])])

    column_mapping = ColumnMapping(
        embeddings={'all_dimensions': reference_df.columns.tolist()}
    )

    # embedding_load에서 선택된 테스트 타입 사용
    if 'selected_test_type' in st.session_state:
        test_option = st.session_state['selected_test_type']
        st.success(f"✅ Using test type setting from Load page: **{test_option}**")
        st.write(f"Current test method: {test_option}")
    else:
        # 기본값 설정 (Load 페이지에서 설정하지 않은 경우)
        test_option = st.selectbox("Select Test Type", 
                                   ["MMD", "Wasserstein Distance", 
                                    "KL Divergence", "JensenShannon Divergence", 
                                    "Energy Distance"])
        st.warning("⚠️ Test type not set in Load page. Using local selection.")

    test_methods = {
        "MMD": mmd(threshold=0.015),
        "Wasserstein Distance": ratio(component_stattest='wasserstein', component_stattest_threshold=0.1, threshold=0.015),
        "KL Divergence": ratio(component_stattest='kl_div', component_stattest_threshold=0.1, threshold=0.015),
        "JensenShannon Divergence": ratio(component_stattest='jensenshannon', component_stattest_threshold=0.1, threshold=0.015),
        "Energy Distance": ratio(component_stattest='ed', component_stattest_threshold=0.1, threshold=0.015),
    }

    # 대시보드용 Report 생성 및 저장
    selected_method = test_methods[test_option]
    visual_report = Report(metrics=[EmbeddingsDriftMetric('all_dimensions', drift_method=selected_method)])
    visual_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

    html_path = os.path.join(HTML_SAVE_PATH, f"{dataset_name}_drift_report.html")
    visual_report.save_html(html_path)
    with open(html_path, "r") as f:
        st.session_state['drift_report_html'] = f.read()

    # 모든 방법에 대한 드리프트 점수 요약 저장
    drift_summary = []
    for name, method in test_methods.items():
        try:
            temp_report = Report(metrics=[EmbeddingsDriftMetric('all_dimensions', drift_method=method)])
            temp_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
            result = temp_report.as_dict().get("metrics", [])[0].get("result", {})
            score = result.get("drift_score", "N/A")
            detected = result.get("drift_detected", "N/A")
            drift_summary.append(f"- {name}: score = {score:.4f}, drift = {detected}")
        except Exception as e:
            drift_summary.append(f"- {name}: failed ({e})")

    summary_text = "\n".join(drift_summary)
    st.session_state['drift_score_summary'] = summary_text

    # Streamlit 내 HTML 시각화
    components.html(st.session_state['drift_report_html'], height=800, scrolling=True, width=1600)
    st.success("✅ Drift report & all scores saved in session_state.")
    
    # 드리프트 메타데이터를 벡터DB에 업데이트
    with st.spinner("💾 Saving drift results to database..."):
        result = update_metadata_to_vectordb(dataset_name)
        if result:
            st.info("🔍 Drift analysis results have been automatically saved to the vector database.")
        else:
            st.warning("⚠️ Failed to save drift results to database.")