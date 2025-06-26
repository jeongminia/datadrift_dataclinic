import streamlit as st
import sys
import os

# 🔥 페이지 설정을 맨 처음에 한 번만!
st.set_page_config(
    page_title="통합 데이터 드리프트 분석 시스템",
    page_icon="🔄",
    layout="wide"
)

# 개별 페이지들에서 st.set_page_config() 호출을 방지
def mock_set_page_config(*args, **kwargs):
    pass

# 기존 set_page_config를 임시로 무력화
original_set_page_config = st.set_page_config
st.set_page_config = mock_set_page_config

# 경로 추가
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'app_database'))
sys.path.append(os.path.join(current_dir, 'app_database/pages'))
sys.path.append(os.path.join(current_dir, 'app_drift'))
sys.path.append(os.path.join(current_dir, 'app_drift/pages'))

# 기존 페이지들 import - 각각 안전하게 처리
upload_data = None
data_load = None
base_visualization = None
vector_database = None
db_export_report = None
embedding_load = None
embedding_visualization = None
detect_datadrift = None
drift_export_report = None

# app_database 페이지들 개별 import
try:
    from app_database.pages import upload_data
    #st.success("✅ upload_data 로드 성공")
except Exception as e:
    st.warning(f"⚠️ upload_data 로드 실패: {e}")

try:
    from app_database.pages import data_load
    #st.success("✅ data_load 로드 성공")
except Exception as e:
    st.warning(f"⚠️ data_load 로드 실패: {e}")

try:
    from app_database.pages import base_visualization
    #st.success("✅ base_visualization 로드 성공")
except Exception as e:
    st.warning(f"⚠️ base_visualization 로드 실패: {e}")

try:
    from app_database.pages import vector_database
    #st.success("✅ vector_database 로드 성공")
except Exception as e:
    st.warning(f"⚠️ vector_database 로드 실패: {e}")

# app_drift 페이지들 개별 import
try:
    from app_drift.pages import embedding_load
    #st.success("✅ embedding_load 로드 성공")
except Exception as e:
    st.warning(f"⚠️ embedding_load 로드 실패: {e}")

try:
    from app_drift.pages import embedding_visualization
    #st.success("✅ embedding_visualization 로드 성공")
except Exception as e:
    st.warning(f"⚠️ embedding_visualization 로드 실패: {e}")

try:
    from app_drift.pages import detect_datadrift
    #st.success("✅ detect_datadrift 로드 성공")
except Exception as e:
    st.warning(f"⚠️ detect_datadrift 로드 실패: {e}")

try:
    from app_drift.pages import export_report as drift_export_report
    #st.success("✅ drift_export_report 로드 성공")
except Exception as e:
    st.warning(f"⚠️ drift_export_report 로드 실패: {e}")

# set_page_config 복원
st.set_page_config = original_set_page_config

st.title("🔄 통합 데이터 드리프트 분석 시스템")

# 두 개의 메인 탭만 생성
tab1, tab2 = st.tabs(["📊 Database Pipeline", "🔍 Drift Analysis & Export"])

with tab1:
    st.header("📊 Database Pipeline")
    st.caption("텍스트 데이터 업로드하여 벡터DB(Milvus) 에 저장")
    
    # 모든 Database 페이지들을 순서대로 표시
    st.markdown("---")
    st.subheader("1️⃣ Upload Data")
    try:
        if upload_data:
            upload_data.render()
        else:
            st.error("Upload Data 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Upload Data 페이지 오류: {e}")
    
    st.markdown("---")
    st.subheader("2️⃣ Load Data")
    try:
        if data_load:
            data_load.render()
        else:
            st.error("Load Data 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Load Data 페이지 오류: {e}")
    
    st.markdown("---")
    st.subheader("3️⃣ Visualization")
    try:
        if base_visualization:
            base_visualization.render()
        else:
            st.error("Visualization 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Visualization 페이지 오류: {e}")
    
    st.markdown("---")
    st.subheader("4️⃣ Vector Database")
    try:
        if vector_database:
            vector_database.render()
        else:
            st.error("Vector Database 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Vector Database 페이지 오류: {e}")

with tab2:
    st.header("🔍 Drift Analysis & Export")
    st.caption("벡터DB에서 축적 불러와 시각화해 드리프트 감지 및 리포트 생성")
    
    # 모든 Drift 페이지들을 순서대로 표시
    st.markdown("---")
    st.subheader("1️⃣ Load Embeddings")
    try:
        if embedding_load:
            embedding_load.render()
        else:
            st.error("Load Embeddings 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Load Embeddings 페이지 오류: {e}")
    
    st.markdown("---")
    st.subheader("2️⃣ Embeddings Visualization")
    try:
        if embedding_visualization:
            embedding_visualization.render()
        else:
            st.error("Embeddings Visualization 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Embeddings Visualization 페이지 오류: {e}")
    
    st.markdown("---")
    st.subheader("3️⃣ Detect Drift")
    try:
        if detect_datadrift:
            detect_datadrift.render()
        else:
            st.error("Detect Drift 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Detect Drift 페이지 오류: {e}")
    
    st.markdown("---")
    st.subheader("4️⃣ Export Report")
    try:
        if drift_export_report:
            drift_export_report.render()
        else:
            st.error("Export Report 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"Export Report 페이지 오류: {e}")
    