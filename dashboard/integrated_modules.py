import streamlit as st
import sys
import os

# 🔥 페이지 설정을 맨 처음에 한 번만!
st.set_page_config(
    page_title="통합 데이터 드리프트 분석 시스템",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
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

# 기존 페이지들 import
upload_data = None
data_load = None
base_visualization = None
vector_database = None
database_export_report = None
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

try:
    from app_database.pages import export_report as database_export_report
    #st.success("✅ database_export_report 로드 성공")
except Exception as e:
    st.warning(f"⚠️ database_export_report 로드 실패: {e}")

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

# integrated_report 모듈 import
try:
    from integrated_report import render_combined_report
    #st.success("✅ integrated_report 로드 성공")
except Exception as e:
    st.warning(f"⚠️ integrated_report 로드 실패: {e}")
    # 폴백 함수 정의
    def render_combined_report(database_export_report=None, drift_export_report=None):
        st.error("통합 리포트 모듈을 불러올 수 없습니다.")

# set_page_config 복원
st.set_page_config = original_set_page_config

st.title("🔄 통합 데이터 드리프트 분석 시스템")
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
           padding: 20px; border-radius: 10px; margin-bottom: 30px;">
    <h3 style="color: white; margin: 0; text-align: center;">
        📊 Database Pipeline → 🔍 Drift Analysis → 📋 Integrated Report
    </h3>
</div>
""", unsafe_allow_html=True)

# 두 개의 메인 탭만 생성
tab1, tab2 = st.tabs(["📊 Database Pipeline", "🔍 Drift Analysis & Export"])

with tab1:
    st.header("📊 Database Pipeline")
    st.caption("텍스트 데이터를 업로드하여 벡터 데이터베이스(Milvus)에 저장하고 분석합니다")
    
    # 진행 상태 표시
    progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
    with progress_col1:
        st.markdown("**1️⃣ Upload**")
    with progress_col2:
        st.markdown("**2️⃣ Load**")
    with progress_col3:
        st.markdown("**3️⃣ Visualize**")
    with progress_col4:
        st.markdown("**4️⃣ Store**")
    
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
    st.caption("벡터 데이터베이스에서 임베딩을 불러와 드리프트를 감지하고 통합 리포트를 생성합니다")
    
    # 진행 상태 표시
    progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
    with progress_col1:
        st.markdown("**1️⃣ Load**")
    with progress_col2:
        st.markdown("**2️⃣ Visualize**")
    with progress_col3:
        st.markdown("**3️⃣ Detect**")
    with progress_col4:
        st.markdown("**4️⃣ Report**")
    
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
    st.subheader("4️⃣ 📋 통합 리포트 생성")
    st.markdown("""
    <div style="background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; margin-bottom: 20px;">
        <strong>💡 Complete Analysis Report</strong><br>
        데이터베이스 정보와 드리프트 분석 결과를 통합한 전체 리포트를 생성합니다.
    </div>
    """, unsafe_allow_html=True)
    try:
        render_combined_report(database_export_report, drift_export_report)
    except Exception as e:
        st.error(f"통합 리포트 생성 중 오류 발생: {e}")
