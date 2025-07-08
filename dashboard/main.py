import streamlit as st
import sys
import os

# 🔥 페이지 설정
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

# 모듈 로드 함수
@st.cache_resource
def load_modules():
    """모든 모듈을 한 번만 로드하고 캐시"""
    modules = {}
    
    # Database 모듈들
    db_modules = [
        ('upload_data', 'app_database.pages.upload_data'),
        ('data_load', 'app_database.pages.data_load'),
        ('base_visualization', 'app_database.pages.base_visualization'),
        ('vector_database', 'app_database.pages.vector_database'),
    ]
    
    for module_key, module_path in db_modules:
        try:
            modules[module_key] = __import__(module_path, fromlist=[''])
        except Exception as e:
            st.warning(f"⚠️ {module_key} 로드 실패: {e}")
            modules[module_key] = None
    
    # Drift 모듈들
    drift_modules = [
        ('embedding_load', 'app_drift.pages.embedding_load'),
        ('embedding_visualization', 'app_drift.pages.embedding_visualization'),
        ('detect_datadrift', 'app_drift.pages.detect_datadrift'),
    ]
    
    for module_key, module_path in drift_modules:
        try:
            modules[module_key] = __import__(module_path, fromlist=[''])
        except Exception as e:
            st.warning(f"⚠️ {module_key} 로드 실패: {e}")
            modules[module_key] = None
    
    # Integrated report
    try:
        modules['report_view'] = __import__('report_view', fromlist=[''])
    except Exception as e:
        st.warning(f"⚠️ report_view 로드 실패: {e}")
        modules['report_view'] = None
    
    return modules

# 모듈 로드
modules = load_modules()

# 페이지 렌더링 함수
def render_page(module, page_name):
    """안전한 페이지 렌더링"""
    try:
        if module and hasattr(module, 'render'):
            module.render()
        else:
            st.error(f"{page_name} 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"{page_name} 페이지 오류: {e}")

# 탭 구성 정의
TAB_CONFIG = {
    "database": {
        "title": "📊 Database Pipeline",
        "caption": "텍스트 데이터를 업로드하여 벡터 데이터베이스(Milvus)에 저장하고 분석합니다.",
        "progress": ["**1️⃣ Upload**", "**2️⃣ Load**", "**3️⃣ Visualize**", "**4️⃣ Store**"],
        "pages": [
            {"title": "1️⃣ Upload Data", "module_key": "upload_data", "name": "Upload Data"},
            {"title": "2️⃣ Load Data", "module_key": "data_load", "name": "Load Data"},
            {"title": "3️⃣ Visualization", "module_key": "base_visualization", "name": "Visualization"},
            {"title": "4️⃣ Vector Database", "module_key": "vector_database", "name": "Vector Database"}
        ]
    },
    "drift": {
        "title": "🔍 Drift Analysis & Export",
        "caption": "벡터 데이터베이스에서 임베딩을 불러와 드리프트를 감지하고 통합 리포트를 생성합니다",
        "progress": ["**1️⃣ Load**", "**2️⃣ Visualize**", "**3️⃣ Detect**", "**4️⃣ Report**"],
        "pages": [
            {"title": "1️⃣ Load Embeddings", "module_key": "embedding_load", "name": "Load Embeddings"},
            {"title": "2️⃣ Embeddings Visualization", "module_key": "embedding_visualization", "name": "Embeddings Visualization"},
            {"title": "3️⃣ Detect Drift", "module_key": "detect_datadrift", "name": "Detect Drift"},
            {"title": "4️⃣ 📋 통합 리포트 생성", "module_key": "report_view", "name": "Integrated Report", "special": True}
        ]
    }
}

# DB/메타DB에서 dataset_name을 가져옴
def load_dataset_name_from_db():
    return "MyDataset"

# DB에서 임베딩을 읽어옴
def load_train_embeddings_from_db():
    return None
def load_test_embeddings_from_db():
    return None

# DB에서 드리프트 요약을 읽어옴
def load_drift_score_summary_from_db():
    return None

# DB에서 드리프트 리포트 HTML을 읽어옴
def load_drift_report_html_from_db():
    return None

def ensure_session_state():
    """session_state에 필요한 값이 없으면 DB에서 불러와 저장"""
    if 'dataset_name' not in st.session_state or not st.session_state['dataset_name']:
        st.session_state['dataset_name'] = load_dataset_name_from_db()
    if 'train_embeddings' not in st.session_state or st.session_state['train_embeddings'] is None:
        st.session_state['train_embeddings'] = load_train_embeddings_from_db()
    if 'test_embeddings' not in st.session_state or st.session_state['test_embeddings'] is None:
        st.session_state['test_embeddings'] = load_test_embeddings_from_db()
    if 'drift_score_summary' not in st.session_state or st.session_state['drift_score_summary'] is None:
        st.session_state['drift_score_summary'] = load_drift_score_summary_from_db()
    if 'train_test_drift_report_html' not in st.session_state or st.session_state['train_test_drift_report_html'] is None:
        st.session_state['train_test_drift_report_html'] = load_drift_report_html_from_db()

# 통합 리포트 렌더링 함수
def render_report_view():
    """통합 리포트 특별 렌더링"""
    st.markdown("""
    <div style="background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; margin-bottom: 20px;">
        <strong>💡 Complete Analysis Report</strong><br>
        데이터베이스 정보와 드리프트 분석 결과를 통합한 전체 리포트를 생성합니다.
    </div>
    """, unsafe_allow_html=True)
    ensure_session_state()  # 🚨 세션 값 보장

    try:
        if modules.get('report_view') and hasattr(modules['report_view'], 'render_combined_report'):
            modules['report_view'].render_combined_report(
                modules.get('database_export_report'), 
                modules.get('drift_export_report')
            )
        else:
            st.error("통합 리포트 모듈을 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"통합 리포트 생성 중 오류 발생: {e}")

# 탭 렌더링 함수
def render_tab_content(tab_key):
    """효율적인 탭 콘텐츠 렌더링"""
    config = TAB_CONFIG[tab_key]
    
    # 헤더 및 설명
    st.header(config["title"])
    st.caption(config["caption"])
    
    # 진행 상태 표시
    progress_cols = st.columns(len(config["progress"]))
    for i, progress_text in enumerate(config["progress"]):
        with progress_cols[i]:
            st.markdown(progress_text)
    
    # 페이지들 렌더링
    for page in config["pages"]:
        st.markdown("---")
        st.subheader(page["title"])
        
        if page.get("special"):
            render_report_view()
        else:
            render_page(modules.get(page["module_key"]), page["name"])

# set_page_config 복원
st.set_page_config = original_set_page_config

st.markdown(
    '''
    <div style="width:100%; display:flex; justify-content:center; align-items:center; margin-bottom:10px; margin-top:10px;">
        <a href="https://www.keti.re.kr" target="_blank">
        <img src="https://raw.githubusercontent.com/keti-datadrift/datadrift_dataclinic/c91304849912308f4e95c83ba57f93c3a6989a49/dashboard/static/KETI_logo_dark-background.svg" 
            alt="KETI Logo" height="50" >
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)
st.title("🔄 통합 데이터 드리프트 분석 시스템")
st.caption("해당 연구는 '분석 모델의 성능저하 극복을 위한 데이터 드리프트 관리 기술 개발'로 2025년 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행되었습니다.")
st.markdown("---")
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h3 style="color: white; text-align: center; margin: 0;"> Select a Task to Start Your Analysis </h3>
        <div style="color: white; text-align: center; margin-top: 10px;">
            📊 <b>Database Pipeline</b> → 🔍 <b>Drift Analysis</b> → 📋 <b>Integrated Report</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

selected_tab = st.selectbox(
    "아래에서 원하는 작업을 선택하면, 해당 파이프라인 UI가 자동으로 바뀝니다.", 
    ["📊 Database Pipeline", "🔍 Drift Analysis & Export"],
    index=0
)
st.markdown("---")

# 선택된 탭에 따라 콘텐츠 렌더링
if selected_tab == "📊 Database Pipeline":
    render_tab_content("database")
elif selected_tab == "🔍 Drift Analysis & Export":
    render_tab_content("drift")

st.markdown("---")
st.markdown("""
            <div class="footer" style="text-align:center; margin-top:30px; color:#888;">
                <strong>
                    <a href="https://github.com/keti-datadrift/datadrift_dataclinic" 
                            target="_blank" style="color: #3498db; text-decoration: none;">
                        DataDrift Dataclinic System
                    </a>
                </strong><br>
                @KETI Korea Electronics Technology Institute, 2025<br>
            </div>
            """,  unsafe_allow_html=True
            )
