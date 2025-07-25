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
sys.path.append(os.path.join(current_dir, 'app_report'))
sys.path.append(os.path.join(current_dir, 'app_report/pages'))

# 페이지 구성 정의 (사이드바 네비게이션용)
PAGE_CONFIG = {
    "home": {
        "title": "🏠 Home"
    },
    "database": {
        "title": "📊 Database Pipeline"
    },
    "drift": {
        "title": "🔍 Drift Analysis"
    },
    "export": {
        "title": "📄 Export Report"
    }
}

def render_sidebar():
    """사이드바 네비게이션 렌더링"""
    with st.sidebar:
        st.title("📋 Navigation")
        st.markdown("---")
        
        # 메인 페이지 선택
        main_pages = list(PAGE_CONFIG.keys())
        selected_main = st.selectbox(
            "Select Category",
            main_pages,
            index=0,
            format_func=lambda x: PAGE_CONFIG[x]["title"]
        )
            
        st.markdown("---")
        
        # 진행 상황 표시
        st.markdown("### 🚀 Progress Tracker")
        
        # 세션 상태 확인
        database_complete = st.session_state.get('database_processed', False)
        drift_complete = st.session_state.get('drift_analysis_complete', False)
        report_complete = st.session_state.get('report_generated', False)
        
        # 진행 상황 시각화
        progress_items = [
            ("Database Setup", database_complete),
            ("Drift Analysis", drift_complete),
            ("Report Export", report_complete)
        ]
        
        for item, complete in progress_items:
            if complete:
                st.success(f"✅ {item}")
            else:
                st.info(f"⏳ {item}")
                
        return selected_main

def render_selected_page(main_page):
    """선택된 페이지 렌더링"""
    try:
        if main_page == "home":
            render_home_page()
        elif main_page == "database":
            render_database_page()
        elif main_page == "drift":
            render_drift_page()
        elif main_page == "export":
            render_export_page()
    except ImportError as e:
        st.error(f"모듈을 불러올 수 없습니다: {e}")
        st.info("해당 기능은 아직 구현되지 않았습니다.")
    except Exception as e:
        st.error(f"페이지 렌더링 중 오류가 발생했습니다: {e}")

def render_database_page():
    """데이터베이스 파이프라인 페이지"""
    st.markdown("## 📊 Database Pipeline")
    st.markdown("데이터 업로드부터 벡터 데이터베이스 저장까지의 전체 과정")
    
    # 단계별 진행
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Data Upload")
        try:
            from app_database.pages.upload_data import render
            render()
        except ImportError:
            st.info("Upload 모듈을 불러올 수 없습니다.")
    
    with col2:
        st.markdown("### ⚙️ Data Processing")
        try:
            from app_database.pages.data_load import render
            render()
        except ImportError:
            st.info("Processing 모듈을 불러올 수 없습니다.")

def render_drift_page():
    """드리프트 분석 페이지"""
    st.markdown("## 🔍 Drift Analysis")
    st.markdown("임베딩 로드부터 드리프트 탐지 및 AI 인사이트까지")
    
    # 탭으로 구성
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Analysis", "🤖 LLM Insights"])
    
    with tab1:
        try:
            from app_drift.pages.embedding_load import render
            render()
        except ImportError:
            st.info("Configuration 모듈을 불러올 수 없습니다.")
    
    with tab2:
        try:
            from app_drift.pages.detect_datadrift import render
            render()
        except ImportError:
            st.info("Analysis 모듈을 불러올 수 없습니다.")
    
    with tab3:
        try:
            from app_report.pages.build_llm import render
            render()
        except ImportError:
            st.info("LLM 모듈을 불러올 수 없습니다.")

def render_export_page():
    """리포트 내보내기 페이지"""
    st.markdown("## 📄 Export Report")
    st.markdown("분석 결과 조회 및 통합 보고서 생성")
    
    # 두 개 섹션으로 구성
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 View Reports")
        try:
            from app_report.pages.load_results import render
            render()
        except ImportError:
            st.info("Reports 모듈을 불러올 수 없습니다.")
    
    with col2:
        st.markdown("### 🔄 Generate Report")
        try:
            from app_report.pages.generate_report import render
            render()
        except ImportError:
            st.info("Generate 모듈을 불러올 수 없습니다.")

def render_home_page():
    """홈 페이지 렌더링"""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; margin-bottom: 30px;">
            <h3 style="color: white; text-align: center; margin: 0;">Welcome to Data Drift Analysis System</h3>
            <div style="color: white; text-align: center; margin-top: 10px;">
                📊 <b>Database Pipeline</b> → 🔍 <b>Drift Analysis</b> → 📄 <b>Export Report</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Database Pipeline
        - **Upload**: 데이터셋 업로드 및 검증
        - **Processing**: 데이터 전처리 및 준비
        """)
        
    with col2:
        st.markdown("""
        ### 🔍 Drift Analysis
        - **Config**: 드리프트 분석 설정
        - **Analysis**: 드리프트 탐지 실행
        - **LLM**: AI 기반 인사이트 생성
        """)
        
    with col3:
        st.markdown("""
        ### 📄 Export Report
        - **Reports**: 생성된 보고서 조회
        - **Generate**: 통합 보고서 생성
        """)
    
    st.markdown("---")
    
    # 시작하기 가이드
    st.markdown("### 🚀 시작하기")
    st.info("""
    1. **왼쪽 사이드바**에서 원하는 기능을 선택하세요
    2. **Database Pipeline**부터 시작하여 순차적으로 진행하는 것을 권장합니다
    3. **Progress Tracker**에서 현재 진행 상황을 확인할 수 있습니다
    """)
    
    # 최근 활동 표시
    if st.session_state.get('recent_activity'):
        st.markdown("### 📈 Recent Activity")
        for activity in st.session_state.recent_activity[-3:]:  # 최근 3개 활동
            st.success(activity)

# set_page_config 복원
st.set_page_config = original_set_page_config

# 사이드바 네비게이션 렌더링
main_page = render_sidebar()

# 메인 헤더
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

# 선택된 페이지 렌더링
render_selected_page(main_page)

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
