import streamlit as st
import sys
import os
import importlib.util
import io
import contextlib

# 페이지 설정
st.set_page_config(
    page_title="통합 데이터 드리프트 분석 시스템",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def mock_set_page_config(*args, **kwargs):
    pass
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

# ------------------------------------- Milvus -------------------------------------
def load_milvus_inspect_function():
    """Milvus inspect 함수 로드"""
    spec = importlib.util.spec_from_file_location(
        "inspect_collections", 
        os.path.join(current_dir, "milvus_db", "inspect-collections.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.milvus_inpect

def load_milvus_remove_function():
    """Milvus remove 함수 로드"""
    spec = importlib.util.spec_from_file_location(
        "remove_collections", 
        os.path.join(current_dir, "milvus_db", "rm-collections.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.milvus_rm

def capture_function_output(func, *args, **kwargs):
    """함수 출력을 캡처하여 문자열로 반환"""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func(*args, **kwargs)
    return f.getvalue()

# ------------------------------------- Side Bar Navigation -------------------------------------
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
    with st.sidebar:
        st.title("📋 Navigation")
        st.markdown("---")
        st.markdown("### Select Pages")
        # 메인 페이지 선택
        main_pages = list(PAGE_CONFIG.keys())
        selected_main = st.selectbox(
            " ",
            main_pages,
            index=0,
            format_func=lambda x: PAGE_CONFIG[x]["title"]
        )
            
        st.markdown("---")
        st.markdown("### Milvus Tracker", 
                    help="Milvus 데이터베이스 상태를 확인하고, 데이터셋을 관리합니다.")
        
        col1, col2 = st.columns(2)

        with col1:
            inspect_clicked = st.button("Inspect Collections", key="inspect_collections")
        
        with col2:
            remove_clicked = st.button("Remove Collections", key="rm_collections")

        if inspect_clicked:
            inspect_function = load_milvus_inspect_function()
            output = capture_function_output(inspect_function)
            
            # 메인 영역에 결과 표시
            with st.expander("🔍 Milvus Collections", expanded=True):
                st.code(output or "컬렉션 정보가 없습니다.", language="text")

        if remove_clicked:
            dataset_name = st.text_input(
                "Dataset", 
                placeholder="특정 데이터셋명 입력 후 Enter", 
                key="dataset_input"
            )
            # 텍스트가 입력되고 Enter 누르면 바로 실행
            if dataset_name:
                remove_function = load_milvus_remove_function()
                output = capture_function_output(remove_function, target=dataset_name)
                
                with st.expander(f"🗑️ '{dataset_name}' 결과", expanded=True):
                    st.code(output, language="text")
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
        elif main_page == "report":
            render_report_page()
    except ImportError as e:
        st.error(f"모듈을 불러올 수 없습니다: {e}")
        st.info("해당 기능은 아직 구현되지 않았습니다.")
    except Exception as e:
        st.error(f"페이지 렌더링 중 오류가 발생했습니다: {e}")

# ------------------------------------- page config -------------------------------------
PAGE_CONFIG = {
        "home": {
        "title": "🏠 Home"
        },

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
            "title": "🔍 Drift Analysis",
            "caption": "벡터 데이터베이스에서 임베딩을 불러와 드리프트를 감지합니다.",
            "progress": ["**1️⃣ Load**", "**2️⃣ Visualize**", "**3️⃣ Detect**"],
            "pages": [
                {"title": "1️⃣ Load Embeddings", "module_key": "embedding_load", "name": "Load Embeddings"},
                {"title": "2️⃣ Embeddings Visualization", "module_key": "embedding_visualization", "name": "Embeddings Visualization"},
                {"title": "3️⃣ Detect Drift", "module_key": "detect_datadrift", "name": "Detect Drift"}
            ]
        },

        "report": {
            "title": "📄 Export Report",
            "caption": "데이터 분석 결과와 드리프트 탐지 결과를 기반으로 Custom LLM을 통하여 통합 리포트를 생성합니다",
            "progress": ["**1️⃣ Load Results**", "**2️⃣ Build Custom LLM**", "**3️⃣ Generate Report**"],
            "pages": [
                {"title": "1️⃣ Load Results", "module_key": "load_results", "name": "Load Results"},
                {"title": "2️⃣ Build Custom LLM", "module_key": "build_llm", "name": "Custom LLM"},
                {"title": "3️⃣ 📋 Generate Report", "module_key": "generate_report", "name": "Integrated Report"}
            ]
        }
    }

def page_render(page_key):
    config = PAGE_CONFIG[page_key]

    st.header(config["title"])
    st.caption(config["caption"])

    for page in config["pages"]:
        st.markdown("---")
        st.subheader(page["title"])

# ------------------------------- database pipeline -------------------------------
def render_database_page():
    st.markdown("## 📊 Database Pipeline")
    st.markdown("데이터 업로드부터 벡터 데이터베이스 저장까지의 전체 과정")

    config = PAGE_CONFIG["database"]
    
    # 진행 상황 표시
    progress_cols = st.columns(len(config["progress"]))
    for i, step in enumerate(config["progress"]):
        with progress_cols[i]:
            st.markdown(step)
    
    st.markdown("---")
    
    # 각 페이지를 순차적으로 렌더링
    for page in config["pages"]:
        st.markdown(f"### {page['title']}")
        try:
            module_path = f"app_database.pages.{page['module_key']}"
            module = __import__(module_path, fromlist=['render'])
            module.render()
        except ImportError:
            st.info(f"{page['name']} 모듈을 불러올 수 없습니다.")
        except Exception as e:
            st.error(f"{page['name']} 렌더링 중 오류: {e}")
        
        st.markdown("---")

# ------------------------------- Drift Analysis -------------------------------
def render_drift_page():
    st.markdown("## 🔍 Drift Analysis")
    st.markdown("임베딩 로드부터 드리프트 탐지 분석까지 제안")
    
    config = PAGE_CONFIG["drift"]

    # 진행 상황 표시
    progress_cols = st.columns(len(config["progress"]))
    for i, step in enumerate(config["progress"]):
        with progress_cols[i]:
            st.markdown(step)
    
    st.markdown("---")
    
    # 각 페이지를 순차적으로 렌더링
    for page in config["pages"]:
        st.markdown(f"### {page['title']}")
        try:
            module_path = f"app_drift.pages.{page['module_key']}"
            module = __import__(module_path, fromlist=['render'])
            module.render()
        except ImportError:
            st.info(f"{page['name']} 모듈을 불러올 수 없습니다.")
        except Exception as e:
            st.error(f"{page['name']} 렌더링 중 오류: {e}")
        
        st.markdown("---")

# ------------------------------- Export Report -------------------------------
def render_report_page():
    st.markdown("## 📄 Export Report")
    st.markdown("분석 결과인 레포트 조회 및 통합 보고서 생성")
    
    config = PAGE_CONFIG["report"]

    # 진행 상황 표시
    progress_cols = st.columns(len(config["progress"]))
    for i, step in enumerate(config["progress"]):
        with progress_cols[i]:
            st.markdown(step)
    
    st.markdown("---")
    
    # 각 페이지를 순차적으로 렌더링
    for page in config["pages"]:
        st.markdown(f"### {page['title']}")
        try:
            module_path = f"app_report.pages.{page['module_key']}"
            module = __import__(module_path, fromlist=['render'])
            module.render()
        except ImportError as e:
            st.error(f"📦 ImportError: {e}")
        except Exception as e:
            st.error(f"🔥 보고서 렌더링 중 오류 발생: {e}")
        
        st.markdown("---")

# ------------------------------------- main -------------------------------------
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
        - **Upload**: 데이터셋 업로드
        - **Load**: 데이터 기본 정보
        - **Visualization**: 워드 클라우드 및 클래스 분포 시각화
        - **Vector Database**: 벡터DB에 임베딩 데이터 저장
        """)
        
    with col2:
        st.markdown("""
        ### 🔍 Drift Analysis
        - **Load**: 임베딩 데이터 로드
        - **Visualization**: 임베딩 데이터 시각화
        - **Detect Drift**: 데이터 드리프트 탐지
        """)
        
    with col3:
        st.markdown("""
        ### 📄 Export Report
        - **Load**: 생성된 보고서 조회
        - **Build LLM**: 사용자 맞춤형 LLM 생성
        - **Generate**: 통합 보고서 생성
        """)
    
    st.markdown("---")
    
    # 시작하기 가이드
    st.markdown("### 🚀 시작하기")
    st.info("""
    1. 왼쪽 사이드바에서 원하는 기능을 선택하세요
    2. **Database Pipeline**부터 시작하여 순차적으로 진행하는 것을 권장합니다
    3. **Milvus Tracker**에서 현재 데이터베이스에 저장된 데이터를 확인할 수 있습니다
    """)
    st.markdown("---")

# ------------------------------------- fixed UI -------------------------------------

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
