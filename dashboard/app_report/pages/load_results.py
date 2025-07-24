import streamlit as st
import os
import glob

def get_available_datasets():
    """사용 가능한 데이터셋 목록"""
    datasets = set()
    reports_path = "reports"
    
    if os.path.exists(reports_path):
        # visualization.html 파일에서 데이터셋 이름 추출
        viz_files = glob.glob(f"{reports_path}/*visualization.html")
        for file in viz_files:
            name = os.path.basename(file).replace("_visualization.html", "")
            if name and name != "None":  # None 값 제외
                datasets.add(name)
        
        # drift_report.html 파일에서 데이터셋 이름 추출
        drift_files = glob.glob(f"{reports_path}/*drift_report.html")
        for file in drift_files:
            name = os.path.basename(file).replace("_train_test_drift_report.html", "")
            if name and name != "None":  # None 값 제외
                datasets.add(name)
        
        # 하나라도 있으면 반환
        return sorted(list(datasets))    
    return []

def check_database_results(dataset_name):
    """특정 데이터셋의 Database Pipeline 결과 확인"""
    reports_path = "reports"
    if os.path.exists(reports_path):
        viz_file = f"{reports_path}/{dataset_name}_visualization.html"
        return os.path.exists(viz_file)
    return False

def check_drift_results(dataset_name):
    """특정 데이터셋의 Drift Analysis 결과 확인"""
    reports_path = "reports"
    if os.path.exists(reports_path):
        drift_file = f"{reports_path}/{dataset_name}_train_test_drift_report.html"
        return os.path.exists(drift_file)
    return False

def load_html_result(result_type, dataset_name):
    """HTML 결과 파일 로드"""
    reports_path = "reports"
    
    if result_type == "visualization":
        file_pattern = f"{reports_path}/{dataset_name}_visualization.html"
    elif result_type == "drift_report":
        file_pattern = f"{reports_path}/{dataset_name}_train_test_drift_report.html"
    else:
        return None
    
    files = glob.glob(file_pattern)
    if files:
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None
    return None

def render():
    """결과 데이터 로드 페이지"""
    #st.write("사용자가 선택한 데이터셋을 기반하여 레포트를 반환합니다.")
    
    # 먼저 데이터셋 선택
    datasets = get_available_datasets()
    
    if not datasets:
        st.error("분석 결과를 찾을 수 없습니다.")
        return
    
    selected_dataset = st.selectbox(
        "📂 반환받고 싶은 리포트에 대한 데이터셋을 선택해주세요:",
        datasets
    )
    
    # 선택된 데이터셋에 대한 상태 표시
    if selected_dataset:
        col1, col2 = st.columns(2)
        
        with col1:
            has_database = check_database_results(selected_dataset)
            status = "✅ 준비됨" if has_database else "⏳ 대기중"
            st.write(f"**📊 Database:** {status}")
            
        with col2:
            has_drift = check_drift_results(selected_dataset)
            status = "✅ 완료됨" if has_drift else "⏳ 대기중"
            st.write(f"**🔍 Drift Analysis:** {status}")
        
        
        # 둘 다 완료되지 않았으면 경고
        if not has_database or not has_drift:
            st.warning(f"⚠️ **{selected_dataset}** 데이터셋에 대한 Database Pipeline과 Drift Analysis를 먼저 실행해주세요.")
            return
        
        # HTML 결과 로드
        db_html = load_html_result("visualization", selected_dataset)
        drift_html = load_html_result("drift_report", selected_dataset)
        
        # 세션에 저장
        st.session_state.selected_dataset = selected_dataset
        st.session_state.database_html = db_html
        st.session_state.drift_html = drift_html
        
        st.success(f"✅ **{selected_dataset}** 데이터셋 결과를 로드했습니다.")
