# 통합 리포트 생성 함수들
import streamlit as st
import base64
import pandas as pd
import os
from datetime import datetime
try:
    import pdfkit
except ImportError:
    pdfkit = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

def check_drift_analysis_complete():
    """드리프트 분석이 완료되었는지 확인"""
    required_keys = [
        'train_embeddings', 'test_embeddings', 
        'drift_score_summary', 'train_test_drift_report_html'
    ]
    return all(key in st.session_state for key in required_keys)

# Fragment를 사용한 독립적인 HTML 생성
@st.fragment
def generate_html_content_fragment(database_export_report, drift_export_report, dataset_name):
    """Fragment로 HTML 생성하여 rerun 최소화"""
    
    # 캐시 키 설정
    db_cache_key = f"db_html_{dataset_name}"
    drift_cache_key = f"drift_html_{dataset_name}"
    
    # Database HTML (캐시 우선 사용)
    database_content = ""
    if database_export_report:
        if db_cache_key in st.session_state:
            database_content = st.session_state[db_cache_key]
        else:
            try:
                with st.spinner("📊 Database 정보 생성 중..."):
                    db_html = database_export_report.generate_html_from_session(dataset_name)
                    if BeautifulSoup:
                        soup = BeautifulSoup(db_html, "html.parser")
                        body = soup.find('body')
                        if body:
                            for h1 in body.find_all('h1'):
                                h1.decompose()
                            database_content = str(body)
                        else:
                            database_content = db_html
                    else:
                        database_content = db_html
                    st.session_state[db_cache_key] = database_content
            except Exception as e:
                database_content = f"<div>Database 정보 생성 오류: {e}</div>"
    
    # Drift HTML (캐시 우선 사용)
    drift_content = ""
    if drift_export_report and check_drift_analysis_complete():
        if drift_cache_key in st.session_state:
            drift_content = st.session_state[drift_cache_key]
        else:
            try:
                with st.spinner("🔍 Drift 분석 결과 생성 중..."):
                    drift_html = drift_export_report.generate_html_from_session()
                    if BeautifulSoup:
                        soup = BeautifulSoup(drift_html, "html.parser")
                        body = soup.find('body')
                        if body:
                            for h1 in body.find_all('h1'):
                                h1.decompose()
                            drift_content = str(body)
                        else:
                            drift_content = drift_html
                    else:
                        drift_content = drift_html
                    st.session_state[drift_cache_key] = drift_content
            except Exception as e:
                drift_content = f"<div>Drift 분석 결과 생성 오류: {e}</div>"
    
    return database_content, drift_content

def generate_combined_html(database_export_report=None, drift_export_report=None):
    """최적화된 HTML 생성 (캐시 활용)"""
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    timestamp = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    
    # Fragment에서 생성된 콘텐츠 가져오기
    database_content, drift_content = generate_html_content_fragment(
        database_export_report, drift_export_report, dataset_name
    )
    
    has_drift = check_drift_analysis_complete()
    
    # 경량화된 HTML 템플릿
    combined_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>{dataset_name} - 통합 분석 리포트</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Malgun Gothic', sans-serif; 
            line-height: 1.6; color: #2c3e50; 
            background: #f8f9fa; padding: 30px;
        }}
        .container {{ 
            max-width: 1000px; margin: 0 auto; 
            background: white; padding: 30px; 
            border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 25px; border-radius: 8px; 
            margin-bottom: 25px; text-align: center;
        }}
        .title {{ font-size: 2em; margin-bottom: 5px; }}
        .subtitle {{ font-size: 1.1em; opacity: 0.9; }}
        .section {{ 
            margin: 25px 0; padding: 20px; 
            border: 1px solid #e9ecef; border-radius: 8px;
        }}
        .section-title {{ 
            font-size: 1.4em; color: #495057; 
            margin-bottom: 15px; padding-bottom: 8px;
            border-bottom: 2px solid #dee2e6;
        }}
        table {{ 
            width: 100%; border-collapse: collapse; margin: 15px 0;
            border-radius: 5px; overflow: hidden;
        }}
        th {{ 
            background: #6c757d; color: white; 
            padding: 10px; text-align: left;
        }}
        td {{ padding: 8px; border-bottom: 1px solid #dee2e6; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        pre {{ 
            background: #f8f9fa; padding: 15px; 
            border-radius: 5px; overflow-x: auto;
        }}
        .footer {{ 
            text-align: center; margin-top: 30px; 
            padding: 15px; background: #f8f9fa; 
            border-radius: 5px; color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">{dataset_name} 통합 분석 리포트</div>
            <div class="subtitle">데이터 드리프트 분석 보고서</div>
            <div style="margin-top: 10px; font-size: 0.9em;">생성일시: {timestamp}</div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Dataset Information & Statistics</div>
            {database_content if database_content else '<p>데이터베이스 정보를 불러올 수 없습니다.</p>'}
        </div>
        
        <div class="section">
            <div class="section-title">🔍 Data Drift Analysis Results</div>
            {drift_content if drift_content and has_drift else '<p>드리프트 분석이 완료되지 않았습니다.</p>'}
        </div>
        
        <div class="footer">
            <strong>KETI DataDrift Detection System</strong><br>
            자동 생성된 분석 리포트
        </div>
    </div>
</body>
</html>"""
    
    return combined_html
    
    # HTML body 부분만 추출하여 정리
    database_content = ""
    if database_html:
        if BeautifulSoup:
            soup = BeautifulSoup(database_html, "html.parser")
            # h1 태그 제거 (중복 제목 방지)
            for h1 in soup.find_all('h1'):
                h1.decompose()
            database_content = str(soup)
        else:
            # BeautifulSoup가 없는 경우 간단한 문자열 처리
            import re
            database_content = re.sub(r'<h1[^>]*>.*?</h1>', '', database_html, flags=re.DOTALL)
    
    drift_content = ""
    if drift_html:
        if BeautifulSoup:
            soup = BeautifulSoup(drift_html, "html.parser")
            # h1 태그 제거 (중복 제목 방지)
            for h1 in soup.find_all('h1'):
                h1.decompose()
            drift_content = str(soup)
        else:
            # BeautifulSoup가 없는 경우 간단한 문자열 처리
            import re
            drift_content = re.sub(r'<h1[^>]*>.*?</h1>', '', drift_html, flags=re.DOTALL)
    
    # 통합 HTML 생성
    combined_html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{dataset_name} - 통합 데이터 드리프트 분석 리포트</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #fff;
                padding: 40px;
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 50px;
                padding-bottom: 30px;
                border-bottom: 3px solid #3498db;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                color: #2c3e50;
                margin-bottom: 10px;
                font-weight: 700;
            }}
            
            .header .subtitle {{
                font-size: 1.2em;
                color: #7f8c8d;
                font-weight: 300;
            }}
            
            .section {{
                margin-bottom: 60px;
                background: #f8f9fa;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .section-title {{
                font-size: 1.8em;
                color: #2c3e50;
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 2px solid #3498db;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .section-content {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            
            h2 {{
                font-size: 1.5em;
                color: #34495e;
                margin: 30px 0 20px 0;
                padding-left: 15px;
                border-left: 4px solid #3498db;
            }}
            
            h3 {{
                font-size: 1.3em;
                color: #495057;
                margin: 25px 0 15px 0;
                font-weight: 600;
            }}
            
            h4 {{
                font-size: 1.1em;
                color: #6c757d;
                margin: 20px 0 10px 0;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }}
            
            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 0.5px;
            }}
            
            tr:hover {{
                background-color: #f8f9fa;
            }}
            
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                margin: 15px 0;
            }}
            
            .comment-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 25px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}
            
            .comment-box h3 {{
                color: white;
                margin-bottom: 15px;
            }}
            
            .drift-explanation {{
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                margin: 25px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}
            
            .drift-explanation h2 {{
                color: white;
                border-left: 4px solid white;
                margin-bottom: 20px;
            }}
            
            pre {{
                background: #2d3748;
                color: #e2e8f0;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
                margin: 20px 0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 0.9em;
                line-height: 1.4;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }}
            
            ul {{
                padding-left: 25px;
                margin: 15px 0;
            }}
            
            li {{
                margin-bottom: 8px;
                line-height: 1.5;
            }}
            
            p {{
                margin: 15px 0;
                line-height: 1.6;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 60px;
                padding-top: 30px;
                border-top: 2px solid #e9ecef;
                color: #6c757d;
                font-size: 0.9em;
            }}
            
            .status-indicator {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 600;
                margin-left: 10px;
            }}
            
            .status-complete {{
                background: #d4edda;
                color: #155724;
            }}
            
            .status-pending {{
                background: #fff3cd;
                color: #856404;
            }}
            
            @media print {{
                body {{
                    padding: 20px;
                }}
                .section {{
                    break-inside: avoid;
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔄 {dataset_name} 통합 분석 리포트</h1>
            <div class="subtitle">데이터 드리프트 분석 및 통계 보고서</div>
        </div>
        
        <div class="section">
            <div class="section-title">
                📊 Dataset Information & Statistics
                <span class="status-indicator status-complete">완료</span>
            </div>
            <div class="section-content">
                {database_content if database_content else '<p><em>데이터베이스 정보를 불러올 수 없습니다.</em></p>'}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">
                🔍 Data Drift Analysis Results
                <span class="status-indicator {'status-complete' if check_drift_analysis_complete() else 'status-pending'}">
                    {'완료' if check_drift_analysis_complete() else '대기중'}
                </span>
            </div>
            <div class="section-content">
                {drift_content if drift_content and check_drift_analysis_complete() else '<p><em>드리프트 분석이 아직 완료되지 않았습니다. "Detect Drift" 단계를 먼저 실행해주세요.</em></p>'}
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Generated by 통합 데이터 드리프트 분석 시스템</strong></p>
            <p>생성일시: {pd.Timestamp.now().strftime('%Y년 %m월 %d일 %H시 %M분')}</p>
        </div>
    </body>
    </html>
    """
    
    return combined_html

def render_combined_report(database_export_report=None, drift_export_report=None):
    """최적화된 통합 리포트 렌더링 - rerun 최소화"""
    
    # 상태 체크
    has_database = database_export_report is not None
    has_drift = check_drift_analysis_complete()
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    
    # 상태 표시 (컴팩트)
    col1, col2 = st.columns(2)
    with col1:
        status = "✅ 준비됨" if has_database else "⏳ 대기중"
        st.write(f"**Database:** {status}")
    with col2:
        status = "✅ 완료됨" if has_drift else "⏳ 대기중"
        st.write(f"**Drift Analysis:** {status}")
    
    # 리포트 생성 가능 여부
    can_generate = has_database or has_drift
    
    if not can_generate:
        st.info("💡 Database Pipeline 또는 Drift Analysis를 먼저 완료해주세요.")
        return
    
    # 버튼 상태를 session_state로 관리
    button_key = f"report_btn_{dataset_name}"
    generate_clicked = st.button(
        "� Complete Report 생성", 
        key=button_key,
        type="primary",
        help="통합 리포트를 생성합니다 (최대 30초 소요)"
    )
    
    # 생성 중 상태 관리
    if generate_clicked:
        st.session_state[f"generating_{dataset_name}"] = True
    
    # 리포트 생성 로직 (상태 기반)
    if st.session_state.get(f"generating_{dataset_name}", False):
        
        # 진행률 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. HTML 생성
            progress_bar.progress(25)
            status_text.text("🔄 HTML 콘텐츠 생성 중...")
            
            html_content = generate_combined_html(database_export_report, drift_export_report)
            
            # 2. 파일 경로 설정
            progress_bar.progress(50)
            status_text.text("📁 파일 경로 설정 중...")
            
            current_dir = os.path.dirname(__file__)
            reports_dir = os.path.join(current_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"{dataset_name}_report_{timestamp}"
            html_path = os.path.join(reports_dir, f"{filename}.html")
            pdf_path = os.path.join(reports_dir, f"{filename}.pdf")
            
            # 3. HTML 파일 저장
            progress_bar.progress(75)
            status_text.text("💾 HTML 파일 저장 중...")
            
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # 4. PDF 생성 시도
            progress_bar.progress(90)
            status_text.text("📋 PDF 생성 중...")
            
            pdf_success = False
            if pdfkit:
                try:
                    options = {
                        'page-size': 'A4',
                        'margin-top': '0.5in',
                        'margin-right': '0.5in',
                        'margin-bottom': '0.5in',
                        'margin-left': '0.5in',
                        'encoding': "UTF-8"
                    }
                    pdfkit.from_file(html_path, pdf_path, options=options)
                    pdf_success = True
                except Exception as e:
                    st.warning(f"PDF 생성 실패: {e}")
            
            # 5. 완료
            progress_bar.progress(100)
            status_text.text("✅ 생성 완료!")
            
            # 다운로드 버튼들
            col1, col2 = st.columns(2)
            
            with col1:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                st.download_button(
                    label="📄 HTML 다운로드",
                    data=html_data,
                    file_name=f"{filename}.html",
                    mime="text/html",
                    key=f"html_dl_{timestamp}"
                )
            
            with col2:
                if pdf_success and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="� PDF 다운로드",
                        data=pdf_data,
                        file_name=f"{filename}.pdf",
                        mime="application/pdf",
                        key=f"pdf_dl_{timestamp}"
                    )
                else:
                    st.error("PDF 생성에 실패했습니다.")
            
            st.success("🎉 통합 리포트가 성공적으로 생성되었습니다!")
            
            # 상태 초기화
            st.session_state[f"generating_{dataset_name}"] = False
            
        except Exception as e:
            st.error(f"리포트 생성 중 오류: {e}")
            st.session_state[f"generating_{dataset_name}"] = False
        
        finally:
            # UI 정리
            progress_bar.empty()
            status_text.empty()
