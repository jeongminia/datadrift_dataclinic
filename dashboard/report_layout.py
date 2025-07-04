from bs4 import BeautifulSoup
import streamlit as st
from datetime import datetime

# 드리프트 분석 완료 여부 확인
def check_drift_analysis_complete():
    required_keys = [
        'train_embeddings', 'test_embeddings', 
        'drift_score_summary', 'train_test_drift_report_html'
    ]
    return all(key in st.session_state for key in required_keys)

# HTML에서 <body> 태그만 추출하고 h1 태그 제거
def get_html_body(html):
    if not html:
        return ''
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find('body')
        if body:
            for h1 in body.find_all('h1'):
                h1.decompose()
            return str(body)
        else:
            return str(soup)
    else:
        import re
        return re.sub(r'<h1[^>]*>.*?</h1>', '', html, flags=re.DOTALL)

# 캐시된 HTML 가져오기 또는 생성하기
def get_cached_html(cache_key, generator_func, *args):
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        html = generator_func(*args)
        body = get_html_body(html)
        st.session_state[cache_key] = body
        return body
    except Exception as e:
        return f"<div>오류: {e}</div>"

# main HTML 생성 함수
def generate_combined_html(database_export_report=None, drift_export_report=None):
    """최적화된 HTML 생성 (캐시 활용)"""
    dataset_name = st.session_state.get('dataset_name')
    timestamp = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    db_cache_key = f"db_html_{dataset_name}"
    drift_cache_key = f"drift_html_{dataset_name}"
    database_content = ''
    drift_content = ''
    if database_export_report:
        database_content = get_cached_html(db_cache_key, database_export_report.generate_html_from_session, dataset_name)
    if drift_export_report and check_drift_analysis_complete():
        drift_content = get_cached_html(drift_cache_key, drift_export_report.generate_html_from_session)
    has_drift = check_drift_analysis_complete()
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
                                    <strong>
                                        <a href="https://github.com/keti-datadrift/datadrift_dataclinic" target="_blank" style="color: #3498db; text-decoration: none;">DataDrift Dataclinic System</a>
                                        DataDrift Dataclinic System
                                        </a>
                                    </strong><br>
                                    @2025 KETI, Korea Electronics Technology Institute<br>
                                </div>
                            </div>
                        </body>
                        </html>"""
    return combined_html