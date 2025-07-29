from bs4 import BeautifulSoup
import streamlit as st
import base64
import os
from datetime import datetime

# ========== HTML 관련 유틸리티 ==========
def get_html_body(html):
    """HTML에서 <body> 태그만 추출하고 h1 태그 제거"""
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

def get_cached_html(cache_key, generator_func, *args, **kwargs):
    """캐시된 HTML 가져오기 또는 생성하기"""
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        html = generator_func(*args, **kwargs)
        body = get_html_body(html)
        st.session_state[cache_key] = body
        return body
    except Exception as e:
        return f"<div>오류: {e}</div>"

# ========== 데이터셋 이름 유틸리티 ==========
def get_dataset_name(*args, **kwargs):
    """다양한 소스에서 dataset_name 추출"""
    if args and len(args) > 0 and args[0]:
        return args[0]
    elif 'dataset_name' in kwargs and kwargs['dataset_name']:
        return kwargs['dataset_name']
    elif st.session_state.get('dataset_name'):
        return st.session_state['dataset_name']
    return 'Dataset'

# ========== 세션 상태 체크 유틸리티 ==========
def check_drift_analysis_complete():
    """드리프트 분석 완료 여부 확인"""
    required_keys = [
        'train_embeddings', 'test_embeddings', 
        'drift_score_summary', 'drift_report_html'
    ]
    return all(key in st.session_state for key in required_keys)

def get_embedding_info():
    """세션에서 임베딩 정보 추출"""
    info = {}
    for key in ['train_embeddings', 'valid_embeddings', 'test_embeddings']:
        embeddings = st.session_state.get(key)
        if embeddings is not None:
            info[key] = embeddings.shape
    
    pca_dim = st.session_state.get('pca_selected_dim')
    if pca_dim:
        info['pca_selected_dim'] = pca_dim
    
    return info

def generate_embedding_info_html(info):
    """임베딩 정보를 HTML로 변환"""
    html_parts = []
    for key, shape in info.items():
        if key == 'pca_selected_dim':
            html_parts.append(f'<p>PCA Reduced Dimension: {shape}</p>')
        else:
            display_name = key.replace('_', ' ').title()
            html_parts.append(f'<p>{display_name}: {shape}</p>')
    return ''.join(html_parts)

# ========== 이미지 처리 유틸리티 ==========
def save_temp_image(img_bytes, dataset_name, key, temp_dir="reports"):
    """임시 이미지 파일 저장"""
    temp_filename = f"{dataset_name}_{key}.png"
    temp_path = f"{temp_dir}/{temp_filename}"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(img_bytes)
    return temp_path

def process_session_image(key, title, dataset_name, width=800):
    """세션에 저장된 단일 이미지를 HTML로 변환"""
    if key not in st.session_state:
        return ''
    
    try:
        img_bytes = st.session_state[key].getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # 임시 파일 저장
        save_temp_image(img_bytes, dataset_name, key)
        
        return f'''<h3>{title}</h3>
<img src="data:image/png;base64,{img_base64}" width="{width}" style="margin: 10px 0;"/><br>'''
    except Exception as e:
        return f'<p>이미지 로드 오류 ({title}): {e}</p>'

def process_all_session_images(dataset_name, img_mappings=None):
    """세션에 저장된 모든 이미지를 처리해서 HTML 생성"""
    if img_mappings is None:
        img_mappings = [
            ('embedding_distance_img', 'Embedding Distance (Original Dimension)'),
            ('embedding_pca_distance_img', 'Embedding Distance after PCA'),
            ('embedding_pca_img', 'Embedding Visualization after PCA'),
        ]
    
    html_parts = []
    for key, title in img_mappings:
        img_html = process_session_image(key, title, dataset_name)
        html_parts.append(img_html)
    return ''.join(html_parts)

# ========== CSS 스타일 유틸리티 ==========
def get_report_css():
    """리포트용 CSS 스타일 반환"""
    return """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
        font-family: 'Malgun Gothic', sans-serif; 
        line-height: 1.6; color: #2c3e50; 
        background: #f8f9fa; padding: 30px;
    }
    .container { 
        max-width: 1000px; margin: 0 auto; 
        background: white; padding: 30px; 
        border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 25px; border-radius: 8px; 
        margin-bottom: 25px; text-align: center;
    }
    .title { font-size: 2em; margin-bottom: 5px; }
    .subtitle { font-size: 1.1em; opacity: 0.9; }
    .section { 
        margin: 25px 0; padding: 20px; 
        border: 1px solid #e9ecef; border-radius: 8px;
    }
    .section-title { 
        font-size: 1.4em; color: #495057; 
        margin-bottom: 15px; padding-bottom: 8px;
        border-bottom: 2px solid #dee2e6;
    }
    table { 
        width: 100%; border-collapse: collapse; margin: 15px 0;
        border-radius: 5px; overflow: hidden;
    }
    th { 
        background: #6c757d; color: white; 
        padding: 10px; text-align: left;
    }
    td { padding: 8px; border-bottom: 1px solid #dee2e6; }
    img { max-width: 100%; height: auto; margin: 10px 0; }
    pre { 
        background: #f8f9fa; padding: 15px; 
        border-radius: 5px; overflow-x: auto;
    }
    .footer { 
        text-align: center; margin-top: 30px; 
        padding: 15px; background: #f8f9fa; 
        border-radius: 5px; color: #6c757d;
    }
    .comment-box {
        background-color: #f4f4f4;
        padding: 15px;
        margin: 10px 0 30px 0;
        border-radius: 8px;
    }
    """

def generate_html_template(dataset_name, database_content, drift_content):
    """전체 HTML 템플릿 생성"""
    timestamp = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    css = get_report_css()
    
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>{dataset_name} - 통합 분석 리포트</title>
    <style>{css}</style>
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
            {database_content}
        </div>
        
        <div class="section">
            <div class="section-title">🔍 Data Drift Analysis Results</div>
            {drift_content}
        </div>
        
        <div class="footer">
            <strong>
                <a href="https://github.com/keti-datadrift/datadrift_dataclinic" target="_blank" style="color: #3498db; text-decoration: none;">
                DataDrift Dataclinic System
                </a>
            </strong><br>
            @2025 KETI, Korea Electronics Technology Institute<br>
        </div>
    </div>
</body>
</html>"""

# ========== 드리프트 콘텐츠 생성 유틸리티 ==========
def generate_drift_content(dataset_name):
    """드리프트 분석 결과 HTML 생성 (LLM 해설 포함)"""
    content_parts = []
    
    # 1. 임베딩 정보
    embedding_info = get_embedding_info()
    if embedding_info:
        content_parts.append(generate_embedding_info_html(embedding_info))
    
    # 2. 이미지들
    images_html = process_all_session_images(dataset_name)
    content_parts.append(images_html)
    
    # 3. Drift Score Summary
    drift_summary = st.session_state.get('drift_score_summary')
    if drift_summary:
        content_parts.append(
            f'<div class="comment-box"><b>Drift Score Summary</b><br>'
            f'<pre style="font-size:1em;">{drift_summary}</pre></div>'
        )
        
        # 4. 🆕 LLM 해설 추가
        llm_explanation = generate_llm_drift_explanation(dataset_name)
        if llm_explanation:
            content_parts.append(llm_explanation)
    
    return ''.join(content_parts) if content_parts else '<p>드리프트 분석 결과가 없습니다.</p>'