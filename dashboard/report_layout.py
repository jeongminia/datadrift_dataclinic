
from bs4 import BeautifulSoup
import streamlit as st
from datetime import datetime
import json
import os
from pymilvus import connections, Collection, utility
import pandas as pd

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
def get_cached_html(cache_key, generator_func, *args, **kwargs):
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        html = generator_func(*args, **kwargs)
        body = get_html_body(html)
        st.session_state[cache_key] = body
        return body
    except Exception as e:
        return f"<div>오류: {e}</div>"

# 드리프트 분석 완료 여부 확인
def check_drift_analysis_complete():
    required_keys = [
        'train_embeddings', 'test_embeddings', 
        'drift_score_summary', 'train_test_drift_report_html'
    ]
    return all(key in st.session_state for key in required_keys)


# Milvus에서 메타데이터 조회 및 HTML 생성
def load_metadata_from_milvus(collection_name):
    connections.connect("default", host="localhost", port="19530")
    if not utility.has_collection(collection_name):
        return None
    collection = Collection(name=collection_name)
    collection.load()
    results = collection.query(
        expr="set_type == 'metadata'",
        output_fields=["dataset_name", "summary_dict", "data_previews", "class_dist_path", "doc_len_path", "doc_len_table", "wordcloud_path"],
        limit=1
    )
    if not results:
        return None
    metadata = results[0]
    # JSON 파싱
    if metadata.get('summary_dict'):
        if isinstance(metadata['summary_dict'], str):
            try:
                metadata['summary_dict'] = json.loads(metadata['summary_dict'])
            except Exception:
                metadata['summary_dict'] = {}
        if not isinstance(metadata['summary_dict'], dict):
            metadata['summary_dict'] = {}
    else:
        metadata['summary_dict'] = {}
    if metadata.get('data_previews'):
        if isinstance(metadata['data_previews'], str):
            try:
                metadata['data_previews'] = json.loads(metadata['data_previews'])
            except Exception:
                metadata['data_previews'] = {}
        if not isinstance(metadata['data_previews'], dict):
            metadata['data_previews'] = {}
    else:
        metadata['data_previews'] = {}
    if metadata.get('doc_len_table'):
        if isinstance(metadata['doc_len_table'], str):
            try:
                metadata['doc_len_table'] = json.loads(metadata['doc_len_table'])
            except Exception:
                pass
    return metadata

def get_metadata_from_milvus(dataset_name=None):
    if not dataset_name:
        dataset_name = 'Dataset'
    connections.connect("default", host="localhost", port="19530")
    collections = utility.list_collections()
    for collection_name in collections:
        metadata = load_metadata_from_milvus(collection_name)
        if metadata and (metadata.get('dataset_name') == dataset_name or collection_name == dataset_name):
            return metadata
    if collections:
        return load_metadata_from_milvus(collections[0])
    return None

def generate_db_html_from_milvus(dataset_name=None):
    metadata = get_metadata_from_milvus(dataset_name)
    if not metadata:
        return f"<html><body><h1>No metadata found for {dataset_name or 'Dataset'}</h1></body></html>"
    html_parts = []
    html_parts.append(f"<h1>{metadata.get('dataset_name', dataset_name or 'Dataset')} Dataset Report</h1>")
    if metadata.get("data_previews"):
        data_previews = metadata["data_previews"]
        if isinstance(data_previews, str):
            try:
                data_previews = json.loads(data_previews)
            except:
                data_previews = {}
        for set_name, set_info in data_previews.items():
            html_parts.append(f"<h2>{set_name.capitalize()} Dataset</h2>")
            total_rows = set_info.get("total_rows", 0)
            columns = set_info.get("columns", [])
            html_parts.append(f"<p><strong>Total Rows:</strong> {total_rows}</p>")
            html_parts.append(f"<p><strong>Columns:</strong> {len(columns)} ({', '.join(columns)})</p>")
            html_parts.append("<h3>Preview</h3>")
            preview_data = set_info.get("data", [])
            if preview_data:
                html_parts.append('<table border="1" style="border-collapse: collapse; width: 100%;">')
                if preview_data:
                    headers = list(preview_data[0].keys())
                    html_parts.append("<thead><tr>")
                    html_parts.append("<th>Index</th>")
                    for header in headers:
                        html_parts.append(f"<th>{header}</th>")
                    html_parts.append("</tr></thead>")
                    html_parts.append("<tbody>")
                    for i, row in enumerate(preview_data):
                        html_parts.append("<tr>")
                        html_parts.append(f"<td>{i+1}</td>")
                        for header in headers:
                            value = str(row.get(header, ""))
                            if len(value) > 100:
                                value = value[:97] + "..."
                            html_parts.append(f"<td>{value}</td>")
                        html_parts.append("</tr>")
                    html_parts.append("</tbody>")
                html_parts.append("</table>")
            else:
                html_parts.append("<p><em>미리보기 데이터가 없습니다.</em></p>")
            html_parts.append("<br>")
    summary_dict = metadata.get('summary_dict', {})
    if not isinstance(summary_dict, dict):
        summary_dict = {}
    for name, summary in summary_dict.items():
        html_parts.append(f"<h2>{name} Dataset</h2>")
        if 'preview' in summary:
            html_parts.append("<h3>Preview</h3>")
            try:
                if hasattr(summary["preview"], 'to_html'):
                    html_parts.append(summary["preview"].to_html(index=False))
                elif isinstance(summary["preview"], list):
                    df = pd.DataFrame(summary["preview"])
                    html_parts.append(df.to_html(index=False))
                else:
                    html_parts.append(f"<p>{summary['preview']}</p>")
            except:
                html_parts.append(f"<p>{summary['preview']}</p>")
        if 'description' in summary:
            html_parts.append("<h3>Description</h3>")
            try:
                if hasattr(summary["description"], 'to_html'):
                    html_parts.append(summary["description"].to_html(index=False))
                elif isinstance(summary["description"], list):
                    df = pd.DataFrame(summary["description"])
                    html_parts.append(df.to_html(index=False))
                else:
                    html_parts.append(f"<p>{summary['description']}</p>")
            except:
                html_parts.append(f"<p>{summary['description']}</p>")
        if 'info' in summary:
            html_parts.append("<h3>Info</h3>")
            try:
                if hasattr(summary["info"], 'to_html'):
                    html_parts.append(summary["info"].to_html(index=False))
                elif isinstance(summary["info"], list):
                    df = pd.DataFrame(summary["info"])
                    html_parts.append(df.to_html(index=False))
                else:
                    html_parts.append(f"<p>{summary['info']}</p>")
            except:
                html_parts.append(f"<p>{summary['info']}</p>")
    html_parts.append("<hr><h2>Visualizations</h2>")
    if metadata.get("class_dist_path") and os.path.exists(metadata["class_dist_path"]):
        abs_path = os.path.abspath(metadata["class_dist_path"])
        html_parts.append(f"<h3>Class Distribution</h3><img src='file://{abs_path}' width='800'><br><br>")
    if metadata.get("doc_len_path") and os.path.exists(metadata["doc_len_path"]):
        abs_path = os.path.abspath(metadata["doc_len_path"])
        html_parts.append(f"<h3>Document Length Distribution</h3><img src='file://{abs_path}' width='800'><br><br>")
    if metadata.get("doc_len_table"):
        doc_len_table = metadata["doc_len_table"]
        if isinstance(doc_len_table, str):
            html_parts.append(doc_len_table)
        elif isinstance(doc_len_table, list):
            try:
                df = pd.DataFrame(doc_len_table)
                html_parts.append("<h3>Document Length Statistics</h3>")
                html_parts.append(df.to_html(index=False))
            except:
                html_parts.append(f"<p>{doc_len_table}</p>")
    if metadata.get("wordcloud_path") and os.path.exists(metadata["wordcloud_path"]):
        abs_path = os.path.abspath(metadata["wordcloud_path"])
        html_parts.append(f"<h3>Word Cloud</h3><img src='file://{abs_path}' width='900'><br><br>")
    html_template = f"""
    <html>
        <head>
            <meta charset=\"utf-8\">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 30px;
                }}
                h1 {{ font-size: 28px; }}
                h2 {{ font-size: 22px; margin-top: 40px; }}
                h3 {{ font-size: 18px; margin-top: 25px; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                    font-size: 13px;
                }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 6px;
                    text-align: left;
                }}
                th {{ background-color: #f5f5f5; }}
                img {{
                    margin-top: 10px;
                    margin-bottom: 20px;
                }}
                .comment-box {{
                    background-color: #f4f4f4;
                    padding: 15px;
                    margin: 10px 0 30px 0;
                    border-radius: 8px;
                }}
                ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 6px;
                }}
            </style>
        </head>
        <body>
            {''.join(html_parts)}
        </body>
    </html>
    """
    return html_template

# 기존 generate_combined_html 대체: Milvus에서 직접 데이터 조회 및 HTML 생성
def generate_combined_html(*args, **kwargs):
    # dataset_name을 args, kwargs, session_state에서 모두 받아올 수 있도록
    dataset_name = None
    if args and len(args) > 0 and args[0]:
        dataset_name = args[0]
    elif 'dataset_name' in kwargs and kwargs['dataset_name']:
        dataset_name = kwargs['dataset_name']
    elif st.session_state.get('dataset_name'):
        dataset_name = st.session_state['dataset_name']
    else:
        dataset_name = 'Dataset'

    timestamp = datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
    db_cache_key = f"db_html_{dataset_name}"
    # DB HTML은 항상 생성 (없으면 안내문구)
    database_content = get_cached_html(db_cache_key, generate_db_html_from_milvus, dataset_name)
    if not database_content or 'No metadata found' in database_content:
        database_content = '<p>데이터베이스 정보를 불러올 수 없습니다.</p>'


    # 드리프트 결과 요약(텍스트)도 함께 추출
    drift_html = st.session_state.get('train_test_drift_report_html', None)
    drift_summary = st.session_state.get('drift_score_summary', None)
    drift_content = ''
    if drift_summary:
        drift_content += f'<div class="comment-box"><b>Drift Score Summary</b><br><pre style="font-size:1em;">{drift_summary}</pre></div>'

    # 임베딩 시각화 이미지가 있으면 PDF에 삽입
    drift_img_keys = [
        ('embedding_distance_img', 'Embedding Distance (Original Dimension)'),
        ('embedding_pca_distance_img', 'Embedding Distance after PCA'),
        ('embedding_pca_img', 'Embedding Visualization after PCA'),
    ]
    for key, title in drift_img_keys:
        img_buf = st.session_state.get(key)
        if img_buf:
            # 이미지를 임시 파일로 저장 (PDFKit에서 file://로 접근 가능해야 함)
            img_path = f"temp/{key}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(img_buf.getbuffer())
            drift_content += f'<h3>{title}</h3><img src="file://{os.path.abspath(img_path)}" width="600"><br>'

    if drift_html:
        drift_content += get_html_body(drift_html)
    if not drift_content:
        drift_content = '<p>드리프트 분석 결과가 없습니다.</p>'

    combined_html = f"""<!DOCTYPE html>
                        <html lang=\"ko\">
                        <head>
                            <meta charset=\"utf-8\">
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
                            <div class=\"container\">
                                <div class=\"header\">
                                    <div class=\"title\">{dataset_name} 통합 분석 리포트</div>
                                    <div class=\"subtitle\">데이터 드리프트 분석 보고서</div>
                                    <div style=\"margin-top: 10px; font-size: 0.9em;\">생성일시: {timestamp}</div>
                                </div>
                                
                                <div class=\"section\">
                                    <div class=\"section-title\">📊 Dataset Information & Statistics</div>
                                    {database_content}
                                </div>
                                
                                <div class=\"section\">
                                    <div class=\"section-title\">🔍 Data Drift Analysis Results</div>
                                    {drift_content}
                                </div>
                                
                                <div class=\"footer\">
                                    <strong>
                                        <a href=\"https://github.com/keti-datadrift/datadrift_dataclinic\" target=\"_blank\" style=\"color: #3498db; text-decoration: none;\">
                                        DataDrift Dataclinic System
                                        </a>
                                    </strong><br>
                                    @2025 KETI, Korea Electronics Technology Institute<br>
                                </div>
                            </div>
                        </body>
                        </html>"""
    return combined_html