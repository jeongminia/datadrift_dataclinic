import os
import json
import base64
import streamlit as st
import pandas as pd
from pymilvus import connections, Collection, utility

# ------------------------------------- Milvus 메타데이터 로드 -------------------------------------
def metadata_milvus(collection_name):
    connections.connect("default", host="localhost", port="19530")

    if not utility.has_collection(collection_name):
        return None
    
    collection = Collection(name=collection_name)
    collection.load()
    
    # 메타데이터 타입의 데이터만 쿼리
    results = collection.query(
                expr="set_type == 'metadata'",
                output_fields=["dataset_name", "summary_dict", "data_previews", 
                            "class_dist_path", "doc_len_path", "doc_len_table", "wordcloud_path",
                            # 추가 사항
                            "dimension", "embedding_size", "original_distance_path",
                            "PCA_distance_path", "PCA_visualization_path", "drift_score_summary"
                            ],
                limit=1
                )
    metadata = results[0]
    
    # JSON 문자열을 딕셔너리로 파싱
    def safe_json_parse(value, default={}):
        if not value:
            return default
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return default
        return default
        
    # summary_dict 파싱
    metadata['summary_dict'] = safe_json_parse(metadata.get('summary_dict'), {})
    
    # data_previews 파싱
    metadata['data_previews'] = safe_json_parse(metadata.get('data_previews'), {})
    
    # doc_len_table 파싱
    metadata['doc_len_table'] = safe_json_parse(metadata.get('doc_len_table'), None)
    
    # embedding_size 파싱
    metadata['embedding_size'] = metadata.get('embedding_size')
    
    # drift_score_summary 파싱
    metadata['drift_score_summary'] = metadata.get('drift_score_summary')
    
    return metadata

# ------------------------------------- Milvus에서 dataset이름으로 콜렉션 찾기 -------------------------------------
def search_metadata(dataset_name):
    if not dataset_name:
        dataset_name = 'Dataset'
    
    connections.connect("default", host="localhost", port="19530")
    collections = utility.list_collections()
    
    # 모든 컬렉션에서 해당 데이터셋 검색
    for collection_name in collections:
        metadata = metadata_milvus(collection_name)
        if metadata and (metadata.get('dataset_name') == dataset_name or collection_name == dataset_name):
            return metadata
    
    # 찾지 못하면 첫 번째 컬렉션의 메타데이터 반환
    if collections:
        return metadata_milvus(collections[0])

# ------------------------------------- Milvus에서 Database Pipeline HTML 리포트 생성 -------------------------------------
def database_html(dataset_name): 

    metadata = search_metadata(dataset_name)
    
    if not metadata:
        return f"<html><body><h1>No metadata found for {dataset_name or 'Dataset'}</h1></body></html>"
    
    html_parts = []
    
    # 리포트 제목 생성
    html_parts.append(f"<h1>{metadata.get('dataset_name', dataset_name or 'Dataset')} Dataset Report</h1>")
    
    # 데이터셋 정보 섹션 생성 (summary_dict에서만 추출)
    summary_dict = metadata.get('summary_dict', {})
    
    if not isinstance(summary_dict, dict):
        summary_dict = {}
    
    # 데이터셋 순서를 train, valid, test 순으로 정렬
    dataset_order = []
    for preferred_name in ['train', 'valid', 'test']:
        if preferred_name in summary_dict:
            dataset_order.append(preferred_name)
    
    # 나머지 데이터셋들을 알파벳 순으로 추가
    for dataset in sorted(summary_dict.keys()):
        if dataset not in ['train', 'valid', 'test']:
            dataset_order.append(dataset)
    
    for set_name in dataset_order:
        html_parts.append(f"<h2>{set_name.capitalize()} Dataset</h2>")
        
        # summary_dict에서 preview, description, info 정보 추출
        if set_name in summary_dict:
            summary = summary_dict[set_name]
            
            # 미리보기 정보가 있으면 표시
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
            
            # 데이터 설명 정보가 있으면 표시
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
            
            # 데이터 정보가 있으면 표시
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
        
        html_parts.append("<br>")
    
    # 시각화 섹션 생성
    html_parts.append("<hr><h2>Visualizations</h2>")
    
    # 클래스 분포 차트
    if metadata.get("class_dist_path") and os.path.exists(metadata["class_dist_path"]):
        abs_path = os.path.abspath(metadata["class_dist_path"])
        html_parts.append(f"<h3>Class Distribution</h3><img src='file://{abs_path}' width='800'><br><br>")
    
    # 문서 길이 분포 차트
    if metadata.get("doc_len_path") and os.path.exists(metadata["doc_len_path"]):
        abs_path = os.path.abspath(metadata["doc_len_path"])
        html_parts.append(f"<h3>Document Length Distribution</h3><img src='file://{abs_path}' width='800'><br><br>")
    
    # 문서 길이 통계 테이블
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
    
    # 워드클라우드 이미지
    if metadata.get("wordcloud_path") and os.path.exists(metadata["wordcloud_path"]):
        abs_path = os.path.abspath(metadata["wordcloud_path"])
        html_parts.append(f"<h3>Word Cloud</h3><img src='file://{abs_path}' width='900'><br><br>")
    
    # 최종 HTML 템플릿 생성 (CSS 스타일 포함)
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

# ------------------------------------- Milvus에서 DataDrift Pipeline HTML 리포트 생성 -------------------------------------
def drift_html(dataset_name):
    
    metadata = search_metadata(dataset_name)
    
    if not metadata:
        return f"<html><body><h1>No drift metadata found for {dataset_name or 'Dataset'}</h1></body></html>"
    
    html_parts = []
    
    # 리포트 제목 생성
    html_parts.append(f"<h1>{metadata.get('dataset_name', dataset_name or 'Dataset')} Data Drift Analysis Results</h1>")
    
    # 임베딩 정보 섹션
    if metadata.get('embedding_size'):
        html_parts.append(f"<p>{metadata.get('embedding_size')}</p>")
    
    # PCA 차원 정보
    if metadata.get('dimension'):
        html_parts.append(f"<p><b>PCA Reduced Dimension:</b> {metadata.get('dimension')}</p>")
    
    # 원본 차원 임베딩 거리 이미지
    if metadata.get("original_distance_path") and os.path.exists(metadata["original_distance_path"]):
        abs_path = os.path.abspath(metadata["original_distance_path"])
        html_parts.append("<h2>Embedding Distance (Original Dimension)</h2>")
        html_parts.append(f"<img src='file://{abs_path}' width='800'><br><br>")
    
    # PCA 후 임베딩 거리 이미지
    if metadata.get("PCA_distance_path") and os.path.exists(metadata["PCA_distance_path"]):
        abs_path = os.path.abspath(metadata["PCA_distance_path"])
        html_parts.append("<h2>Embedding Distance after PCA</h2>")
        html_parts.append(f"<img src='file://{abs_path}' width='800'><br><br>")
    
    # PCA 후 임베딩 시각화 이미지
    if metadata.get("PCA_visualization_path") and os.path.exists(metadata["PCA_visualization_path"]):
        abs_path = os.path.abspath(metadata["PCA_visualization_path"])
        html_parts.append("<h2>Embedding Visualization after PCA</h2>")
        html_parts.append(f"<img src='file://{abs_path}' width='800'><br><br>")
    
    # 드리프트 스코어 요약 섹션
    if metadata.get('drift_score_summary'):
        html_parts.append("<hr><h2>Drift Score Summary</h2>")
        html_parts.append(f"<pre>{metadata.get('drift_score_summary')}</pre>")
    
    # 최종 HTML 템플릿 생성 (CSS 스타일 포함)
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
        
