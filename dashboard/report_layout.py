import streamlit as st
import json
import os
import pandas as pd
from pymilvus import connections, Collection, utility

# utils.py에서 함수들 import
from utils import (
    get_cached_html, get_dataset_name, generate_html_template,
    generate_drift_content, check_drift_analysis_complete
)

# Milvus 관련 함수
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

# 기존 generate_combined_html 대체: 간소화된 메인 함수
def generate_combined_html(*args, **kwargs):
    """통합 HTML 리포트 생성 - 메인 함수 (대폭 간소화)"""
    dataset_name = get_dataset_name(*args, **kwargs)
    
    # 1. 데이터베이스 정보 가져오기
    db_cache_key = f"db_html_{dataset_name}"
    database_content = get_cached_html(db_cache_key, generate_db_html_from_milvus, dataset_name)
    if not database_content or 'No metadata found' in database_content:
        database_content = '<p>데이터베이스 정보를 불러올 수 없습니다.</p>'
    
    # 2. 드리프트 분석 결과 생성
    drift_content = generate_drift_content(dataset_name)
    
    # 3. 전체 HTML 생성
    return generate_html_template(dataset_name, database_content, drift_content)