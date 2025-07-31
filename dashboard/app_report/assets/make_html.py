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
                            "class_dist_path", "doc_len_path", "doc_len_table", "wordcloud_path"],
                limit=1
                )
    metadata = results[0]
    
    # JSON 문자열을 딕셔너리로 파싱
    if metadata.get('summary_dict'):
        if isinstance(metadata['summary_dict'], str):
            metadata['summary_dict'] = json.loads(metadata['summary_dict'])
        if not isinstance(metadata['summary_dict'], dict):
            metadata['summary_dict'] = {}
    else:
        metadata['summary_dict'] = {}
    
    # 데이터 미리보기 정보 파싱
    if metadata.get('data_previews'):
        if isinstance(metadata['data_previews'], str):
            metadata['data_previews'] = json.loads(metadata['data_previews'])
        if not isinstance(metadata['data_previews'], dict):
            metadata['data_previews'] = {}
    else:
        metadata['data_previews'] = {}
    
    # 문서 길이 테이블 정보 파싱
    if metadata.get('doc_len_table'):
        if isinstance(metadata['doc_len_table'], str):
            metadata['doc_len_table'] = json.loads(metadata['doc_len_table'])
    
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
    
    # 데이터 미리보기 섹션 생성
    if metadata.get("data_previews"):
        data_previews = metadata["data_previews"]
        if isinstance(data_previews, str):
            data_previews = json.loads(data_previews)
        
        # 각 데이터셋(train, test, validation 등)에 대해 처리
        for set_name, set_info in data_previews.items():
            html_parts.append(f"<h2>{set_name.capitalize()} Dataset</h2>")
            
            total_rows = set_info.get("total_rows", 0)
            columns = set_info.get("columns", [])
            html_parts.append(f"<p><strong>Total Rows:</strong> {total_rows}</p>")
            html_parts.append(f"<p><strong>Columns:</strong> {len(columns)} ({', '.join(columns)})</p>")
            
            # 데이터 미리보기 테이블 생성
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
                            # 너무 긴 텍스트는 잘라서 표시
                            if len(value) > 100:
                                value = value[:97] + "..."
                            html_parts.append(f"<td>{value}</td>")
                        html_parts.append("</tr>")
                    html_parts.append("</tbody>")
                html_parts.append("</table>")
            else:
                html_parts.append("<p><em>미리보기 데이터가 없습니다.</em></p>")
            
            html_parts.append("<br>")
    
    # 데이터셋 요약 정보 섹션 생성
    summary_dict = metadata.get('summary_dict', {})
    if not isinstance(summary_dict, dict):
        summary_dict = {}
    
    for name, summary in summary_dict.items():
        html_parts.append(f"<h2>{name} Dataset</h2>")
        
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
    html_parts = []

    dataset_name = st.session_state.get("dataset_name", "Dataset")
    html_parts.append(f"<h1>{dataset_name} Drift Report</h1>")

    for key in ["train_embeddings", "valid_embeddings", "test_embeddings"]:
        if key in st.session_state:
            shape = st.session_state[key].shape
            html_parts.append(f"<p><b>{key.replace('_', ' ').title()}:</b> {shape}</p>")

    if 'embedding_distance_img' in st.session_state:
        img_bytes = st.session_state['embedding_distance_img'].getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        html_parts.append("<hr><h2>Embedding Distance (Original Dimension)</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')

    if 'pca_selected_dim' in st.session_state:
        html_parts.append(f"<p><b>PCA Reduced Dimension:</b> {st.session_state['pca_selected_dim']}</p>")

    if 'embedding_pca_distance_img' in st.session_state:
        img_bytes = st.session_state['embedding_pca_distance_img'].getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        html_parts.append("<hr><h2>Embedding Distance after PCA</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')

    if 'embedding_pca_img' in st.session_state:
        img_bytes = st.session_state['embedding_pca_img'].getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        html_parts.append("<hr><h2>Embedding Visualization after PCA</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')

    if 'drift_score_summary' in st.session_state:
        html_parts.append("<hr><h2>Quantitative Drift Scores</h2>")
        
        if f"{dataset_name}_drift_report.html" in st.session_state:
            drift_report_html = st.session_state[f"{dataset_name}_drift_report.html"]
            if isinstance(drift_report_html, str):
                html_parts.append(drift_report_html)
            else:
                html_parts.append("<p>Drift report not available.</p>")
        else:
            html_parts.append("<p>Drift report not available.</p>")
        
