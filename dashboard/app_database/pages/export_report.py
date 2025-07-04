import pandas as pd
import pdfkit
import streamlit as st
import os
import json
from pymilvus import utility, Collection, connections

# Import utils from parent directory
try:
    from ..utils import gen_summarization
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import gen_summarization

@st.cache_data(ttl=1800)
def load_metadata_from_milvus(collection_name):
    """Milvus DB에서 메타데이터 로드"""
    try:
        connections.connect("default", host="localhost", port="19530")
        
        if not utility.has_collection(collection_name):
            return None
            
        collection = Collection(name=collection_name)
        collection.load()
        
        results = collection.query(
            expr="set_type == 'metadata'",
            output_fields=["dataset_name", "summary_dict", "data_previews", "class_dist_path", 
                          "doc_len_path", "doc_len_table", "wordcloud_path"],
            limit=1
        )
        
        if not results:
            return None
            
        metadata = results[0]
        
        # JSON 필드 파싱 - 더 강력한 파싱
        if metadata.get('summary_dict'):
            try:
                if isinstance(metadata['summary_dict'], str):
                    metadata['summary_dict'] = json.loads(metadata['summary_dict'])
                # 이미 dict인 경우 그대로 사용
                if not isinstance(metadata['summary_dict'], dict):
                    metadata['summary_dict'] = {}
            except:
                metadata['summary_dict'] = {}
        else:
            metadata['summary_dict'] = {}
            
        # data_previews 파싱 추가
        if metadata.get('data_previews'):
            try:
                if isinstance(metadata['data_previews'], str):
                    metadata['data_previews'] = json.loads(metadata['data_previews'])
                if not isinstance(metadata['data_previews'], dict):
                    metadata['data_previews'] = {}
            except:
                metadata['data_previews'] = {}
        else:
            metadata['data_previews'] = {}
            
        if metadata.get('doc_len_table'):
            try:
                if isinstance(metadata['doc_len_table'], str):
                    metadata['doc_len_table'] = json.loads(metadata['doc_len_table'])
                # 이미 list나 dict인 경우 그대로 사용
            except:
                pass
        
        return metadata
        
    except Exception as e:
        st.error(f"메타데이터 로드 실패: {e}")
        return None

def get_metadata_from_milvus(dataset_name=None):
    """Milvus에서 메타데이터를 찾아서 반환"""
    if not dataset_name:
        dataset_name = 'Dataset'
    
    try:
        connections.connect("default", host="localhost", port="19530")
        collections = utility.list_collections()
        
        # dataset_name과 일치하는 컬렉션 찾기
        for collection_name in collections:
            metadata = load_metadata_from_milvus(collection_name)
            if metadata and (metadata.get('dataset_name') == dataset_name or collection_name == dataset_name):
                return metadata
        
        # 찾지 못했다면 첫 번째 컬렉션 사용
        if collections:
            return load_metadata_from_milvus(collections[0])
            
    except Exception as e:
        st.error(f"Milvus 연결 실패: {e}")
        return None
    
    return None

def generate_html_from_session(dataset_name=None):
    """Milvus에서 직접 HTML 생성 (session_state 사용 안 함)"""
    metadata = get_metadata_from_milvus(dataset_name)
    
    if not metadata:
        return f"<html><body><h1>No metadata found for {dataset_name or 'Dataset'}</h1></body></html>"
    
    html_parts = []
    html_parts.append(f"<h1>{metadata.get('dataset_name', dataset_name or 'Dataset')} Dataset Report</h1>")

    if metadata.get("data_previews"):
        data_previews = metadata["data_previews"]
        # 문자열인 경우 JSON 파싱
        if isinstance(data_previews, str):
            try:
                data_previews = json.loads(data_previews)
            except:
                data_previews = {}
        
        for set_name, set_info in data_previews.items():
                html_parts.append(f"<h2>{set_name.capitalize()} Dataset</h2>")
                
                # 데이터셋 통계 정보
                total_rows = set_info.get("total_rows", 0)
                columns = set_info.get("columns", [])
                html_parts.append(f"<p><strong>Total Rows:</strong> {total_rows}</p>")
                html_parts.append(f"<p><strong>Columns:</strong> {len(columns)} ({', '.join(columns)})</p>")
                
                # 미리보기 테이블
                html_parts.append("<h3>Preview</h3>")
                preview_data = set_info.get("data", [])
                
                if preview_data:
                    html_parts.append('<table border="1" style="border-collapse: collapse; width: 100%;">')
                    
                    # 헤더
                    if preview_data:
                        headers = list(preview_data[0].keys())
                        html_parts.append("<thead><tr>")
                        html_parts.append("<th>Index</th>")  # 인덱스 컬럼
                        for header in headers:
                            html_parts.append(f"<th>{header}</th>")
                        html_parts.append("</tr></thead>")
                        
                        # 데이터 행들 (최대 10개)
                        html_parts.append("<tbody>")
                        for i, row in enumerate(preview_data):
                            html_parts.append("<tr>")
                            html_parts.append(f"<td>{i+1}</td>")  # 인덱스
                            for header in headers:
                                value = str(row.get(header, ""))
                                # 텍스트가 너무 길면 자르기 (100자로 제한)
                                if len(value) > 100:
                                    value = value[:97] + "..."
                                html_parts.append(f"<td>{value}</td>")
                            html_parts.append("</tr>")
                        html_parts.append("</tbody>")
                    
                    html_parts.append("</table>")
                else:
                    html_parts.append("<p><em>미리보기 데이터가 없습니다.</em></p>")
                
                html_parts.append("<br>")  # 섹션 구분

    # Milvus에서 직접 summary_dict 사용
    summary_dict = metadata.get('summary_dict', {})
    
    # summary_dict가 dict인지 확인하고 처리
    if not isinstance(summary_dict, dict):
        summary_dict = {}
    
    for name, summary in summary_dict.items():
        html_parts.append(f"<h2>{name} Dataset</h2>")
        
        # Preview 처리
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

        # Description 처리
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

        # Info 처리
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

    # 이미지 파일들 처리 - Milvus 데이터 직접 사용
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
            <meta charset="utf-8">
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

def render():
    # dataset_name을 session_state나 기본값에서 가져오기
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    
    # Milvus에서 메타데이터 확인
    metadata = get_metadata_from_milvus(dataset_name)
    if not metadata:
        st.error(f"No metadata found in Milvus for dataset: {dataset_name}")
        return

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(root_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    html_out = generate_html_from_session(dataset_name)
    html_path = os.path.join(reports_dir, f"{dataset_name}_report.html")
    pdf_path = os.path.join(reports_dir, f"{dataset_name}_report.pdf")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    try:
        config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")  # 필요 시 수정
        options = {
            'enable-local-file-access': None
        }
        pdfkit.from_file(html_path, pdf_path, configuration=config, options=options)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Dataset Report (PDF)",
                data=f,
                file_name=f"{dataset_name}_report.pdf",
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")