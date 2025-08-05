import os
import re
import streamlit as st
# pdf loader + split
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs embedding and vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# retriever, LLM 연결, QA chain 구성
from langchain_community.llms.ollama import Ollama
# Milvus 메타데이터 가져오기
from .make_html import search_metadata

@st.cache_resource

# ========== ollama base LLM 설정 - llm ==========
def custom_llm(model_name: str = None,
                model_temperature: float = None,
                model_max_tokens: int = None,
                model_top_p: float = None,
                ) -> str:

    pdf_path = "pdf_db/datadrift_tech_docs.pdf"
    
    if not os.path.exists(pdf_path):
        st.warning(f"드리프트 기술 문서가 없습니다: {pdf_path}")
        return None, None
    
    # 1. PDF 로드 및 텍스트 분할
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
        
    # 2. 텍스트 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
    chunks = text_splitter.split_documents(documents)

    # 3. HuggingFace 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    # 4. FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. 세션 상태에서 LLM 설정 가져오기
    if not model_name:
        model_name = st.session_state.get('selected_model', 'huihui_ai/exaone3.5-abliterated:7.8b')
    if model_temperature is None:
        model_temperature = st.session_state.get('model_temperature', 0.7)
    if model_max_tokens is None:
        model_max_tokens = st.session_state.get('max_tokens', 1000)
    if model_top_p is None:
        model_top_p = st.session_state.get('top_p', 0.9)
    
    llm = Ollama(
        model=model_name,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
        top_p=model_top_p,
    )

    return vectorstore, llm


def generate_explanation(dataset_name: str) -> str:
    """Ollama LLM을 활용한 드리프트 결과 해설 생성"""
    
    # 1. Milvus 메타데이터에서 데이터 가져오기
    metadata = search_metadata(dataset_name)
    if not metadata:
        return "<p>메타데이터를 찾을 수 없습니다.</p>"
    
    # 2. 드리프트 요약 정보 가져오기
    drift_summary = metadata.get('drift_score_summary', '')
    if not drift_summary:
        return "<p>드리프트 스코어 정보를 찾을 수 없습니다.</p>"
    
    # 3. 임베딩 정보 가져오기
    embedding_info = metadata.get('embedding_size', '')
    if not embedding_info:
        embedding_info = "임베딩 정보 없음"
    
    # 4. 검색 쿼리 구성
    search_queries = [
        f"MMD Wasserstein KL divergence interpretation",
        f"data drift analysis embedding",
        "drift detection threshold explanation",
    ]
    
    # 5. 관련 지식 검색
    knowledge_context = []
    vectorstore, llm = custom_llm()
    
    if not all([vectorstore, llm]):
        return "<p>LLM 초기화 실패</p>"
    
    for query in search_queries:
        relevant_docs = vectorstore.similarity_search(query, k=3)
        relevant_chunks = [doc.page_content for doc in relevant_docs]
        knowledge_context.extend(relevant_chunks)
    
    # 중복 제거 및 길이 제한
    knowledge_context = list(set(knowledge_context))[:3]  # 상위 3개만 사용
    
    # 6. 구조화된 정보 구성
    context_text = "\n\n".join(knowledge_context)
    
    # 7. 개선된 프롬프트 템플릿 (세션에서 가져오기)
    custom_prompt = st.session_state.get('custom_prompt', 
        """다음 데이터셋의 드리프트 분석 결과를 전문가 관점에서 해석해주세요:

데이터셋: {dataset_name}
임베딩 정보: {embedding_info}
드리프트 스코어: {drift_summary}

참고 자료:
{context}

위 정보를 바탕으로 다음 관점에서 분석해주세요:
1. 드리프트 발생 여부와 정도
2. 각 메트릭의 의미와 해석
3. 실무적 권장사항
4. 주의사항

HTML 형식으로 답변하되, <h3>, <p>, <ul>, <li> 태그를 사용해주세요.""")
    
    # 8. Ollama LLM 호출
    try:
        # 프롬프트 생성
        formatted_prompt = custom_prompt.format(
            dataset_name=dataset_name,
            embedding_info=embedding_info,
            drift_summary=drift_summary,
            context=context_text
        )
        
        # LLM 호출
        explanation = llm.invoke(formatted_prompt)
        
        return explanation
        
    except Exception as e:
        st.warning(f"AI 해설 생성 중 오류: {e}")
        return f"<p>AI 해설 생성 중 오류가 발생했습니다: {e}</p>"


def llm_html(dataset_name: str) -> str:
    """LLM 해설을 HTML 형태로 반환"""
    
    explanation = generate_explanation(dataset_name)
    
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
                p {{ margin-bottom: 15px; line-height: 1.6; }}
                ul {{ margin: 10px 0; padding-left: 20px; }}
                li {{ margin-bottom: 8px; }}
            </style>
        </head>
        <body>
            <h1>{dataset_name} AI 분석 리포트</h1>
            {explanation}
        </body>
    </html>
    """
    
    return html_template 