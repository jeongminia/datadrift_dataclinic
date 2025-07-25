import os
import re
import streamlit as st
# pdf loader + split
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs embedding and vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# retriever, LLM 연결, QA chain 구성
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

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
        model_name = st.session_state.get('custom_llm_model', 'huihui_ai/exaone3.5-abliterated:7.8b')
    if model_temperature is None:
        model_temperature = st.session_state.get('custom_llm_temperature', 0.7)
    if model_max_tokens is None:
        model_max_tokens = st.session_state.get('custom_llm_max_tokens', 1000)
    if model_top_p is None:
        model_top_p = st.session_state.get('custom_llm_top_p', 0.9)
    
    llm = OllamaLLM(
        model=model_name,
        temperature=model_temperature,
        max_tokens=model_max_tokens,
        top_p=model_top_p,
    )

    return vectorstore, llm


def generate_explanation(dataset_name: str) -> str:
    """Ollama LLM을 활용한 드리프트 결과 해설 생성"""
    
    # 1. 현재 드리프트 결과 수집
    drift_summary = st.session_state.get('drift_score_summary')
    if not drift_summary:
        return ''
    
    # 2. 강화된 임베딩 정보 수집
    embedding_info = {}
    for key in ['train_embeddings', 'valid_embeddings', 'test_embeddings']:
        embeddings = st.session_state.get(key)
        if embeddings is not None:
            embedding_info[key] = {
                'shape': embeddings.shape,
                'mean': float(embeddings.mean()) if hasattr(embeddings, 'mean') else 'N/A',
                'std': float(embeddings.std()) if hasattr(embeddings, 'std') else 'N/A'
            }
    embedding_info_text = ""
    for key, info in embedding_info.items():
        if isinstance(info, dict):
            embedding_info_text += f"{key}: shape={info['shape']}, mean={info['mean']}, std={info['std']}\n"
        else:
            embedding_info_text += f"{key}: {info}\n"

    # 4. 검색 쿼리 구성
    search_queries = [
        f"MMD Wasserstein KL divergence interpretation",
        f"data drift analysis embedding",
        "drift detection threshold explanation",
    ]
    
    # 5. 관련 지식 검색
    knowledge_context = []
    for query in search_queries:
        vectorstore, llm = custom_llm()
    
        if not all([vectorstore, llm]):
            return []
        
        relevant_docs = vectorstore.similarity_search(query, k=3)
        relevant_chunks = [doc.page_content for doc in relevant_docs]
        knowledge_context.extend(relevant_chunks)
    
    # 중복 제거 및 길이 제한
    knowledge_context = list(set(knowledge_context))[:3]  # 상위 3개만 사용
    
    # 6. 구조화된 정보 구성
    context_text = "\n\n".join(knowledge_context)
    
    # 7. 개선된 프롬프트 템플릿
    prompt_template = st.session_state.get('custom_prompt_template')
    
    # 8. Ollama LLM 호출
    try:
        llm = custom_llm()
        
        # 프롬프트 생성
        formatted_prompt = prompt_template.format(
            dataset_name=dataset_name,
            embedding_info=embedding_info_text,
            drift_summary=drift_summary,
            context=context_text
        )
        
        # LLM 호출
        explanation = llm.invoke(formatted_prompt)
        
        return explanation
        
    except Exception as e:
        st.warning(f"AI 해설 생성 중 오류: {e}")
        return 