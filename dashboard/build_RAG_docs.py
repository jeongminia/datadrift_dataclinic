# rag_engine.py
import os
import streamlit as st
# pdf loader + split
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs embedding and vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# retriever
# LLM 연결, QA chain 구성
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ========== RAG 관련 유틸리티 ==========
@st.cache_resource
def load_drift_knowledge_base():
    """드리프트 기술 문서를 FAISS 벡터 DB로 로드 (LangChain 활용)"""
    pdf_path = "pdf_db/datadrift_tech_docs.pdf"
    
    if not os.path.exists(pdf_path):
        st.warning(f"드리프트 기술 문서가 없습니다: {pdf_path}")
        return None, None
    
    try:
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
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 4. FAISS 벡터스토어 생성
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # 5. Ollama LLM 설정
        llm = Ollama(
            model="hyperclovax-seed-text-instruct-1.5B",
            temperature=0.7
        )
        
        return vectorstore, llm
        
    except Exception as e:
        st.error(f"드리프트 지식 베이스 로딩 실패: {e}")
        return None, None



def search_drift_knowledge(query: str, top_k: int = 3):
    """드리프트 지식 베이스에서 관련 정보 검색 (LangChain 활용)"""
    vectorstore, llm = load_drift_knowledge_base()
    
    if not all([vectorstore, llm]):
        return []
    
    try:
        # 유사도 검색으로 관련 문서 찾기
        relevant_docs = vectorstore.similarity_search(query, k=top_k)
        
        # 문서 내용만 추출
        relevant_chunks = [doc.page_content for doc in relevant_docs]
        
        return relevant_chunks
        
    except Exception as e:
        st.error(f"지식 베이스 검색 실패: {e}")
        return []

def generate_llm_drift_explanation(dataset_name: str) -> str:
    """Ollama LLM을 활용한 드리프트 결과 해설 생성"""
    
    # 1. 현재 드리프트 결과 수집
    drift_summary = st.session_state.get('drift_score_summary', '')
    
    # 임베딩 정보 수집 (utils.py 의존성 제거)
    embedding_info = {}
    for key in ['train_embeddings', 'valid_embeddings', 'test_embeddings']:
        embeddings = st.session_state.get(key)
        if embeddings is not None:
            embedding_info[key] = embeddings.shape
    
    pca_dim = st.session_state.get('pca_selected_dim')
    if pca_dim:
        embedding_info['pca_selected_dim'] = pca_dim
    
    if not drift_summary:
        return ''
    
    # 2. 검색 쿼리 구성
    search_queries = [
        f"MMD Wasserstein KL divergence interpretation",
        f"data drift analysis embedding",
        "drift detection threshold explanation",
        f"PCA dimension reduction drift analysis"
    ]
    
    # 3. 관련 지식 검색
    knowledge_context = []
    for query in search_queries:
        relevant_chunks = search_drift_knowledge(query, top_k=2)
        knowledge_context.extend(relevant_chunks)
    
    # 중복 제거 및 길이 제한
    knowledge_context = list(set(knowledge_context))[:3]  # 상위 3개만 사용
    
    # 4. LLM 프롬프트 구성
    context_text = "\n\n".join(knowledge_context)
    
    embedding_info_text = ""
    for key, shape in embedding_info.items():
        if key == 'pca_selected_dim':
            embedding_info_text += f"PCA 차원: {shape}\n"
        else:
            embedding_info_text += f"{key}: {shape}\n"
    
    # 5. Ollama용 프롬프트 템플릿
    prompt_template = PromptTemplate(
        input_variables=["dataset_name", "embedding_info", "drift_summary", "context"],
        template="""당신은 데이터 드리프트 분석 전문가입니다. 다음 정보를 바탕으로 드리프트 분석 결과를 이해하기 쉽게 해설해주세요.

                    데이터셋 정보:
                    - 이름: {dataset_name}
                    - {embedding_info}

                    드리프트 분석 결과:
                    {drift_summary}

                    참고 지식:
                    {context}

                    다음 순서로 한국어로 설명해주세요:
                    1. 각 드리프트 메트릭의 의미와 현재 수치 해석
                    2. 전체적인 드리프트 상황 평가
                    3. 권장사항

                    500자 내외로 간결하게 작성해주세요."""
    )
    
    # 6. Ollama LLM 호출
    try:
        vectorstore, llm = load_drift_knowledge_base()
        
        if not llm:
            return generate_fallback_explanation()
        
        # 프롬프트 생성
        formatted_prompt = prompt_template.format(
            dataset_name=dataset_name,
            embedding_info=embedding_info_text,
            drift_summary=drift_summary,
            context=context_text
        )
        
        # LLM 호출
        explanation = llm.invoke(formatted_prompt)
        
        # HTML 형식으로 포맷팅
        return f"""
        <div class="comment-box" style="background-color: #e8f5e8; border-left: 4px solid #28a745;">
            <h4>🤖 AI 드리프트 분석 해설</h4>
            <p style="margin-top: 10px; line-height: 1.6;">{explanation}</p>
            <small style="color: #6c757d; font-style: italic;">
                * 이 해설은 로컬 AI(Ollama)가 기술 문서를 바탕으로 생성한 것입니다.
            </small>
        </div>
        """
        
    except Exception as e:
        st.warning(f"AI 해설 생성 중 오류: {e}")
        return generate_fallback_explanation()

def generate_fallback_explanation() -> str:
    """LLM 호출 실패 시 기본 해설 제공"""
    return f"""
    <div class="comment-box" style="background-color: #fff3cd; border-left: 4px solid #ffc107;">
        <h4>⚠️ AI 해설 생성 불가</h4>
        <p>현재 AI 해설 서비스(Ollama)를 이용할 수 없습니다. 다음 기본 가이드를 참고해주세요:</p>
        <ul>
            <li><strong>MMD, Wasserstein Distance</strong>: 낮을수록 두 데이터셋이 유사함</li>
            <li><strong>KL Divergence, JensenShannon</strong>: 0에 가까울수록 분포가 유사함</li>
            <li><strong>drift = False</strong>: 데이터 분포가 안정적</li>
            <li><strong>drift = True</strong>: 데이터 분포 변화 감지됨</li>
        </ul>
    </div>
    """