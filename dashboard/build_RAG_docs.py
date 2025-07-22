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
        llm = OllamaLLM(
            model="huihui_ai/exaone3.5-abliterated:7.8b",
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
    
    # 2. 강화된 임베딩 정보 수집
    embedding_info = {}
    pca_analysis = {}
    
    # 기본 임베딩 정보
    for key in ['train_embeddings', 'valid_embeddings', 'test_embeddings']:
        embeddings = st.session_state.get(key)
        if embeddings is not None:
            embedding_info[key] = {
                'shape': embeddings.shape,
                'mean': float(embeddings.mean()) if hasattr(embeddings, 'mean') else 'N/A',
                'std': float(embeddings.std()) if hasattr(embeddings, 'std') else 'N/A'
            }
    
    # PCA 분석 정보 강화
    pca_dim = st.session_state.get('pca_selected_dim')
    if pca_dim:
        pca_analysis['selected_dimension'] = pca_dim
        
        # PCA 좌표 분석 (가능한 경우)
        train_pca = st.session_state.get('train_embeddings_pca')
        test_pca = st.session_state.get('test_embeddings_pca')
        
        if train_pca is not None and test_pca is not None:
            import numpy as np
            train_center = np.mean(train_pca, axis=0)
            test_center = np.mean(test_pca, axis=0)
            center_distance = np.linalg.norm(train_center - test_center)
            
            pca_analysis.update({
                'train_center': f"({train_center[0]:.3f}, {train_center[1]:.3f})",
                'test_center': f"({test_center[0]:.3f}, {test_center[1]:.3f})",
                'center_distance': f"{center_distance:.3f}"
            })
    
    if not drift_summary:
        return ''
    # 3. 드리프트 메트릭 상세 분석
    drift_metrics_analysis = {}
    if drift_summary:
        lines = drift_summary.split('\n')
        for line in lines:
            if 'score =' in line and 'drift =' in line:
                # 메트릭별 점수 추출
                parts = line.split(':')
                if len(parts) >= 2:
                    metric_name = parts[0].strip('- ')
                    score_part = parts[1].split(',')[0].replace('score =', '').strip()
                    drift_part = parts[1].split(',')[1].replace('drift =', '').strip()
                    
                    try:
                        score_value = float(score_part)
                        drift_metrics_analysis[metric_name] = {
                            'score': score_value,
                            'drift_detected': drift_part,
                            'threshold_level': 'low' if score_value < 0.01 else 'medium' if score_value < 0.05 else 'high'
                        }
                    except:
                        pass
    
    # 4. 검색 쿼리 구성
    search_queries = [
        f"MMD Wasserstein KL divergence interpretation",
        f"data drift analysis embedding",
        "drift detection threshold explanation",
        f"PCA dimension reduction drift analysis"
    ]
    
    # 5. 관련 지식 검색
    knowledge_context = []
    for query in search_queries:
        relevant_chunks = search_drift_knowledge(query, top_k=2)
        knowledge_context.extend(relevant_chunks)
    
    # 중복 제거 및 길이 제한
    knowledge_context = list(set(knowledge_context))[:3]  # 상위 3개만 사용
    
    # 6. 구조화된 정보 구성
    context_text = "\n\n".join(knowledge_context)
    
    # 임베딩 정보 텍스트 구성
    embedding_info_text = ""
    for key, info in embedding_info.items():
        if isinstance(info, dict):
            embedding_info_text += f"{key}: shape={info['shape']}, mean={info['mean']}, std={info['std']}\n"
        else:
            embedding_info_text += f"{key}: {info}\n"
    
    # PCA 분석 텍스트 구성
    pca_info_text = ""
    if pca_analysis:
        for key, value in pca_analysis.items():
            pca_info_text += f"{key}: {value}\n"
    
    # 메트릭 분석 텍스트 구성
    metrics_info_text = ""
    for metric, analysis in drift_metrics_analysis.items():
        metrics_info_text += f"{metric}: score={analysis['score']}, level={analysis['threshold_level']}, drift={analysis['drift_detected']}\n"
    
    # 7. 개선된 프롬프트 템플릿 (4단계 구조)
    prompt_template = PromptTemplate(
        input_variables=["dataset_name", "embedding_info", "drift_summary", "context"],
        template="""당신은 데이터 드리프트 분석 전문가입니다. 다음 정보를 바탕으로 4단계 구조화된 드리프트 분석 해설을 작성해주세요.

            데이터셋 정보:
            - 이름: {dataset_name}
            - 임베딩 정보: {embedding_info}
            - 드리프트 분석 결과: {drift_summary}

            참고 지식:
            {context}

            아래와 같이 4단계 구조로 한국어로 작성해주세요:

            [기술적 분석] 각 드리프트 메트릭의 수치적 의미와 임계값 대비 해석을 제시합니다. MMD, Wasserstein, KL Divergence 등의 점수를 구체적으로 분석하세요.

            [현 상황 분석] 현재 드리프트 상황이 모델 성능과 서비스 안정성에 미칠 영향을 평가합니다. 재학습 필요성과 위험도를 판단하세요.

            [시각적 분석] PCA 시각화 결과와 연계하여 데이터 분포 변화를 해석합니다. 가능하다면 중심점 이동과 분포 겹침 정도를 언급하세요.

            [권장사항] 즉시 조치사항과 모니터링 방안을 제시합니다. 구체적인 임계값과 실행 계획을 포함하세요.

            각 단계는 반드시 대괄호로 시작하며, 2-3문장으로 간결하게 작성하고, 전체 500자 내외로 제한하세요.
    """
    )
    
    # RAG Input Variables를 HTML로 포맷팅
    input_variables_html = f"""
    <div class="rag-input-variables" style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 20px 0;">
        <h4 style="color: #495057; margin-bottom: 15px;">🔍 RAG 분석 입력 변수</h4>
        
        <div style="margin-bottom: 15px;">
            <h5 style="color: #007bff; margin-bottom: 8px;">📊 Dataset Information</h5>
            <pre style="background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6; margin: 0;">{dataset_name}</pre>
        </div>
        
        <div style="margin-bottom: 15px;">
            <h5 style="color: #28a745; margin-bottom: 8px;">🎯 Embedding Information</h5>
            <pre style="background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6; margin: 0; max-height: 150px; overflow-y: auto;">{embedding_info_text}</pre>
        </div>
              
        <div style="margin-bottom: 15px;">
            <h5 style="color: #6f42c1; margin-bottom: 8px;">📋 Drift Summary</h5>
            <pre style="background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6; margin: 0; max-height: 120px; overflow-y: auto;">{drift_summary}</pre>
        </div>
    </div>
    """
    
    # 8. Ollama LLM 호출
    try:
        vectorstore, llm = load_drift_knowledge_base()
        
        # 프롬프트 생성
        formatted_prompt = prompt_template.format(
            dataset_name=dataset_name,
            embedding_info=embedding_info_text,
            pca_info=pca_info_text,
            metrics_info=metrics_info_text,
            drift_summary=drift_summary,
            context=context_text
        )
        
        # LLM 호출
        explanation = llm.invoke(formatted_prompt)
        
        # HTML 형식으로 포맷팅 (구조화된 스타일링)
        formatted_explanation = format_structured_explanation(explanation)
        
        return f"""
        {input_variables_html}
        <div class="comment-box" style="background-color: #e8f5e8; border-left: 4px solid #28a745; padding: 20px; margin: 10px 0;">
            <h4 style="margin-top: 0; color: #28a745;">🤖 AI 데이터 드리프트 분석 해설</h4>
            <div style="line-height: 1.8; font-size: 14px;">
                {formatted_explanation}
            </div>
            <small style="color: #6c757d; font-style: italic; margin-top: 15px; display: block;">
                * 이 해설은 로컬 AI(Ollama)가 기술 문서를 바탕으로 생성한 것입니다.
            </small>
        </div>
        """
        
    except Exception as e:
        st.warning(f"AI 해설 생성 중 오류: {e}")
        return 

def format_structured_explanation(explanation: str) -> str:
    """LLM 응답을 구조화된 HTML로 변환"""
    
    # 4단계 섹션 스타일 정의
    section_styles = {
        '기술적 분석': 'background-color: #f8f9fa; border-left: 3px solid #007bff; color: #0056b3;',
        '비즈니스 임팩트': 'background-color: #fff3cd; border-left: 3px solid #ffc107; color: #856404;',
        '시각적 분석': 'background-color: #d1ecf1; border-left: 3px solid #17a2b8; color: #0c5460;',
        '권장사항': 'background-color: #d4edda; border-left: 3px solid #28a745; color: #155724;'
    }
    
    # 대괄호로 시작하는 섹션 찾기
    pattern = r'\[([^\]]+)\]\s*(.*?)(?=\[|$)'
    matches = re.findall(pattern, explanation, re.DOTALL)
    
    if not matches:
        # 구조화되지 않은 응답인 경우 기본 포맷팅
        return explanation.replace('\n', '<br>')
    
    formatted_html = ""
    for section_title, content in matches:
        section_title = section_title.strip()
        content = content.strip()
        
        # 해당 섹션 스타일 가져오기
        style = section_styles.get(section_title, 'background-color: #f8f9fa; border-left: 3px solid #6c757d; color: #495057;')
        
        # HTML 섹션 구성
        formatted_html += f"""
        <div style="margin: 15px 0; padding: 15px; border-radius: 5px; {style}">
            <h5 style="margin: 0 0 10px 0; font-weight: bold;">📋 {section_title}</h5>
            <p style="margin: 0; line-height: 1.6;">{content}</p>
        </div>
        """
    
    return formatted_html