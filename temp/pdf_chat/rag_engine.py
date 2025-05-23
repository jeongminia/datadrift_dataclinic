# rag_engine.py
import os
# pdf loader + split
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs embedding and vectorstore
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# retriever
# LLM 연결, QA chain 구성
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
#from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain

def process_pdf(pdf_path: str, milvus_host="localhost", milvus_port="19530"):
    # 1. PDF → 문서 추출
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                              chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # 3. 임베딩 및 벡터 DB 저장
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. LLM 구성
    llm = Ollama(
        model="yi:34b-chat",
        temperature=0.7,
        top_p=0.9,
        verbose=True
    )

    # 5. 프롬프트 템플릿 정의
    my_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=(
            "너는 전문적인 PDF 문서 Q&A 챗봇이야.\n"
            "아래 문서 내용을 바탕으로 사용자의 질문에 대해 반드시 한국어로, 정확하고 논리적으로, 근거를 들어 상세하게 답변해줘.\n"
            "영어, 일본어, 중국어 등 다른 언어를 절대 사용하지 말고, 반드시 한국어로만 답변해.\n"
            "가능하다면 문서의 근거 문장도 함께 인용해줘.\n"
            "답변이 짧거나 불명확하면 안 되고, 친절하고 자세하게 설명해줘.\n"
            "문서 내용: {context}\n"
            "대화 기록: {chat_history}\n"
            "질문: {question}\n"
            "답변:"
        )
    )

    # 6. QA 체인 구성 
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": my_prompt}  # 프롬프트 명시
    )

    return qa_chain, len(chunks)
