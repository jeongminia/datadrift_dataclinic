# rag_engine.py
import os
# pdf loader + split
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs embedding and vectorstore
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# retriever
# LLM 연결, QA chain 구성
from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama
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
    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0.7,
        top_p=0.9,
        verbose=True
    )

    # 5. QA 체인 구성
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        verbose=True
    )

    return qa_chain, len(chunks)
