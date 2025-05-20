# rag_engine.py
import os
# pdf loader + split
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs embedding and vectorstore
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
# retriever
#from langchain.retrievers import MilvusRetriever
# LLM 연결, QA chain 구성
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain

def process_pdf(pdf_path: str, milvus_host="localhost", milvus_port="19530"):
    # 1. PDF → 문서 추출
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. 문서 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # 3. 임베딩 및 벡터 DB 저장
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Milvus.from_documents(chunks, embedding, connection_args={"host": milvus_host, "port": milvus_port})
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. LLM 구성
    llm = LlamaCpp(
        model_path="/home/keti/datadrift_jm/models/gpt4all/ggml-model-Q4_K_M.gguf",
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        n_ctx=2048,
        n_batch=128,
        n_threads=8,
        n_gpu_layers=-1,
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
