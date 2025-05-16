import os
import streamlit as st
from streamlit_chat import message
# pdf loader + split
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# docs embedding and vectorstore
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus 
# retriever and reranking
from langchain.retrievers import MilvusRetriever
# LLM 연결, QA chain 구성
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

st.title("PDF Chatbot for Data Drift")
st.write("Upload a PDF file to chat with the content.")

# initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# upload PDF
uploaded_file = st.file_uploader(" ", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    # 파일 크기 제한 
    max_size_mb = 100
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        st.error(f"File size exceeds {max_size_mb}MB limit.")
        st.stop()

    st.write("😀 File uploaded successfully!")

    # 안전한 파일명 생성
    safe_filename = os.path.basename(uploaded_file.name).replace(" ", "_")
    temp_file_path = os.path.join("temp", safe_filename)
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("💾 File saved to temp directory.")

    # show a loading spinner while processing the file
    with st.spinner("Processing..."):
        # 분할
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        txt_split = RecursiveCharacterTextSplitter(  # Split the documents into chunks
            chunk_size=1000,
            chunk_overlap=200,
        )
        texts = txt_split.split_documents(documents)
        st.write(f"✔️ Loaded {len(texts)} chunks from the PDF file.")

        # 벡터 임베딩 ; Create embeddings and vectorstore
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Milvus.from_documents(texts, embedding_model, connection_args={"host": "localhost", "port": "19530"})
        st.write("Embeddings and vectorstore created.")

        # 재검색기 설정 ; reranker 없이 기본 retriever만 사용
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 랭체인 소환 ; LangChain용 LlamaCpp 모델 설정
        llm = LlamaCpp(
            # Path to the model file ; gguf 포맷 모델을 다운로드해서 직접 경로 지정
            model_path="/home/keti/datadrift_jm/models/gpt4all/ggml-model-Q4_K_M.gguf",
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            n_ctx=2048,
            n_batch=128,        # 생성 속도 향상 (VRAM 상황에 따라 조정)
            n_threads=8,        # CPU thread 수 (멀티코어 사용 시)
            n_gpu_layers=-1,    # 모든 레이어를 GPU로 올림
            verbose=True
            )

        # 세션 상태에 QA 체인이 없을 경우 초기화
        if "qa" not in st.session_state:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',  # or 'map_reduce', 'refine' if 문서가 길다면
                retriever=retriever,
            )
            st.session_state.qa = qa
        # PDF 처리 완료 플래그 설정
        st.session_state['text_processed'] = True

# chat interface
response_container = st.container()

# 텍스트 처리됐는지 체크
is_text_processed = st.session_state.get('text_processed', False)

with st.form(key='chat_form', clear_on_submit=True):
    # 사용자 입력
    user_input = st.text_input("Ask a question about the PDF content:", 
                               key="user_input",
                               disabled=not is_text_processed,)
    submit_button = st.form_submit_button(label='Send', 
                                          disabled=not is_text_processed)
    if submit_button and user_input:
        # 사용자 질문을 세션 상태에 저장
        st.session_state['history'].append({"role": "user", "content": user_input})
        
        # LLM에 질문하고 답변 받기
        with st.spinner("Generating response..."):
            response = st.session_state.qa.invoke({"query": user_input})['result']
        
        # 답변을 세션 상태에 저장
        st.session_state['history'].append({"role": "assistant", "content": response})
        

with response_container:
    for i, (user_msg, bot_msg) in enumerate(zip(st.session_state['history'][::2], st.session_state['history'][1::2])):
        message(user_msg['content'], is_user=True, key=f"user_{i}")
        message(bot_msg['content'], key=f"bot_{i}")