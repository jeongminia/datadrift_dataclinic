import os
import streamlit as st
from streamlit_chat import message
from rag_engine import process_pdf

st.title("PDF Chatbot for Data Drift")

# 세션 상태 초기화
if 'history' not in st.session_state:
    st.session_state['history'] = []

uploaded_file = st.file_uploader(" ", type=["pdf", "txt", "docx"])

model_options = ["모델을 선택하세요", 
                 "yi:34b-chat", "llama3", "mistral", "phi3", "exaone3.5:7.8b",
                 "bnksys/yanolja-eeve-korean-instruct-10.8b:latest",
                 "jinbora/deepseek-r1-Bllossom:8b", "granite3.3:8b", "ggml", "kollama"]
selected_model = st.radio(
    "⏩ 답변에 사용할 LLM 모델을 선택하세요.",
    model_options,
    key="model_select",
    horizontal=True
)

# 실제 모델 선택 시에만 동작하도록 분기
if uploaded_file is not None and selected_model != "모델을 선택하세요":
    max_size_mb = 1024
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        st.error(f"File size exceeds {max_size_mb}MB limit.")
        st.stop()

    st.success("😀 File uploaded successfully!")
    safe_filename = os.path.basename(uploaded_file.name).replace(" ", "_")
    temp_file_path = os.path.join("temp", safe_filename)
    os.makedirs("temp", exist_ok=True)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("💾 File saved to temp directory.")

    # 모델 또는 파일이 바뀌면 QA 체인 새로 생성
    if (
        "last_uploaded_file" not in st.session_state or
        st.session_state["last_uploaded_file"] != safe_filename or
        st.session_state.get("selected_model") != selected_model
    ):
        with st.spinner("📚 Loading PDF and initializing QA..."):
            qa_chain, n_chunks = process_pdf(temp_file_path, model_name=selected_model)
            st.session_state.qa = qa_chain
            st.session_state["last_uploaded_file"] = safe_filename
            st.session_state["selected_model"] = selected_model
            st.session_state['text_processed'] = True

            # 최초 1회 인사말 설정
            if 'initialized' not in st.session_state:
                st.session_state['history'] = [
                    {"role": "user", "content": "PDF Ready!"},
                    {"role": "assistant", "content": "안녕하세요! PDF에서 궁금한 내용을 물어보세요 😊"}
                ]
                st.session_state['initialized'] = True
                st.rerun()

# 응답 출력
with st.container():
    for i, (user_msg, bot_msg) in enumerate(zip(st.session_state['history'][::2], st.session_state['history'][1::2])):
        message(user_msg['content'], is_user=True, key=f"user_{i}")
        message(bot_msg['content'], key=f"bot_{i}")

# 사용자 입력창
is_text_processed = st.session_state.get('text_processed', False)
user_input = st.text_input("Ask a question about the PDF content:", key="user_input", disabled=not is_text_processed)

if st.button("Send", disabled=not is_text_processed) and user_input:
    st.session_state['history'].append({"role": "user", "content": user_input})

    chat_history = [
        (st.session_state['history'][i]["content"], st.session_state['history'][i+1]["content"])
        for i in range(0, len(st.session_state['history']) - 1, 2)
    ]

    with st.spinner("🤔 Generating response..."):
        result = st.session_state.qa.invoke({
            "question": user_input,
            "chat_history": chat_history
        })
        response = result['answer']

    st.session_state['history'].append({"role": "assistant", "content": response})

    st.rerun()
