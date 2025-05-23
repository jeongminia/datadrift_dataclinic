import os
import streamlit as st
from streamlit_chat import message
from rag_engine import process_pdf

st.title("PDF Chatbot for Data Drift")

# 세션 상태 초기화
if 'history' not in st.session_state:
    st.session_state['history'] = []

uploaded_file = st.file_uploader(" ", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
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

    if (
    "last_uploaded_file" not in st.session_state or
    st.session_state["last_uploaded_file"] != safe_filename
    ):
        with st.spinner("📚 Loading PDF and initializing QA..."):
            qa_chain, n_chunks = process_pdf(temp_file_path)
            st.session_state.qa = qa_chain
            st.session_state["last_uploaded_file"] = safe_filename
            st.session_state['text_processed'] = True

            # 최초 1회 인사말 설정
            if 'initialized' not in st.session_state:
                st.session_state['history'] = [
                {"role": "user", "content": "PDF Ready!"},
                {"role": "assistant", "content": "안녕하세요! PDF에서 궁금한 내용을 물어보세요 😊"}
                ]
                st.session_state['initialized'] = True  # ✅ rerun 이후 다시 안 들어오게 막음
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
