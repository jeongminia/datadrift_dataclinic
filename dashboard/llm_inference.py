import os
import streamlit as st
from streamlit_chat import message
from rag_engine import build_vector_db, load_qa_chain

st.title("PDF Chatbot for Data Drift")

# PDF DB 경로
PDF_DIR = "./pdf_db"
DB_PATH = "faiss_db"

# 최초 1회만 벡터DB 생성 (이미 있으면 생략)
if not os.path.exists(DB_PATH):
    build_vector_db(PDF_DIR, DB_PATH)

st.write("아래에서 LLM 모델을 선택하세요.")

model_options = [
    "모델을 선택하세요",
    "exaone3.5:7.8b",
    "huihui_ai/exaone3.5-abliterated:7.8b",
    "joonoh/HyperCLOVAX-SEED-Text-Instruct-1.5B:latest",
]

selected_model = st.selectbox(
   "⏩ 답변에 사용할 LLM 모델을 선택하세요.",
    model_options,
    key="model_select"
)

# QA 체인 세션에 저장
if "qa" not in st.session_state or st.session_state.get("selected_model") != selected_model:
    with st.spinner("🔄 벡터DB 및 LLM 로딩 중..."):
        st.session_state.qa = load_qa_chain(DB_PATH, model_name=selected_model)
        st.session_state["selected_model"] = selected_model
        st.session_state['history'] = [
            {"role": "user", "content": "PDF Ready!"},
            {"role": "assistant", "content": "안녕하세요! PDF에서 궁금한 내용을 물어보세요 😊"}
        ]

# 응답 출력
with st.container():
    for i, (user_msg, bot_msg) in enumerate(zip(st.session_state['history'][::2], st.session_state['history'][1::2])):
        message(user_msg['content'], is_user=True, key=f"user_{i}")
        message(bot_msg['content'], key=f"bot_{i}")

user_input = st.text_input("Ask a question about the PDF content:", key="user_input")
if st.button("Send") and user_input:
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