import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# 1. PDF 폴더 내 모든 PDF를 벡터DB로 저장
def build_vector_db(pdf_dir, db_path="faiss_db", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    all_chunks = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, fname))
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(all_chunks, embedding)
    vectorstore.save_local(db_path)
    print(f"✅ {len(all_chunks)}개 청크를 {db_path}에 저장 완료")
    return db_path

# 2. DB에서 QA 체인 생성
def load_qa_chain(db_path, model_name):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = Ollama(
        model=model_name,
        temperature=0.7,
        top_p=0.9,
        verbose=True
    )

    my_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=(
            "너는 전문적인 PDF 문서 Q&A 챗봇이야.\n"
            "아래 문서 내용을 바탕으로 사용자의 질문에 대해 반드시 한국어로, 정확하고 논리적으로, 근거를 들어 상세하게 답변해줘.\n"
            "영어, 일본어, 중국어 등 다른 언어를 절대 사용하지 말고, 반드시 한국어로만 답변해.\n"
            "답변이 짧거나 불명확하면 안 되고, 친절하고 명료하게 설명해줘.\n"
            "문서 내용: {context}\n"
            "대화 기록: {chat_history}\n"
            "질문: {question}\n"
            "답변:"
        )
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": my_prompt}
    )
    return qa_chain