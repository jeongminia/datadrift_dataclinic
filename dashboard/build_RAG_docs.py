from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 문서 불러오기 → chunk 나누기 → 벡터화
texts = load_text_chunks_from_drift_docs()  # 사용자 정의 함수
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(texts, embedding_model)
