import streamlit as st
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
# data & model load, Embedding & visualization
from utils import visualize_similarity_distance, plot_reduced

warnings.filterwarnings("ignore")

def render():
    if 'embedding_data' not in st.session_state or not st.session_state['embedding_data']:
        st.error("❌ 'embedding_data' is not initialized or empty. Please load it from VectorDB.")
        return

    embedding_data = st.session_state['embedding_data']
    dataset_name = st.session_state.get('dataset_name', 'Dataset')
    st.title(f"Embedding Visualization Page of {dataset_name}")

    # 데이터셋 분리 함수
    def extract_embeddings(key):
        return np.array([
            res["vector"] for res in embedding_data if res.get("set_type", "").lower() == key
        ], dtype=np.float32)

    train_embeddings = extract_embeddings("train")
    valid_embeddings = extract_embeddings("valid")
    test_embeddings = extract_embeddings("test")

    # 세션에 저장
    st.session_state['train_embeddings'] = train_embeddings
    st.session_state['valid_embeddings'] = valid_embeddings
    st.session_state['test_embeddings'] = test_embeddings
    st.session_state['embedding_overview_text'] = (
        f"Train: {train_embeddings.shape}, "
        f"Valid: {valid_embeddings.shape}, "
        f"Test: {test_embeddings.shape}"
    )
    st.session_state['dataset_summary'] = (
        f"Train: {train_embeddings.shape}, "
        f"Valid: {valid_embeddings.shape}, "
        f"Test: {test_embeddings.shape}"
    )

    # 정보 출력
    st.write(st.session_state['embedding_overview_text'])

    if train_embeddings.size == 0 or valid_embeddings.size == 0 or test_embeddings.size == 0:
        st.error("One or more embedding datasets are empty. Please check the data.")
        return

    # 🔷 Original Dimension 시각화
    st.subheader("Original Dimension")
    try:
        fig_dist = visualize_similarity_distance(valid_embeddings, test_embeddings, train_embeddings)
        if fig_dist is not None:
            st.session_state['embedding_distance_fig'] = fig_dist

            buf_dist = io.BytesIO()
            fig_dist.savefig(buf_dist, format="png")
            buf_dist.seek(0)
            st.session_state['embedding_distance_img'] = buf_dist

            st.pyplot(fig_dist)
        else:
            st.warning("`visualize_similarity_distance` did not return a figure.")
    except Exception as e:
        st.error(f"Error in visualizing similarity distance: {e}")

    # 🔷 PCA 시각화
    st.subheader("Dimension Reduction with PCA")
    dim_option = st.selectbox("Select Size of Dimension", [10, 50, 100, 200, 300, 400, 500])

    st.session_state['pca_selected_dim'] = dim_option

    @st.cache_data
    def apply_pca(embeddings, n_components):
        return PCA(n_components=n_components).fit_transform(embeddings)

    train_pca = apply_pca(train_embeddings, dim_option)
    valid_pca = apply_pca(valid_embeddings, dim_option)
    test_pca = apply_pca(test_embeddings, dim_option)

    st.write(f"Train PCA shape: {train_pca.shape}")
    st.write(f"Validation PCA shape: {valid_pca.shape}")
    st.write(f"Test PCA shape: {test_pca.shape}")

    st.markdown(f"<b>PCA Reduced Dimension:</b> {dim_option}", unsafe_allow_html=True)

    try:
        fig_pca_dist = visualize_similarity_distance(valid_pca, test_pca, train_pca)
        if fig_pca_dist is not None:
            st.session_state['embedding_pca_distance_fig'] = fig_pca_dist
            st.pyplot(fig_pca_dist)

            buf_pca_dist = io.BytesIO()
            fig_pca_dist.savefig(buf_pca_dist, format="png")
            buf_pca_dist.seek(0)
            st.session_state['embedding_pca_distance_img'] = buf_pca_dist

        else:
            st.warning("`visualize_similarity_distance` did not return a figure (PCA).")

        fig_pca_plot = plot_reduced(valid_pca, test_pca, train_pca)
        st.session_state['embedding_pca_fig'] = fig_pca_plot

        buf_pca = io.BytesIO()
        fig_pca_plot.savefig(buf_pca, format="png")
        buf_pca.seek(0)
        st.session_state['embedding_pca_img'] = buf_pca

        st.pyplot(fig_pca_plot)
    except Exception as e:
        st.error(f"Error in PCA-based visualization: {e}")

#    st.write("Debug Info: ", st.session_state)