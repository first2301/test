from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.config import ELASTICSEARCH_URL, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_CACHE_DIR
import os

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        cache_folder=EMBEDDING_MODEL_CACHE_DIR,
        model_kwargs={"device": "cpu"},                    # GPU 사용 안함
        encode_kwargs={"normalize_embeddings": True}       # 임베딩 정규화 (권장)
    )
    return ElasticsearchStore(
        index_name="documents",
        embedding=embeddings,
        es_url=ELASTICSEARCH_URL,
        es_user="elastic",  # 보안이 비활성화된 경우
        es_password="elastic",  # 보안이 비활성화된 경우
    )