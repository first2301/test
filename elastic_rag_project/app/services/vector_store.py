# from langchain_community.vectorstores import ElasticsearchStore
from langchain_elasticsearch import ElasticsearchStore
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import ELASTICSEARCH_URL, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_CACHE_DIR

def get_vector_store():
    """
    Elasticsearch 벡터 저장소를 생성하고 반환합니다.

    이 함수는 다음과 같은 작업을 수행합니다:
    1. HuggingFace 임베딩 모델을 초기화합니다.
    2. Elasticsearch 벡터 저장소를 설정합니다.
    3. 설정된 벡터 저장소를 반환합니다.

    Returns:
        ElasticsearchStore: 설정된 Elasticsearch 벡터 저장소 인스턴스
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        cache_folder=EMBEDDING_MODEL_CACHE_DIR,
        model_kwargs={"device": "cpu"},                    # GPU 사용 안함
        encode_kwargs={"normalize_embeddings": True}       # 임베딩 정규화 (권장)
    )

    elastic_vector_search = ElasticsearchStore(
        index_name="documents",
        embedding=embeddings,
        es_url=ELASTICSEARCH_URL,
        es_user="elastic",  # 보안이 비활성화된 경우
        es_password="elastic",  # 보안이 비활성화된 경우
    )

    return elastic_vector_search