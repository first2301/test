from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings
from utils.config import ELASTICSEARCH_URL, EMBEDDING_MODEL_PATH

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    return ElasticsearchStore(
        index_name="documents",
        embedding=embeddings,
        es_url=ELASTICSEARCH_URL
    )