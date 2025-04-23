from langchain_community.document_loaders.csv_loader import CSVLoader
from app.services.vector_store import get_vector_store
from typing import List

def load_csv_to_elasticsearch(file_path: str, index_name: str = "documents") -> List[str]:

    """
    주어진 CSV 파일을 로드하여 Elasticsearch에 벡터화된 문서로 저장합니다.

    Args:
        file_path (str): CSV 파일의 경로
        index_name (str): Elasticsearch 인덱스 이름 (기본값: "documents")

    Returns:
        List[str]: 저장된 문서의 ID 목록
    """
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    vector_store =  get_vector_store()

    ids = vector_store.add_documents(documents)
    return ids