from langchain_community.document_loaders.csv_loader import CSVLoader
from services.vector_store import get_vector_store
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


def add_test_data(vector_store):
    from uuid import uuid4
    from langchain_core.documents import Document

    document_1 = Document(
        page_content="개인정보 보호법에 따라 개인정보 수집 시에는 반드시 이용자의 동의를 받아야 합니다.",
        metadata={"source": "법률"},
    )

    document_2 = Document(
        page_content="개인정보 유출 시에는 지체 없이 해당 사실을 이용자에게 알리고 대통령령으로 정하는 바에 따라 조치한 결과를 이용자에게 알려야 합니다.",
        metadata={"source": "법률"},
    )

    document_3 = Document(
        page_content="개인정보 보호책임자는 개인정보 처리에 관한 법률 및 이 법의 준비와 집행을 맡는 이사 또는 이와 동등한 임원으로 지정하여야 합니다.",
        metadata={"source": "법률"},
    )

    document_4 = Document(
        page_content="개인정보의 수집과 이용 목적이 달성되면 지체 없이 해당 개인정보를 파기해야 합니다.",
        metadata={"source": "법률"},
    )

    document_5 = Document(
        page_content="개인정보의 안전성 확보를 위해 암호화, 접근제한, 접속이력 관리 등의 기술적 조치를 취해야 합니다.",
        metadata={"source": "법률"},
    )

    document_6 = Document(
        page_content="개인정보 보호법 위반 시 5년 이하의 징역 또는 5천만원 이하의 벌금이 부과될 수 있습니다.",
        metadata={"source": "법률"},
    )

    document_7 = Document(
        page_content="개인정보의 수집과 이용에 대한 동의는 구체적이고 명확한 목적을 제시한 후 받아야 합니다.",
        metadata={"source": "법률"},
    )

    document_8 = Document(
        page_content="개인정보 보호법은 개인정보의 수집, 사용, 제공, 관리, 보호 등에 관한 사항을 규정합니다.",
        metadata={"source": "법률"},
    )

    document_9 = Document(
        page_content="개인정보 보호책임자는 개인정보 처리방침의 수립과 공개, 개인정보 보호 교육 등을 수행합니다.",
        metadata={"source": "법률"},
    )

    document_10 = Document(
        page_content="개인정보의 수집과 이용에 대한 동의는 이용자가 쉽게 인지하고 선택할 수 있도록 명확하게 표시해야 합니다.",
        metadata={"source": "법률"},
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)