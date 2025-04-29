from fastapi import APIRouter
from models.schemas import QueryRequest, QueryResponse
from services.rag_chain import get_rag_chain
from services.document_loader import add_test_data as add_test_data_to_elasticsearch
from services.vector_store import get_vector_store
import time
from datetime import datetime, timedelta


router = APIRouter()

@router.post("/chat") # , response_model=QueryResponse
async def chat_endpoint(request: QueryRequest):
    
    start = datetime.now()
    rag_chain, retriever, llm = get_rag_chain()
    question = request.question
    try:
        docs = retriever.invoke(question)

        if docs:
            answer = rag_chain.invoke(question)
            
            answer = answer.get("result")
        else:
            fallback_prompt = f"'{question}'에 대해 네가 알고 있는 정보로 최대한 자세히 설명해줘"
            answer = llm.invoke(fallback_prompt)    
            end = datetime.now()
            # 경과 시간
            elapsed = end - start
            print(f"종료 시간: {elapsed.strftime('%H:%M:%S.%f')}")

        return {"response": answer}
    
    except Exception as e:
        print(f"에러발생: {str(e)}")
        return {"response": "❌ 답변을 생성할 수 없습니다."}


@router.post("/add_test_data")
def add_test_data():
    vector_store = get_vector_store()
    add_test_data_to_elasticsearch(vector_store)
    return {"message": "susscess!"}


