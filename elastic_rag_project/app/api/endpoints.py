from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.schemas import QueryRequest, QueryResponse
from services.rag_chain import get_rag_chain
from services.document_loader import add_test_data as add_test_data_to_elasticsearch
from services.vector_store import get_vector_store

router = APIRouter()

@router.post("/chat") # , response_model=QueryResponse
async def chat_endpoint(request: QueryRequest):
    rag_chain = get_rag_chain()
    answer = rag_chain.run(request.question)

    # return JSONResponse(content={"response":QueryResponse(answer=answer)})
    return {"response": answer}


@router.post("/add_test_data")
def add_test_data():
    vector_store = get_vector_store()
    add_test_data_to_elasticsearch(vector_store)


