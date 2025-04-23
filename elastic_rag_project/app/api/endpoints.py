from fastapi import APIRouter
from models.schemas import QueryRequest, QueryResponse
from services.rag_chain import get_rag_chain

router = APIRouter()

@router.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    rag_chain = get_rag_chain()
    answer = rag_chain.run(request.question)
    return QueryResponse(answer=answer)

