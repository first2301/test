from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.chat_model import ChatRequest, ChatResponse
from services.select_model import get_chain_by_model
from services.llm_cpp_service import get_llama_cpp_chain
from langchain.memory import ConversationBufferMemory

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    # chain = get_chain_by_model(req.model, memory)
    chain = get_llama_cpp_chain(req.model, memory)
    answer = chain.run(req.question)
    # return ChatResponse(response=answer)
    # return {"response": answer}
    return JSONResponse(content={"response": answer})

# @router.post("/chat", response_model=ChatResponse)
# async def chat(req: ChatRequest):
#     return {"response": "Hello, World!"}