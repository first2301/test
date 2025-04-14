import glob
from fastapi import APIRouter, HTTPException
from models.request_model import QueryRequest
from services.bllossom_service import ChatbotService
from services.rag_multisource_chatbot import RAGMultiSourceChatbotService

router = APIRouter()

@router.post("/generate")# generate
async def generate_text(request: QueryRequest):
    """프롬프트 받아서 응답 생성"""
    chatbot = ChatbotService()
    output = chatbot.chat(request.prompt)

    return {"response": output}


CSV_PATH = glob.glob('../data/*.csv')

@router.post("/generate_rag")# generate_rag
async def generate_text(request: QueryRequest):
    """
    질문 받아서 LLM + 문서 기반으로 응답 생성
    """
    
    chatbot = RAGMultiSourceChatbotService(model_path="../ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf")
    chatbot.load_sources(source_path=CSV_PATH)
    output = chatbot.ask(request.prompt)
    return {"response": output}
