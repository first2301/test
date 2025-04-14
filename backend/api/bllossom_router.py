from fastapi import APIRouter
from models.request_model import QueryRequest
from services.bllossom_service import ChatbotService

router = APIRouter()

@router.post("/generate")# generate
async def generate_text(request: QueryRequest):
    """프롬프트 받아서 응답 생성"""
    chatbot = ChatbotService()
    output = chatbot.chat(request.prompt)

    return {"response": output}