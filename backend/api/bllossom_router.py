import glob
from fastapi import APIRouter, HTTPException
from models.request_model import QueryRequest
from services.bllossom_service import ChatbotService
from services.rag_multisource_chatbot import RAGMultiSourceChatbotService
from services.rag_service import RagChatbotService

router = APIRouter()

LLM_MODEL_PATH = "../ai_models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf"
EMBEDDING_MODEL_PATH = "../ai_models/intfloat/multilingual-e5-large-instruct"  # 또는 로컬 모델 경로

@router.post("/generate")# generate
async def generate_text(request: QueryRequest):
    """프롬프트 받아서 응답 생성"""
    chatbot = ChatbotService()
    output = chatbot.chat(request.prompt)

    return {"response": output}


@router.post("/generate_rag")# generate_rag_url
async def generate_text(request: QueryRequest):
    """
    질문 받아서 LLM + 문서 기반으로 응답 생성
    """
    chatbot = RagChatbotService(
        model_path=LLM_MODEL_PATH,
        embedding_model_path=EMBEDDING_MODEL_PATH  # 또는 로컬 모델 경로
    )

    # CSV 파일에서 벡터스토어 생성
    csv_path = "../data/한국인터넷진흥원_개인정보포털 상황별 FAQ정보_20240731.csv"
    chatbot.build_vectorstore_from_csv(csv_path)

    # RAG 체인 구성
    chatbot.build_rag_chain()

    output = chatbot.chat(request.prompt)
    
    return {"response": output}


# CSV_PATH = glob.glob('../data/*.csv')
# @router.post("/generate_rag")# generate_rag
# async def generate_text(request: QueryRequest):
#     """
#     질문 받아서 LLM + 문서 기반으로 응답 생성
#     """
    
#     chatbot = RAGMultiSourceChatbotService(model_path=MODEL_PATH)
#     chatbot.load_sources(source_path=CSV_PATH)
#     output = chatbot.ask(request.prompt)
#     return {"response": output}

